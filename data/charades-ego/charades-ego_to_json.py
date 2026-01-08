#!/usr/bin/env python3

"""
Convert Charades-Ego Dataset to Router Planner Format

This script converts the Charades-Ego dataset into the jsonl format used by LLMRouter.
It uses a VLM to describe a short time window of video frames (stitched as a filmstrip),
then builds a text query and evaluates multiple candidate LLMs to produce routing data.

Input: Charades-Ego dataset (CSV annotations + Video files)
Output: 
    - default_routing_train_data.jsonl
    - default_routing_test_data.jsonl
    - query_embeddings.pt
Note: outputs are written to the current working directory.

Usage:
    python charades-ego_to_json.py --data_root /path/to/data --sample_size 100 \
        --task_type activity --top_k 1 --num_frames 5
"""

import io
import base64
import os
import json
import argparse
import time
import random
import re
import pandas as pd
import numpy as np
import cv2
from PIL import Image

from llmrouter.utils import (
    setup_environment,
    process_final_data, 
    call_api, 
    calculate_task_performance,
    get_longformer_embedding,
)

# Setup environment
setup_environment()

# ============================================================================
# VLM Helper Functions
# ============================================================================

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def describe_image(image, model_name, vlm_config, vlm_prompt=None):
    """
    Generate description for a PIL Image using VLM.
    """
    if vlm_prompt is None:
        vlm_prompt = f"""You are given a filmstrip of frames sampled from a short time window of a video.
Return ONLY a JSON object with the following keys (no extra text):
{{
  "motion": string,
  "objects": [string],
  "summary": string,
}}
Definitions:
- motion: describe what the person is doing (hands/arms/body) in one short phrase.
- objects: list the main target objects the person is acting on / interacting with (not background clutter).
- summary: a short description of the scene in one sentence.
"""
    
    base64_image = encode_image_to_base64(image)
    
    content = [
        {"type": "text", "text": vlm_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
    
    api_endpoint = vlm_config["api_endpoint"]
    api_name = vlm_config["model"]
    service = vlm_config["service"]

    request = {
        "api_endpoint": api_endpoint,
        "query": content, 
        "model_name": model_name,
        "api_name": api_name,
        "service": service
    }
    
    result = call_api(request, max_tokens=1024, temperature=0.2)
    return result["response"]

# -----------------------------------------------------------------------------
# Charades(-Ego) annotation parsing helpers
# -----------------------------------------------------------------------------

def parse_action_segments(actions_value, classes):
    """
    Parse Charades(-Ego) `actions` field into time-stamped action segments.
    """
    segments = []
    if actions_value is None:
        return segments

    s = str(actions_value).strip()
    if not s or s.lower() == "nan":
        return segments

    for raw in s.split(";"):
        item = raw.strip()
        if not item:
            continue
        parts = item.split()
        if len(parts) < 3:
            continue

        cls_id = str(parts[0]).lower()
        if cls_id not in classes:
            continue

        try:
            start = float(parts[1])
            end = float(parts[2])
        except Exception:
            continue

        if end <= start:
            continue

        segments.append({
            "cls_id": cls_id,
            "label": classes[cls_id],
            "start": start,
            "end": end
        })

    return segments


def load_id_label_file(path: str, prefix: str) -> dict:
    """
    Load a mapping file with one entry per line: <id> <label...>
    Supports both prefixed ids (e.g. v03) and numeric ids (e.g. 3).
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            rid, label = parts
            # No normalization; use id as-is
            out[rid] = label.strip()
    return out

def load_action_to_verb_object(mapping_path: str) -> tuple[dict, dict]:
    """
    Load Charades action -> (object, verb) mapping from Charades_v1_mapping.txt.
    """
    action_to_verb: dict[str, str] = {}
    action_to_object: dict[str, str] = {}

    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            a, o, v = parts[0], parts[1], parts[2]
            if a and v:
                action_to_verb[a] = v
            if a and o:
                action_to_object[a] = o

    return action_to_verb, action_to_object

# -----------------------------------------------------------------------------
# Video frame sampling (time-windowed) + stitching
# -----------------------------------------------------------------------------

def extract_frames_in_time_window(video_path, start_sec, end_sec, num_frames):
    """
    Extract frames ONLY within [start_sec, end_sec] of a video.
    Output: List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []

    start_frame = int(max(0.0, start_sec) * fps)
    end_frame = int(max(0.0, end_sec) * fps)

    # Clamp and ensure at least 1 frame range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))
    if end_frame <= start_frame:
        end_frame = min(total_frames - 1, start_frame + max(1, num_frames))

    mid = int((start_frame + end_frame) / 2)
    window_frames = max(1, int(5.0 * fps))  # 5 seconds
    inner_start = max(0, mid - window_frames // 2)
    inner_end = min(total_frames - 1, inner_start + window_frames)

    indices = np.linspace(inner_start, inner_end, num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames

# -----------------------------------------------------------------------------
# Output parsing: extract label IDs (e.g., c### / v### / o###) from model outputs
# -----------------------------------------------------------------------------

def extract_ids(raw_response: str, valid_ids: list, top_k: int):
    """
    Extract up to top_k ids like c### / v### / o### from model output.
    Returns a de-duplicated list in appearance order.
    """
    valid_set = set(valid_ids or [])
    if not isinstance(raw_response, str) or top_k <= 0:
        return []

    found = []
    seen = set()
    for m in re.finditer(r"\b([cvoCVO]\d{1,4})\b", raw_response.strip()):
        rid = m.group(1).lower()
        if rid in valid_set and rid not in seen:
            found.append(rid)
            seen.add(rid)
            if len(found) >= top_k:
                break
    return found

def stitch_images(images):
    """
    Horizontally stitch a list of PIL images into a single filmstrip image.
    """
    target_height = 360
    resized_images = []
    for img in images:
        aspect = img.width / img.height
        new_w = int(target_height * aspect)
        resized_images.append(img.resize((new_w, target_height)))
            
    widths, heights = zip(*(i.size for i in resized_images))
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in resized_images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width
    return new_im

# ============================================================================
# Main Conversion Logic
# ============================================================================

def convert_charades_ego(data_root, vlm_name, sample_size, task_type, top_k, num_frames):
    print("=== CONVERTING CHARADES-EGO DATASET ===")
    
    # Paths
    charades_meta_dir = os.path.join(data_root, "CharadesEgo", "CharadesEgo")
    
    csv_ego_path = os.path.join(charades_meta_dir, "CharadesEgo_v1_test_only1st.csv")
    csv_exo_path = os.path.join(charades_meta_dir, "CharadesEgo_v1_test_only3rd.csv")
    classes_path = os.path.join(charades_meta_dir, "Charades_v1_classes.txt")
    mapping_path = os.path.join(charades_meta_dir, "Charades_v1_mapping.txt")
    verbclasses_path = os.path.join(charades_meta_dir, "Charades_v1_verbclasses.txt")
    objectclasses_path = os.path.join(charades_meta_dir, "Charades_v1_objectclasses.txt")
    video_root = os.path.join(data_root, "CharadesEgo", "CharadesEgo_v1_480")
    if not os.path.exists(video_root):
        # Fallback to CharadesEgo_v1 (original resolution)
        video_root = os.path.join(data_root, "CharadesEgo", "CharadesEgo_v1")
    
    if not os.path.exists(video_root):
        print(f"Warning: Video directory not found at {video_root} or ..._480 version")

    if not os.path.exists(csv_ego_path) or not os.path.exists(csv_exo_path):
        raise FileNotFoundError(f"Annotations file not found at {csv_ego_path} or {csv_exo_path}")

    # Load Classes
    classes = {}
    with open(classes_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                classes[parts[0]] = parts[1]
    
    # Normalize class ids to lowercase for consistent handling
    classes = {str(k).lower(): v for k, v in classes.items()}

    # Load verb/object label spaces + action->verb/object mapping
    verb_id_to_label = load_id_label_file(verbclasses_path, prefix="v")
    obj_id_to_label = load_id_label_file(objectclasses_path, prefix="o")
    action_to_verb, action_to_object = load_action_to_verb_object(mapping_path)

    # --------------------------------------------------------------------
    # Build class id <-> label mapping
    # --------------------------------------------------------------------
    activity_ids_sorted = sorted(
        list(classes.keys()),
        key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 10**9
    )
    verb_ids_sorted = sorted(verb_id_to_label.keys(), key=lambda x: int(re.findall(r"\d+", x)[0]))
    obj_ids_sorted = sorted(obj_id_to_label.keys(), key=lambda x: int(re.findall(r"\d+", x)[0]))

    if task_type == "activity":
        task_name = "charades-ego_activity"
        choices = activity_ids_sorted
        id_to_label = classes
        mapping_title = "Activity id mapping (id: name):"
        example_id = "c104"
    elif task_type == "verb":
        task_name = "charades-ego_verb"
        choices = verb_ids_sorted
        id_to_label = verb_id_to_label
        mapping_title = "Verb id mapping (id: name):"
        example_id = "v003"
    elif task_type == "object":
        task_name = "charades-ego_object"
        choices = obj_ids_sorted
        id_to_label = obj_id_to_label
        mapping_title = "Object id mapping (id: name):"
        example_id = "o012"
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use one of: activity, verb, object")

    mapping_lines = "\n".join([f"{cid}: {id_to_label.get(cid, '')}" for cid in choices])
    system_prompt = f"""You are a strict classifier for Charades(-Ego).
Task: output the top-{top_k} most likely ids from the mapping below (example: {example_id}).

Output format:
- Output ONLY ids separated by commas. No spaces. No extra words.
- Output exactly {top_k} ids.

How to use the VLM JSON:
- Focus on the person's motion and the main target objects they act on.
- Ignore static background items unless they are being used.
- Prefer the MOST specific label supported by the evidence.
- If the VLM summary mentions multiple possibilities, pick the single best-supported id.

{mapping_title}
{mapping_lines}
"""

    # Load VLM Config
    vlm_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_vlm.json")
    with open(vlm_config_path, 'r') as f:
        all_vlms = json.load(f)
        vlm_config = all_vlms[vlm_name]


    print(f"Using VLM API Model: {vlm_config['model']}")

    # Load CSVs
    df_ego = pd.read_csv(csv_ego_path)
    df_exo = pd.read_csv(csv_exo_path)

    df_ego["base_id"] = df_ego["id"].apply(lambda vid: vid[:-3] if vid.endswith("EGO") else vid)
    df_exo["base_id"] = df_exo["id"]

    # Keep only paired samples (both views exist). User confirmed they always exist.
    pairs = pd.merge(df_ego, df_exo, on="base_id", suffixes=("_ego", "_exo"), how="inner")
    pairs = pairs.sample(n=sample_size, replace=True, random_state=None)
    print(f"Sampled {len(pairs)} paired samples (scenes).")

    converted_data = []

    # Load LLM Candidates
    llm_candidates_path = "../example_data/llm_candidates/default_llm.json"
    with open(llm_candidates_path, 'r') as f:
        llm_candidates = json.load(f)

    for idx, row in enumerate(pairs.to_dict("records")):
        # In the paired CSVs, both views exist; we only simulate missing modalities.
        ego_video_id = row.get("id_ego")
        exo_video_id = row.get("id_exo")
        base_id = row.get("base_id")

        # Randomly simulate missing modality but ensure at least one view exists
        selected_mode = random.choice(["ego", "exo", "both"])
        
        # --------------------------------------------------------------------
        # Single-label ground truth via time-stamped action segment
        # --------------------------------------------------------------------
        ego_segments = parse_action_segments(row.get("actions_ego"), classes)
        exo_segments = parse_action_segments(row.get("actions_exo"), classes)
        if not ego_segments or not exo_segments:
            continue

        # Align segments across views: same class id and midpoint time difference < 1 second
        exo_by_cls = {}
        for s in exo_segments:
            exo_by_cls.setdefault(s["cls_id"], []).append(s)

        candidates = []
        for s_ego in ego_segments:
            mid_ego = (float(s_ego["start"]) + float(s_ego["end"])) / 2.0
            for s_exo in exo_by_cls.get(s_ego["cls_id"], []):
                mid_exo = (float(s_exo["start"]) + float(s_exo["end"])) / 2.0
                diff = abs(mid_ego - mid_exo)
                if diff <= 1.0:
                    candidates.append((diff, s_ego, s_exo))

        if not candidates:
            continue

        # Choose the best-aligned pair (smallest time diff)
        _, selected_ego_seg, selected_exo_seg = min(candidates, key=lambda x: x[0])

        selected_cls_id = selected_ego_seg["cls_id"]
        activity_id = str(selected_cls_id).lower()
        activity_label = selected_ego_seg["label"]

        if task_type == "activity":
            ground_truth = activity_id
            ground_truth_label = activity_label
        elif task_type == "verb":
            verb_id = action_to_verb[activity_id]
            ground_truth = verb_id
            ground_truth_label = verb_id_to_label[verb_id]
        elif task_type == "object":
            obj_id = action_to_object[activity_id]
            ground_truth = obj_id
            ground_truth_label = obj_id_to_label[obj_id]

        ego_start = float(selected_ego_seg["start"])
        ego_end = float(selected_ego_seg["end"])
        exo_start = float(selected_exo_seg["start"])
        exo_end = float(selected_exo_seg["end"])

        # For bookkeeping in task_id (used by notebook): use the selected mode's time window
        if selected_mode == "ego":
            action_start, action_end = ego_start, ego_end
        elif selected_mode == "exo":
            action_start, action_end = exo_start, exo_end
        else:
            action_start = (ego_start + exo_start) / 2.0
            action_end = (ego_end + exo_end) / 2.0

        # Helper to process a view
        def process_view(video_id_for_view, start_sec, end_sec):
            vpath = os.path.join(video_root, f"{video_id_for_view}.mp4")
            frames = extract_frames_in_time_window(
                vpath,
                start_sec=start_sec,
                end_sec=end_sec,
                num_frames=num_frames
            )
            stitched_img = stitch_images(frames)
            return describe_image(stitched_img, vlm_name, vlm_config)

        # Generate descriptions
        ego_desc = "Not provided"
        exo_desc = "Not provided"

        if selected_mode == 'ego' or selected_mode == 'both':
            ego_desc = process_view(ego_video_id, ego_start, ego_end)
        
        if selected_mode == 'exo' or selected_mode == 'both':
            exo_desc = process_view(exo_video_id, exo_start, exo_end)

        # Construct prompt
        query_prompt = f"""{system_prompt}
Determine the correct id using the structured VLM outputs below.

First-person view VLM JSON: {ego_desc}
Third-person view VLM JSON: {exo_desc}
"""

        # Run Real Inference for each candidate model
        for model_key, model_info in llm_candidates.items():
            request = {
                "api_endpoint": model_info["api_endpoint"],
                "query": query_prompt,
                "model_name": model_key,
                "api_name": model_info["model"],
                "service": model_info["service"]
            }
            
            # Call API
            # Some models (e.g. gemma-2-9b-it) do not allow temperature to be 0
            result = call_api(request, max_tokens=128, temperature=0.01)

            raw_response = str(result.get("response", ""))
            pred_ids = extract_ids(raw_response, valid_ids=choices, top_k=top_k)
            final_pred = ",".join(pred_ids) if pred_ids else raw_response.strip()

            # For pass@k behavior we use CEM, for single-label we use EM
            metric = "cem" if top_k > 1 else "em"
            performance = calculate_task_performance(
                prediction=final_pred,
                ground_truth=ground_truth,
                task_name=task_name,
                metric=metric
            ) or 0.0
            
            converted_data.append({
                "task_name": task_name,
                "query": query_prompt,
                "gt": ground_truth,
                "metric": metric,
                "choices": choices,
                "task_id": f"{base_id}|{action_start:.2f}-{action_end:.2f}|{selected_mode}|pass@{top_k}",
                "gt_label": ground_truth_label,
                "model_name": model_key,
                "response": final_pred,
                "performance": performance, 
                "token_num": result["token_num"],
                "input_tokens": result["prompt_tokens"], 
                "output_tokens": result["completion_tokens"],
                "response_time": result["response_time"],
                "api_key_used": "",
                "user_id": None,
                "fig_id": None
            })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(pairs)} samples...")
    
    # Generate embeddings for queries
    rows = []
    total_samples = len(converted_data)
    
    for rid, row in enumerate(converted_data):
        emb = get_longformer_embedding(row['query'])
        row['query_embedding'] = emb.cpu().numpy()
        rows.append(row)
            
        if (rid + 1) % 10 == 0:
            print(f"Embeddings: {rid + 1}/{total_samples}...")
            
    df_all = pd.DataFrame(rows)
    df_train, df_test, embedding_dict = process_final_data(df_all)

    return df_train, df_test

def main():
    parser = argparse.ArgumentParser(description="Convert Charades-Ego to Router JSONL")

    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing CharadesEgo folders")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of video scenes to sample")

    # Task setup
    parser.add_argument(
        "--task_type",
        type=str,
        default="activity",
        choices=["activity", "verb", "object"],
        help="Classification target to build",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Model outputs top-k ids; scored as pass@k when k>1")

    # VLM + frame sampling
    parser.add_argument("--vlm_name", type=str, default="gemma-3-27b-it", help="VLM model ID")
    parser.add_argument("--num_frames", type=int, default=5, help="Frames to sample per view (within the time window)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    train_df, test_df = convert_charades_ego(args.data_root, args.vlm_name, args.sample_size, args.task_type, args.top_k, args.num_frames)
    
    print(f"\nConversion completed in {time.time() - start_time:.1f}s")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    main()
