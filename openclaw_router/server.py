"""
OpenClaw Router Server
======================
OpenAI-compatible API server with intelligent LLM routing.

Usage:
    llmrouter serve --config configs/openclaw_example.yaml

Or directly:
    python server.py --config config.yaml
"""

import json
import os
import re
import sys
from typing import AsyncGenerator, Optional, Dict, Any, List

# Check dependencies
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import httpx
    import uvicorn
except ImportError:
    print("Please install: pip install fastapi uvicorn httpx pydantic")
    sys.exit(1)

# Handle both relative and direct imports
try:
    from .config import OpenClawConfig, LLMConfig, MODELS_WITHOUT_SYSTEM_ROLE, MODEL_CONTEXT_LIMITS
    from .routers import OpenClawRouter, _safe_log
    from .media import process_multimodal_content, MediaConfig
except ImportError:
    from config import OpenClawConfig, LLMConfig, MODELS_WITHOUT_SYSTEM_ROLE, MODEL_CONTEXT_LIMITS
    from routers import OpenClawRouter, _safe_log
    from media import process_multimodal_content, MediaConfig


# ============================================================
# Request/Response Models
# ============================================================

class Message(BaseModel):
    role: str
    content: Any  # Can be string or list (multimodal)


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    user: Optional[str] = None  # Optional user id (used for memory scoping if enabled)


# ============================================================
# Message Processing
# ============================================================

def normalize_content(content: Any) -> str:
    """Convert multimodal content to plain string"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif "text" in part:
                    text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content) if content else ""


def normalize_messages(messages: List[Dict], model_id: str = "") -> List[Dict]:
    """Normalize message format for compatibility"""
    normalized = []
    system_content = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = normalize_content(msg.get("content", ""))

        if role == "system":
            system_content = content
        else:
            normalized.append({"role": role, "content": content})

    # Handle models without system role support
    if system_content and model_id in MODELS_WITHOUT_SYSTEM_ROLE:
        if normalized and normalized[0]["role"] == "user":
            normalized[0]["content"] = f"[System Instructions]\n{system_content}\n\n[User Message]\n{normalized[0]['content']}"
        else:
            normalized.insert(0, {"role": "user", "content": f"[System Instructions]\n{system_content}"})
    elif system_content:
        normalized.insert(0, {"role": "system", "content": system_content})

    return normalized


def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 4 chars = 1 token)"""
    return len(text) // 4


def adjust_max_tokens(messages: List[Dict], model_id: str, requested_max: int) -> int:
    """Adjust max_tokens based on context limit"""
    context_limit = MODEL_CONTEXT_LIMITS.get(model_id, 32768)

    input_text = " ".join(m.get("content", "") for m in messages)
    input_tokens = estimate_tokens(input_text)

    available = context_limit - input_tokens - 100
    if available < 100:
        available = 100

    result = min(requested_max, available)

    # NVIDIA API limits max_tokens to 1024
    if model_id in MODELS_WITHOUT_SYSTEM_ROLE:
        result = min(result, 1024)

    return result


def clean_response(result: Dict) -> Dict:
    """Clean response for OpenAI compatibility"""
    usage_raw = result.get("usage", {})
    usage = {}
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        if key in usage_raw and usage_raw[key] is not None:
            usage[key] = usage_raw[key]

    cleaned = {
        "id": result.get("id", ""),
        "object": result.get("object", "chat.completion"),
        "model": result.get("model", ""),
        "choices": [],
        "usage": usage
    }

    for choice in result.get("choices", []):
        cleaned_choice = {
            "index": choice.get("index", 0),
            "finish_reason": choice.get("finish_reason", "stop")
        }
        if "message" in choice:
            msg = choice["message"]
            cleaned_choice["message"] = {
                "role": msg.get("role", "assistant"),
                "content": msg.get("content", "")
            }
        cleaned["choices"].append(cleaned_choice)

    return cleaned


def clean_streaming_chunk(chunk: Dict) -> Optional[Dict]:
    """Clean streaming chunk for OpenAI compatibility"""
    choices = chunk.get("choices", [])
    if not choices:
        return None

    cleaned = {
        "id": chunk.get("id", ""),
        "object": chunk.get("object", "chat.completion.chunk"),
        "choices": []
    }

    for choice in choices:
        finish_reason = choice.get("finish_reason")
        cleaned_choice = {
            "index": choice.get("index", 0),
            "finish_reason": finish_reason
        }

        if "delta" in choice:
            delta = choice["delta"]
            if finish_reason == "stop":
                cleaned_choice["delta"] = {}
            else:
                cleaned_delta = {}
                if "role" in delta:
                    cleaned_delta["role"] = delta["role"]
                if "content" in delta:
                    cleaned_delta["content"] = delta["content"]
                cleaned_choice["delta"] = cleaned_delta
        else:
            cleaned_choice["delta"] = {}

        cleaned["choices"].append(cleaned_choice)

    return cleaned


# ============================================================
# LLM Backend
# ============================================================

class LLMBackend:
    """LLM API caller"""

    def __init__(self, config: OpenClawConfig):
        self.config = config

    async def call(self, llm_name: str, messages: List[Dict], max_tokens: int = 4096,
                   temperature: Optional[float] = None, stream: bool = False):
        """Call LLM API"""
        if llm_name not in self.config.llms:
            raise HTTPException(status_code=404, detail=f"LLM '{llm_name}' not found")

        llm_config = self.config.llms[llm_name]
        api_key = self.config.get_api_key(llm_config.provider, llm_config)

        if stream:
            return self._call_streaming(llm_config, messages, max_tokens, temperature, api_key)
        else:
            return await self._call_sync(llm_config, messages, max_tokens, temperature, api_key)

    async def _call_sync(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                         temperature: Optional[float], api_key: str) -> Dict:
        """Synchronous API call"""
        normalized = normalize_messages(messages, llm.model_id)
        adjusted_max = adjust_max_tokens(normalized, llm.model_id, max_tokens)


        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": normalized,
                "max_tokens": adjusted_max,
            }
            if temperature is not None:
                body["temperature"] = temperature

            resp = await client.post(
                f"{llm.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0
            )

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

            result = resp.json()
            return clean_response(result)

    async def _call_streaming(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                              temperature: Optional[float], api_key: str) -> AsyncGenerator:
        """Streaming API call"""
        normalized = normalize_messages(messages, llm.model_id)
        adjusted_max = adjust_max_tokens(normalized, llm.model_id, max_tokens)

        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": normalized,
                "max_tokens": adjusted_max,
                "stream": True
            }
            if temperature is not None:
                body["temperature"] = temperature

            async with client.stream(
                "POST",
                f"{llm.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0
            ) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    print(f"[Backend Streaming] Error {resp.status_code}: {error.decode()[:200]}")
                    yield f'data: {json.dumps({"error": error.decode()[:200]})}\n\n'
                    return

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield line + "\n\n"


# ============================================================
# FastAPI App Factory
# ============================================================

def create_app(config: OpenClawConfig = None, config_path: str = None) -> FastAPI:
    """Create FastAPI application"""
    if config is None and config_path:
        config = OpenClawConfig.from_yaml(config_path)
    elif config is None:
        config = OpenClawConfig()

    app = FastAPI(
        title="OpenClaw Router",
        description="OpenAI-compatible API with intelligent LLM routing",
        version="1.0.0"
    )

    # Initialize components
    router = OpenClawRouter(config)
    backend = LLMBackend(config)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "strategy": config.router.strategy,
            "llms": list(config.llms.keys())
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "description": llm.description}
                for name, llm in config.llms.items()
            ] + [{"id": "auto", "object": "model", "description": "Auto router"}]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Extract user query for routing (with optional media understanding)
        user_query = ""
        media_description = None

        # Find and process the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            raw_content = messages[last_user_idx]["content"]

            # Process multimodal content if media is enabled
            # Supports both OpenAI format (list) and OpenClaw format (string with [media attached:...])
            if config.media.enabled:
                # Use together API key as fallback
                together_key = config.api_keys.get("together")
                processed_text, media_desc = await process_multimodal_content(
                    raw_content, config.media, fallback_key=together_key
                )
                user_query = processed_text[:500]
                media_description = media_desc
                if media_desc:
                    print(f"[Media] Processed: {media_desc[:80]}...")
                    # IMPORTANT: Replace the message content with processed text
                    # so LLM sees the image description instead of [media attached: ...]
                    messages[last_user_idx]["content"] = processed_text
            else:
                user_query = normalize_content(raw_content)[:500]

        if not user_query:
            user_query = "general query"

        # Select model
        available_models = list(config.llms.keys())
        if request.model == "auto" or request.model not in available_models:
            selected_model = await router.select_model(user_query, user=request.user)
            # ASCII-only log to avoid Windows GBK UnicodeEncodeError.
            print(f"[Router] Query: '{user_query[:50]}...' -> {selected_model}")
        else:
            selected_model = request.model

        # Handle streaming
        if request.stream:
            async def generate():
                prefix_sent = False
                content_buffer = ""
                buffered_chunks = []

                try:
                    stream_gen = await backend.call(
                        selected_model, messages, request.max_tokens,
                        request.temperature, stream=True
                    )
                    async for chunk in stream_gen:
                        if not config.show_model_prefix:
                            yield chunk
                            continue

                        # Add model prefix to first content chunk
                        if "[DONE]" in chunk:
                            # Flush buffer before DONE
                            if buffered_chunks and not prefix_sent:
                                content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                first = buffered_chunks[0]
                                try:
                                    data = json.loads(first[6:]) if first.startswith("data: ") else {}
                                    if data.get("choices") and data["choices"][0].get("delta"):
                                        data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                        yield f"data: {json.dumps(data)}\n\n"
                                except:
                                    pass
                            yield chunk
                        else:
                            try:
                                json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                                data = json.loads(json_str.strip())
                                cleaned = clean_streaming_chunk(data)

                                if cleaned:
                                    choices = cleaned.get("choices", [])
                                    if choices and "delta" in choices[0]:
                                        content = choices[0]["delta"].get("content", "")

                                        if not prefix_sent:
                                            content_buffer += content
                                            buffered_chunks.append(chunk)

                                            if len(content_buffer) > 30 or (content_buffer and not content_buffer.startswith("[")):
                                                content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                                first = buffered_chunks[0]
                                                first_data = json.loads(first[6:] if first.startswith("data: ") else first)
                                                if first_data.get("choices") and first_data["choices"][0].get("delta"):
                                                    first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                                    yield f"data: {json.dumps(first_data)}\n\n"
                                                    prefix_sent = True
                                                    buffered_chunks = []
                                        else:
                                            yield f"data: {json.dumps(cleaned)}\n\n"
                                    else:
                                        if prefix_sent:
                                            yield f"data: {json.dumps(cleaned)}\n\n"
                            except:
                                yield chunk
                except Exception as e:
                    print(f"[Stream Error] {type(e).__name__}: {e}")
                    yield f'data: {json.dumps({"error": str(e)})}\n\n'

            return StreamingResponse(generate(), media_type="text/event-stream")

        else:
            result = await backend.call(
                selected_model, messages, request.max_tokens,
                request.temperature, stream=False
            )

            # Add model prefix
            if config.show_model_prefix and result.get("choices"):
                content = result["choices"][0].get("message", {}).get("content", "")
                if content:
                    # Remove any existing prefix
                    content = re.sub(r'^\[[\w\-\.]+\]\s*', '', content)
                    result["choices"][0]["message"]["content"] = f"[{selected_model}] {content}"

            result["model"] = selected_model
            return result

    @app.get("/")
    async def root():
        return {
            "name": "OpenClaw Router",
            "version": "1.0.0",
            "strategy": config.router.strategy,
            "llms": list(config.llms.keys()),
            "endpoints": {
                "chat": "POST /v1/chat/completions",
                "models": "GET /v1/models",
                "health": "GET /health"
            }
        }

    @app.get("/routers")
    async def list_routers():
        """List available routing strategies"""
        return {
            "available_routers": router.get_available_routers(),
            "current": config.router.strategy
        }

    @app.websocket("/v1/chat/ws")
    async def chat_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time streaming"""
        await websocket.accept()
        try:
            # Receive request
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            messages = [{"role": m.role, "content": m.content} for m in request.messages]

            # Extract user query for routing
            user_query = ""
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                raw_content = messages[last_user_idx]["content"]
                if config.media.enabled:
                    together_key = config.api_keys.get("together")
                    processed_text, _ = await process_multimodal_content(
                        raw_content, config.media, fallback_key=together_key
                    )
                    user_query = processed_text[:500]
                    messages[last_user_idx]["content"] = processed_text
                else:
                    user_query = normalize_content(raw_content)[:500]

            if not user_query:
                user_query = "general query"

            # Select model
            available_models = list(config.llms.keys())
            if request.model == "auto" or request.model not in available_models:
                selected_model = await router.select_model(user_query, user=request.user)
                _safe_log(f"[WS Router] Query: '{user_query[:50]}...' -> {selected_model}")
            else:
                selected_model = request.model

            # Call LLM backend in streaming mode
            prefix_sent = False
            content_buffer = ""
            buffered_chunks = []

            stream_gen = await backend.call(
                selected_model, messages, request.max_tokens,
                request.temperature, stream=True
            )

            async for chunk in stream_gen:
                if not config.show_model_prefix:
                    await websocket.send_text(chunk)
                    continue

                if "[DONE]" in chunk:
                    if buffered_chunks and not prefix_sent:
                        content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                        first = buffered_chunks[0]
                        try:
                            data_chunk = json.loads(first[6:]) if first.startswith("data: ") else {}
                            if data_chunk.get("choices") and data_chunk["choices"][0].get("delta"):
                                data_chunk["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                await websocket.send_text(f"data: {json.dumps(data_chunk)}\n\n")
                        except:
                            pass
                    await websocket.send_text(chunk)
                else:
                    try:
                        json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                        data_chunk = json.loads(json_str.strip())
                        cleaned = clean_streaming_chunk(data_chunk)

                        if cleaned:
                            choices = cleaned.get("choices", [])
                            if choices and "delta" in choices[0]:
                                content = choices[0]["delta"].get("content", "")

                                if not prefix_sent:
                                    content_buffer += content
                                    buffered_chunks.append(chunk)

                                    if len(content_buffer) > 30 or (content_buffer and not content_buffer.startswith("[")):
                                        content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                        first = buffered_chunks[0]
                                        first_data = json.loads(first[6:] if first.startswith("data: ") else first)
                                        if first_data.get("choices") and first_data["choices"][0].get("delta"):
                                            first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                            await websocket.send_text(f"data: {json.dumps(first_data)}\n\n")
                                            prefix_sent = True
                                            buffered_chunks = []
                                else:
                                    await websocket.send_json(cleaned)
                            else:
                                if prefix_sent:
                                    await websocket.send_json(cleaned)
                    except:
                        await websocket.send_text(chunk)

        except WebSocketDisconnect:
            _safe_log("[WS] Client disconnected")
        except Exception as e:
            _safe_log(f"[WS Error] {type(e).__name__}: {e}")
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
        finally:
            try:
                await websocket.close()
            except:
                pass

    return app


def run_server(app: FastAPI = None, config_path: str = None, host: str = "0.0.0.0", port: int = 8000):
    """Run the server"""
    if app is None:
        app = create_app(config_path=config_path)

    print(f"""
============================================================
  OpenClaw Router
============================================================
  Server: http://{host}:{port}
  API:    http://{host}:{port}/v1/chat/completions
  Health: http://{host}:{port}/health
============================================================
""")

    uvicorn.run(app, host=host, port=port)


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenClaw Router Server")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    run_server(config_path=args.config, host=args.host, port=args.port)
