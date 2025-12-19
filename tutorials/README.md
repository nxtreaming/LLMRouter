# LLMRouter Tutorials - Structure Plan

This directory contains a comprehensive set of Google Colab notebooks for learning LLMRouter.

## 📚 Tutorial Structure

### Beginner Level

1. **00_Quick_Start.ipynb** (15 min)
   - Install LLMRouter
   - Run your first inference
   - Try built-in routers
   - Basic concepts

2. **01_Installation_and_Setup.ipynb** (20 min)
   - Detailed installation
   - Environment configuration
   - API key setup
   - Verify installation

3. **02_Data_Preparation.ipynb** (30 min)
   - Understanding data formats
   - LLM candidates JSON
   - Query data (JSONL)
   - Routing data format
   - Data preprocessing

### Intermediate Level

4. **03_Training_Single_Round_Routers.ipynb** (45 min)
   - KNN Router training
   - SVM Router training
   - MLP Router training
   - Matrix Factorization Router
   - Comparing results

5. **04_Training_Advanced_Routers.ipynb** (45 min)
   - Graph Router
   - Dual Contrastive Router
   - Causal LM Router
   - Hybrid LLM Router

6. **05_Inference_and_Evaluation.ipynb** (30 min)
   - Single query inference
   - Batch inference
   - Route-only mode
   - Performance metrics
   - Cost analysis

7. **06_Interactive_Chat_Interface.ipynb** (20 min)
   - Launch Gradio interface
   - Query modes (current_only, full_context, retrieval)
   - Customizing chat interface

### Advanced Level

8. **07_Creating_Custom_Routers.ipynb** (60 min)
   - Understanding MetaRouter interface
   - Building a simple custom router
   - Adding training support
   - Testing custom router
   - Plugin system

9. **08_Multi_Round_Routers.ipynb** (45 min)
   - Understanding multi-round routing
   - KNN Multi-Round Router
   - LLM Multi-Round Router
   - Complex query decomposition

10. **09_Adding_New_LLM_Models.ipynb** (30 min)
    - LLM candidates format
    - Adding model configurations
    - Model embeddings
    - API integration
    - Testing new models

11. **10_Creating_Custom_Datasets.ipynb** (45 min)
    - Dataset requirements
    - Converting existing datasets
    - ChatBot Arena format
    - MT-Bench format
    - Custom domain data
    - Data validation

12. **11_Advanced_Customization.ipynb** (45 min)
    - Custom embedding models
    - Custom evaluation metrics
    - Cost-aware routing
    - Latency optimization
    - A/B testing routers

### Expert Level

13. **12_Production_Deployment.ipynb** (30 min)
    - API deployment
    - Scaling considerations
    - Monitoring and logging
    - Error handling
    - Best practices

14. **13_Research_and_Experimentation.ipynb** (45 min)
    - Benchmarking routers
    - Ablation studies
    - Novel routing strategies
    - Publishing results

## 🎯 Tutorial Features

Each notebook includes:
- ✅ Clear learning objectives
- ✅ Step-by-step instructions
- ✅ Runnable code cells
- ✅ Expected outputs
- ✅ Common errors and solutions
- ✅ Exercises (optional)
- ✅ Next steps

## 📝 Notebook Template

```markdown
# Tutorial Title

**Estimated Time:** XX minutes
**Level:** Beginner/Intermediate/Advanced
**Prerequisites:** List of required notebooks

## Learning Objectives

By the end of this tutorial, you will:
- [ ] Objective 1
- [ ] Objective 2
- [ ] Objective 3

## Setup

[Installation and imports]

## Section 1: Topic Name

[Content]

## Exercises (Optional)

[Practice exercises]

## Summary

[Key takeaways]

## Next Steps

[Link to next tutorial]
```

## 🚀 Quick Navigation

**For Complete Beginners:** Start with 00 → 01 → 02 → 03 → 05

**For ML Practitioners:** Start with 01 → 03 → 04 → 07

**For Researchers:** 02 → 10 → 11 → 13

**For Production Use:** 01 → 05 → 09 → 12

## 📦 Downloading Data

All tutorials use example data from the repository:
- `data/example_data/` - Small datasets for quick testing
- `data/chatbot_arena/` - Real-world data (optional)

## 💡 Tips for Using These Tutorials

1. **Run in Order**: Follow the recommended sequence for best learning experience
2. **Experiment**: Modify code cells to understand behavior
3. **Save Your Work**: Make a copy before making changes
4. **Use GPU**: Enable GPU runtime for training tutorials
5. **Ask Questions**: Use GitHub issues for questions

## 🔗 Additional Resources

- Main Documentation: [README.md](../README.md)
- Custom Routers Guide: [docs/CUSTOM_ROUTERS.md](../docs/CUSTOM_ROUTERS.md)
- API Reference: [Generated from docstrings]
- Paper: [Link to paper]

## 🤝 Contributing

Found an error or want to add a tutorial? Please submit a PR!

---

**Note:** These notebooks are designed to run on Google Colab with free GPU access.
