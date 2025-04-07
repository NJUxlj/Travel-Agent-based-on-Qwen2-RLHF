---
language:
- en
license: llama3
configs:
- config_name: default
  data_files:
  - split: train
    path: data.json
---

<div align="center">
  <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63da3d7ae697e5898cb86854/H-vpXOX6KZu01HnV87Jk5.jpeg" width="320" height="320" />
  <h1>Enhancing Human-Like Responses in Large Language Models</h1>
</div>

<p align="center">
  🤗 <a href="https://huggingface.co/collections/HumanLLMs">Models</a> | 📊 <a href="https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset">Dataset</a> | 📄 <a href="https://arxiv.org/abs/2501.05032">Paper</a>
</p>

# **Human-Like-DPO-Dataset**

This dataset was created as part of research aimed at improving conversational fluency and engagement in large language models. It is suitable for formats like **Direct Preference Optimization (DPO)** to guide models toward generating more human-like responses.

The dataset includes **10,884 samples** across **256 topics**, including:

- Technology
- Daily Life
- Science
- History
- Arts

Each sample contains:
- **Conversational Question**: Natural, engaging questions that reflect everyday human dialogue.
- **Human-Like Response**: A natural, conversational answer generated to mimic human interaction.
- **Formal Response**: A structured, professional answer reflecting traditional AI responses.

# **Dataset Usage**
This dataset can be used to fine-tune LLMs to:
- Improve conversational coherence.
- Reduce mechanical or impersonal responses.
- Enhance emotional intelligence in dialogue systems.

More details on dataset creation and usage can be found in the accompanying [research paper](https://arxiv.org/abs/2501.05032).