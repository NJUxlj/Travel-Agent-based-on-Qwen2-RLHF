## 基于Qwen2.5+LoRA微调+RLHF+RAG的旅游路径规划智能体

#### 注：
- 如果各位运行项目出现问题无法解决的话，可以直接给我提issue，我会抽空fix的。
- 最好附上报错截图。



#### 项目的组成部分
- 本项目希望借助轻量级大模型`Qwen2.5-0.5B~3B`帮助用户在本地更好地规划旅行路径，提高旅行体验。本项目是基于Qwen2.5+SFT微调+RLHF+RAG的旅游路径规划智能体。

本项目有以下几个主要部分
- Qwen2模型本体

- SFT + RLHF 系统
- SFTTrainer
- PPOTrainer
- GRPOTrainer


- RAG系统
    辅助工具：
    - 工具调用系统 (Google Search + Weather API + Hotel Booking API + Plane Ticket API + Shortest Path API) 
    - 自定义的Prompt模板, 继承了 query+context+文档库匹配段落+工具列表+工具格式。它可以让大模型返回用户query命中的工具的函数API字符串. 
    - ToolDispatcher: 工具调度器，它可以解析用户query命中的工具的函数API字符串，随后使用ToolExecutor对象调用对应的工具函数。
    - ToolExecutor: 实际调用各种工具API的执行器。
    - ChatPDF

    传统RAG：
    - 基于传统文档匹配+BM25实现的RAG
    - 完全基于Langchain实现的RAG


    MemWalker:

    Self-RAG:

    RAG Dispatcher
    - 可以通过用户传入的参数来选择调用 MemWalker, Self-RAG, 或是传统RAG


- 前端UI
 - Gradio




- 数据预处理系统




## 环境配置
**GPU**: RTX3090 x 2
**Platform**: AutoDL
- NAME="Ubuntu"
- VERSION="20.04.4 LTS (Focal Fossa)"
- CUDA=12.4
- Pytorch=2.5.0

```shell
pip install -r requirements.txt
```


## Download SFT dataset
```bash

cd src/data

huggingface-cli download --resume-download JasleenSingh91/travel-QA --local-dir travel-QA


```

## Download RAG dataset

```bash
cd src/data

huggingface-cli download --resume-download BruceNju/crosswoz-sft --local-dir crosswoz-sft

```

## Data Example Format
- `JasleenSingh91/travel-QA`
```Plain Text

{"question":.......,"response":.....}


```




- `crosswoz`
- crosswoz用在哪里：在RAG系统统中，我们会用query去匹配crosswoz中的history这个字段下的数据，作为对已有的context和tool-use response 的补充。
```Plain Text
{
            "dialog_id": "391",
            "turn_id": 0,
            "role": "usr",
            "content": "你好，麻烦帮我推荐一个门票免费的景点。",
            "dialog_act": "[[\"General\", \"greet\", \"none\", \"none\"], [\"Inform\", \"景点\", \"门票\", \"免费\"], [\"Request\", \"景点\", \"名称\", \"\"]]",
            "history": "[]",
            "user_state": "[[1, \"景点\", \"门票\", \"免费\", true], [1, \"景点\", \"评分\", \"5分\", false], [1, \"景点\", \"地址\", \"\", false], [1, \"景点\", \"游玩时间\", \"\", false], [1, \"景点\", \"名称\", \"\", true], [2, \"餐馆\", \"名称\", \"拿渡麻辣香锅(万达广场店)\", false], [2, \"餐馆\", \"评分\", \"\", false], [2, \"餐馆\", \"营业时间\", \"\", false], [3, \"酒店\", \"价格\", \"400-500元\", false], [3, \"酒店\", \"评分\", \"4.5分以上\", false], [3, \"酒店\", \"周边景点\", [], false], [3, \"酒店\", \"名称\", \"\", false]]",
            "goal": "[[1, \"景点\", \"门票\", \"免费\", false], [1, \"景点\", \"评分\", \"5分\", false], [1, \"景点\", \"地址\", \"\", false], [1, \"景点\", \"游玩时间\", \"\", false], [1, \"景点\", \"名称\", \"\", false], [2, \"餐馆\", \"名称\", \"拿渡麻辣香锅(万达广场店)\", false], [2, \"餐馆\", \"评分\", \"\", false], [2, \"餐馆\", \"营业时间\", \"\", false], [3, \"酒店\", \"价格\", \"400-500元\", false], [3, \"酒店\", \"评分\", \"4.5分以上\", false], [3, \"酒店\", \"周边景点\", [], false], [3, \"酒店\", \"名称\", \"\", false]]",
            "sys_usr": "[43, 18]",
            "sys_state": "",
            "sys_state_init": ""
      }
```

## 如何运行

```shell
# 使用最基础的 RAG
python main.py --function use_rag

# 使用调度器进行RAG类型选择
python main.py --function rag_dispatcher --rag_type self_rag
```




## Experiment Setup
### Model
1. 我们使用了Qwen2.5作为LLM模型
    - 目前仅测试了Qwen2.5的1.5B参数版本

### Project Structure
1. 核心代码都放在 **`src/`** 目录下.
2. **`src/`** 的目录结构：

```yaml
src:
    data:
     - processed_data
     - data_augmentation.py
     - data_preprocessor.py
     - crosswoz-sft:
        - ...
    - travel_qa:
        - ...
     - init.py
    configs:
     - config.py
     - ds_config.py
    agents:
     - travel_knowledge:
        - tour_pages
        - tour_pdfs
     - agent.py
     - bm25.py
     - tools.py # 各种 Executors: google,weather, transportation; 以及 ToolDispatcher
     - zhipuAPI.py
     - prompt_template.py
     - chat_pdf.py 
     - rag.py
     - mem_walker.py
     - self_rag.py
     - corrective_rag.py # 现在还没写好这个，别跑！
     - rag_dispatcher.py
    finetune:
     - dpo_trainer.py
     - sft_trainer.py  # SFTTrainer
     - ppo_trainer.py
     - grpo_trainer.py
     - multi_task_trainer.py
     - init.py
    pretrain:
     - pretrain.py
     - init.py
    models:
     - qwen2
        - modeling_qwen2.py
        - configuration_qwen2.py
        - tokenization_qwen2.py
        - dola_decode.py
     - model.py # TravelAgent类所在的位置
     - init.py
    ui:
     - app.py
     - mindmap.py
     - init.py

data:
     - 各种数据集
utils.py
configs:
     - config.py
     - init.py
```


## python script function explanation
1. **`src/agents/agent.py`**





## SFT Running Snapshot
![SFT](image/SFT_train_snapshot.png)



## 根据 Travel Agent 的路线规划生成思维导图（Mind Map）运行结果
![image](image/mindmap_output.png)





## RAG运行结果
![image](https://github.com/user-attachments/assets/ceec5972-c689-47ba-91d9-9df160e54dd8)
![image](https://github.com/user-attachments/assets/27aea7e5-620b-42dd-a68d-070c5c0be2cb)
![image](https://github.com/user-attachments/assets/577a138a-f3e7-48f0-bd97-e319ae7982c7)


## RAG Web Demo 运行结果
![rag_web_demo](image/rag_web_demo.png)
![rag_web_demo2](image/rag_web_demo2.png)

#### 运行结果解释
我们给RAG的问题包含了：question+context， context是由数据集中前5个与question最接近的样本组成的。





## SFT Evaluation Result
```Plain Text
Epoch  Rouge1   Rouge2  RougeL  BLEU


```


## Citation
- we refer to many other projects when building this project.
- [knowledge-graph-from-GPT](https://github.com/tomhartke/knowledge-graph-from-GPT.git)
- [ai-travel-agent](https://github.com/nirbar1985/ai-travel-agent.git)
- [GPT2](https://github.com/affjljoo3581/GPT2.git)
- [RLHF_instructGPT](https://github.com/LanXiu0523/RLHF_instructGPT.git)

- Dataset Citation:

```bibtext
@inproceedings{zhu2020crosswoz,  
    title={CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset},  
    author={Zhu, Qi and Zhang, Zheng and Fang, Yan and Li, Xiang and Takanobu, Ryuichi and Li, Jinchao and Peng, Baolin and Gao, Jianfeng and Zhu, Xiaoyan and Huang, Minlie},  
    booktitle={Transactions of the Association for Computational Linguistics},  
    year={2020},  
    url={https://arxiv.org/abs/2002.11893}  
}
```