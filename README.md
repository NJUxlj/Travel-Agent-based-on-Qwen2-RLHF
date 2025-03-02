## 基于Qwen2.5+LoRA微调+RLHF+RAG的旅游路径规划智能体





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


## Data Example Format
- `JasleenSingh91/travel-QA`
```Plain Text

{"question":.......,"response":.....}


```




- `crosswoz`
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
python main.py
```

```shell
python rag_naive.py
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
     - init.py
    training:
     - dpo_trainer.py
     - sft_trainer.py
     - multi_task_trainer.py
     - init.py
    models:
     - model.py
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

### Dataset
1. 我们使用了一个旅游对话数据集：CrossWOZ


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


## Travel Agent运行结果
![image](image/Snipaste_2024-12-30_20-16-28.png)





## RAG运行结果
![image](https://github.com/user-attachments/assets/ceec5972-c689-47ba-91d9-9df160e54dd8)
![image](https://github.com/user-attachments/assets/27aea7e5-620b-42dd-a68d-070c5c0be2cb)
![image](https://github.com/user-attachments/assets/577a138a-f3e7-48f0-bd97-e319ae7982c7)


#### 运行结果解释
我们给RAG的问题包含了：question+context， context是由数据集中前5个与question最接近的样本组成的。






## Citation
- we refer to many other projects when building this project.
- [knowledge-graph-from-GPT](https://github.com/tomhartke/knowledge-graph-from-GPT.git)
- [ai-travel-agent](https://github.com/nirbar1985/ai-travel-agent.git)
- [GPT2](https://github.com/affjljoo3581/GPT2.git)
- [RLHF_instructGPT](https://github.com/LanXiu0523/RLHF_instructGPT.git)

