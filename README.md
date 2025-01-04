## 基于Qwen2.5+LoRA微调+RLHF+RAG的旅游路径规划智能体

Man, what can I say !
## 实验报告
有 实验报告.pdf 可供了解。
请注意：报告中仅描述了该项目的早期版本，很多重要模块都没加

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
