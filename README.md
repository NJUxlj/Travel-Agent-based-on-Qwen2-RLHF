## 基于huggingface大模型微调+RLHF+RAG的旅游路径规划智能体

## 实验报告
有 实验报告.pdf 可供了解。
请注意：报告中仅描述了该项目的早期版本，很多重要模块都没加

## 环境配置




## 如何运行

```shell
python main.py
```

```shell
python rag_naive.py
```

## Travel Agent运行结果
![image](/root/autodl-tmp/Travel-Agent-based-on-LLM-and-SFT/image/Snipaste_2024-12-30_20-16-28.png)





## RAG运行结果
![image](https://github.com/user-attachments/assets/ceec5972-c689-47ba-91d9-9df160e54dd8)
![image](https://github.com/user-attachments/assets/27aea7e5-620b-42dd-a68d-070c5c0be2cb)
![image](https://github.com/user-attachments/assets/577a138a-f3e7-48f0-bd97-e319ae7982c7)


#### 运行结果解释
我们给RAG的问题包含了：question+context， context是由数据集中前5个与question最接近的样本组成的。
