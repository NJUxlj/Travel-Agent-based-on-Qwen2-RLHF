---
{}
---
---  
                    annotations_creators:  
                    - expert-generated  
                    language:  
                    - zh  
                    language_creators:  
                    - expert-generated  
                    license:  
                    - MIT  
                    multilinguality:  
                    - monolingual  
                    pretty_name: crosswoz-sft  
                    size_categories:  
                    - 1K<n<10K  
                    source_datasets:  
                    - original  
                    task_categories:  
                    - conversational  
                    task_ids:  
                    - dialogue-modeling  
                    tags:  
                    - crosswoz  
                    - dialogue  
                    - chinese  

                    dataset_info:  
                    name: crosswoz-sft  
                    description: |  
                          
    这是一个基于CrossWOZ数据集处理的对话数据集，专门用于大模型的监督微调（SFT）任务。  
    数据集包含多轮对话、用户目标、对话状态等信息，适合训练任务型对话系统。  

    原始数据来源于CrossWOZ项目，经过专门的预处理使其更适合现代大模型训练。  
    
    ## 核心特征：
    ---
    这是首个大规模的中文跨域任务型对话数据集
    包含6,012个对话，102,000个话语，覆盖5个领域(酒店、餐厅、景点、地铁和出租车)
    约60%的对话包含跨域用户目标
    主要创新点：

    更具挑战性的域间依赖关系：
    - 一个领域的选择会动态影响其他相关领域的选择
    - 例如用户选择的景点会影响后续酒店的推荐范围(需要在景点附近)
    
    完整的标注：
    - 同时提供用户端和系统端的对话状态标注
    - 包含对话行为(dialogue acts)的标注
    - 用户状态标注有助于追踪对话流程和建模用户行为
    
    高质量的数据收集：
    - 采用同步对话收集方式
    - 两个经过训练的工作人员实时对话
    - 相比MultiWOZ的异步方式，可以确保对话的连贯性
    
    数据规模：
    ---
    - 训练集：5,012个对话
    - 验证集：500个对话
    - 测试集：500个对话
    平均每个对话包含3.24个子目标，16.9个回合
    对话复杂度高于MultiWOZ等现有数据集
    
    这个数据集为研究跨域对话系统提供了新的测试平台，可用于对话状态追踪、策略学习等多个任务的研究。
      
  
                        statistics:  
                            splits:  
                        
  train: 84692,
  validation: 8458,
  test: 8476
  
                            features:  
                            - name: dialog_id  
                                dtype: string  
                            - name: turn_id  
                                dtype: int64  
                            - name: role  
                                dtype: string  
                            - name: content  
                                dtype: string  
                            - name: dialog_act  
                                dtype: string  
                            - name: history  
                                dtype: string  
                            - name: user_state  
                                dtype: string  
                            - name: goal  
                                dtype: string  
                            - name: sys_usr  
                                dtype: string  
                            - name: sys_state  
                                dtype: string  
                            - name: sys_state_init  
                                dtype: string  


                example:  
                    train:
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

                preprocessing:  
                    steps:  
                    - 对话历史的展平和格式化  
                    - 状态信息的JSON序列化  
                    - 特殊字段的标准化处理  
                    - 数据集的划分（训练集/验证集/测试集）  

                usage:  
                loading_code: |  
                    ```python  
                    from datasets import load_dataset  

                    # 加载完整数据集  
                    dataset = load_dataset("BruceNju/crosswoz-sft")  

                    # 加载特定分割  
                    train_dataset = load_dataset("BruceNju/crosswoz-sft", split="train")  
                    validation_dataset = load_dataset("BruceNju/crosswoz-sft", split="validation")  
                    test_dataset = load_dataset("BruceNju/crosswoz-sft", split="test")  
                    ```  

        splits:  
            train: 84692 samples  
            validation: 8458 samples  
            test: 8476 samples  

        citation: |  
        @inproceedings{zhu2020crosswoz,  
            title={CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset},  
            author={Zhu, Qi and Zhang, Zheng and Fang, Yan and Li, Xiang and Takanobu, Ryuichi and Li, Jinchao and Peng, Baolin and Gao, Jianfeng and Zhu, Xiaoyan and Huang, Minlie},  
            booktitle={Transactions of the Association for Computational Linguistics},  
            year={2020},  
            url={https://arxiv.org/abs/2002.11893}  
        }  

        version:  
        version_name: "1.0.0"  
        changes:  
            - 2025-01-03: 首次发布  
        
        
        