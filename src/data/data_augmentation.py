







def generate_template_data():  
    """生成模板化的旅游对话数据"""  
    templates = [  
        {  
            "user_template": "我想去{city}旅游，有什么建议吗？",  
            "assistant_template": "对于{city}旅游，我建议：\n1. 必访景点：{attractions}\n2. 建议游玩天数：{days}天\n3. 最佳旅游季节：{season}\n4. 特色美食：{food}"  
        },  
        {  
            "user_template": "{attraction}的门票怎么买？",  
            "assistant_template": "{attraction}的门票信息：\n1. 价格：{price}\n2. 购买渠道：{channels}\n3. 开放时间：{time}\n4. 注意事项：{notes}"  
        }  
    ]  
    
    # 城市数据  
    cities_data = {  
        "北京": {  
            "attractions": "故宫、长城、天坛",  
            "days": "4-5",  
            "season": "春秋两季",  
            "food": "烤鸭、炸酱面"  
        }  
        # ... 更多城市  
    }  
    
    # 生成数据  
    formatted_data = []  
    for template in templates:  
        for city, info in cities_data.items():  
            messages = [  
                {"role": "user", "content": template["user_template"].format(city=city)},  
                {"role": "assistant", "content": template["assistant_template"].format(**info)}  
            ]  
            formatted_data.append({  
                "messages": messages,  
                "id": f"template_{city}_{len(formatted_data)}"  
            })  
    
    return formatted_data