import gradio as gr  
from src.agents.agent import RAG  
from src.agents.tools import ToolDispatcher  
from src.models.model import TravelAgent  
from src.configs.config import RAG_DATA_PATH, SFT_MODEL_PATH  


# 初始化RAG系统  
def initialize_rag():  
    agent = TravelAgent(model_name=SFT_MODEL_PATH)  
    rag = RAG(  
        agent=agent,  
        dataset_name_or_path=RAG_DATA_PATH,  
        use_db=True,  
        use_prompt_template=True  
    )  
    return rag  


# 创建Gradio界面组件  
def create_demo(rag:RAG):  
    with gr.Blocks(title="Travel RAG Assistant", theme=gr.themes.Soft()) as demo:  
        gr.Markdown("# 🌍 智能旅行规划助手") 
        
        
        # 聊天界面  
        with gr.Row():  
            with gr.Column(scale=2):  
                chatbot = gr.Chatbot(height=600, label="对话记录")  
                query_box = gr.Textbox(  
                    placeholder="输入您的旅行问题...",  
                    label="用户输入",  
                    lines=3  
                )   
                
                # 示例问题  
                examples = gr.Examples(  
                    examples=[  
                        ["帮我规划上海3日游，预算5000元"],  
                        ["查找北京到巴黎下周最便宜的机票"],  
                        ["推荐杭州西湖附近的4星级酒店"],  
                        ["查询东京下周的天气预报"]  
                    ],  
                    inputs=query_box,  
                    label="示例问题"  
                )  
                
            # 结果显示区域  
            with gr.Column(scale=1):  
                with gr.Tab("工具调用结果"):  
                    tool_output = gr.JSON(label="工具执行详情")  
                with gr.Tab("数据库匹配结果"):  
                    db_output = gr.DataFrame(  
                        headers=["相关结果"],  
                        datatype=["str"],  
                        # max_rows=5,  
                        # overflow_row_behaviour="show_ends"  
                    )  
                with gr.Tab("原始响应"):  
                    raw_output = gr.Textbox(  
                        lines=8,  
                        max_lines=12,  
                        label="完整响应"  
                    )  
                    
        # 控制按钮  
        with gr.Row():  
            submit_btn = gr.Button("提交", variant="primary") 
            clear_btn = gr.Button("清空对话")
            
            
        # 处理逻辑  
        def respond(query, chat_history):  
            # 生成响应  
            prompt = rag.prompt_template.generate_prompt(  
                query,   
                "\n".join([f"User:{u}\nSystem:{s}" for u, s in chat_history])  
            )  
            
            # 获取模型响应  
            tool_call_str = rag.agent.generate_response(prompt)  
            
            # 执行工具调用  
            tool_result = rag.dispatcher.execute(tool_call_str)  
            
            # 数据库查询  
            db_result = rag.query_db(query)  
            
            # 生成自然语言响应  
            response = f"{tool_call_str}\n\n工具结果：{tool_result}\n数据库匹配：{db_result[:2]}"  
            
            # 获取最终旅行计划
            travel_plan = rag.get_travel_plan(response)
            
            # 更新对话历史  
            chat_history.append((query, travel_plan))  
            
            return {  
                chatbot: gr.update(value=chat_history),  
                tool_output: tool_result,  
                db_output: [[res] for res in db_result],  
                raw_output: response,  
                query_box: ""  
            }  
            
        # 绑定事件  
        submit_btn.click(  
            fn=respond,  
            inputs=[query_box, chatbot],  
            outputs=[chatbot, tool_output, db_output, raw_output, query_box]  
        )  
        
        clear_btn.click(  
            fn=lambda: ([], [], [], ""),  
            outputs=[chatbot, tool_output, db_output, query_box]  
        )  

    return demo 




if __name__ == "__main__":  
    rag_system = initialize_rag()  
    demo = create_demo(rag_system)  
    demo.launch(  
        server_name="0.0.0.0",  
        server_port=7860,  
        share=False,  
        favicon_path="./travel_icon.png"  
    )  