import gradio as gr
import torch
from typing import Dict, Tuple, List
from ..models.model import TravelAgent
from .mindmap import generate_mindmap
import re

'''

案例：How do I travel from Shanghai to Paris?
'''

class TravelAgentUI:
    def __init__(self, agent:TravelAgent):
        self.agent = agent
        self.chat_history = []
        
        # 预设的示例问题  
        self.example_prompts = [  
            "推荐三个适合12月份旅游的城市",  
            "帮我规划一个为期3天的北京旅游行程",  
            "我想去海边度假，预算8000元，有什么建议？",  
            "推荐几个适合带父母旅游的目的地",  
            "帮我列出去日本旅游需要准备的物品清单"  
        ]  
    def set_example_text(self, example: str) -> str:  
        """设置示例文本到输入框"""  
        return example  
        
    def _format_chat_history(self) -> str:
        """格式化聊天历史"""
        formatted = ""  
        for msg in self.chat_history:  
            if msg["role"] == "user":  
                formatted += f"User: {msg['content']}\n"  
            elif msg["role"] == "assistant":  
                formatted += f"Assistant: {msg['content']}\n\n"  
        
        if formatted == "":  
            formatted = "System: You are a Travel Agent that can help user plan a route from one start location to a end location. This plan you give should be in detail.\n\n"  
        
        return formatted + "User: "  
    
    def merge_history_into_mindmap(self) -> str:
        """将聊天历史合并为思维导图"""
        content = self._format_chat_history()
        return re.sub(r"User:\s*$", "", content)
    
    def generate_mindmap_using_chatbot(self) -> str:
        """生成思维导图"""
        content = self.merge_history_into_mindmap()
        return generate_mindmap(content)
        
    def respond(
        self,
        message: str,
        history: List[Tuple[str, str]],
        temperature: float,
        top_p: float
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """处理用户输入并生成回复"""
        # 构建提示词
        formatted_chat_history = self._format_chat_history()
        prompt = f"{formatted_chat_history}{message}\nAssistant:"
        
        # 生成回复
        response = self.agent.generate_response(
            prompt=prompt,
            max_length=1024,
            temperature=temperature,
            top_p=top_p
        )
        
        self.chat_history.append({"role": "user", "content": message})  
        self.chat_history.append({"role": "assistant", "content": response})  
        
        # return response, self.chat_history
        return self.chat_history
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(css="footer {display: none !important}") as interface:
            gr.Markdown("# 🌍 AI Travel Agent")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        [],
                        type="messages",
                        elem_id="chatbot",
                        height=500
                    )
                    
                with gr.Column(scale=1):
                    with gr.Accordion("Settings", open=False):
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.1,
                            label="Top P"
                        )
            
            with gr.Row():
                message = gr.Textbox(
                    show_label=False,
                    placeholder="输入您的旅行相关问题...",
                    container=True,
                    min_width=300,  # 固定宽度
                    lines=5,  # 设置行数
                )
                
                submit = gr.Button("发送")
                
                mindmap_button = gr.Button("生成思维导图")
                
                
            # 添加示例提示按钮  
            with gr.Row():  
                example_buttons = []  
                for example in self.example_prompts:  
                    btn = gr.Button(example, size="sm")  
                    example_buttons.append(btn)  
                    # 绑定点击事件到输入框  
                    btn.click(  
                        fn=self.set_example_text,  
                        inputs=[btn],  
                        outputs=[message]  
                    )  
            
            # 添加说明文本  
            gr.Markdown("""  
            ### 💡 使用提示：  
            - 点击上方按钮可快速选择常见问题  
            - 您也可以直接输入自定义问题  
            - 可以在设置中调整回复的多样性（Temperature）和质量（Top P）  
            """)  
            
            with gr.Row():
                mindmap_output = gr.Image(
                    label="Generated Mind Map",
                    show_label=False,
                    height=500,
                )
            
            # 绑定事件
            tmp=None
            
            submit_click = submit.click(
                self.respond,
                inputs=[message, chatbot, temperature, top_p],
                outputs=[chatbot]
            )
            
            # message.submit(
            #     self.respond,
            #     inputs=[message, chatbot, temperature, top_p],
            #     outputs=[message, chatbot]
            # )
            
            mindmap_button.click(
                self.generate_mindmap_using_chatbot,
                inputs=[],
                outputs=[mindmap_output]
            )
            
        return interface

# 创建并启动界面
def launch_ui(agent):
    ui = TravelAgentUI(agent)
    interface = ui.create_interface()
    interface.launch(share=True)