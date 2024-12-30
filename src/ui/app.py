import gradio as gr
import torch
from typing import Dict, Tuple, List
from ..models.model import TravelAgent
from .mindmap import generate_mindmap

class TravelAgentUI:
    def __init__(self, agent:TravelAgent):
        self.agent = agent
        self.chat_history = []
        
    def _format_chat_history(self) -> str:
        """格式化聊天历史"""
        formatted = ""
        for user_msg, bot_msg in self.chat_history:
            formatted += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
        return formatted
        
    def respond(
        self,
        message: str,
        history: List[Tuple[str, str]],
        temperature: float,
        top_p: float
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """处理用户输入并生成回复"""
        # 构建提示词
        chat_history = self._format_chat_history()
        prompt = f"{chat_history}User: {message}\nAssistant:"
        
        # 生成回复
        response = self.agent.generate_response(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p
        )
        
        # 更新历史
        history.append((message, response))
        self.chat_history = history
        
        return response, history
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(css="footer {display: none !important}") as interface:
            gr.Markdown("# 🌍 AI Travel Agent")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        height=300
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
                    container=False
                )
                
                submit = gr.Button("发送")
                
                mindmap_button = gr.Button("生成思维导图")
            
            with gr.Row():
                mindmap_output = gr.Image(
                    label="Generated Mind Map",
                    show_label=False,
                    height=500,
                )
            
            # 绑定事件
            submit_click = submit.click(
                self.respond,
                inputs=[message, chatbot, temperature, top_p],
                outputs=[message, chatbot]
            )
            
            # message.submit(
            #     self.respond,
            #     inputs=[message, chatbot, temperature, top_p],
            #     outputs=[message, chatbot]
            # )
            
            mindmap_button.click(
                generate_mindmap,
                inputs=[message],
                outputs=[mindmap_output]
            )
            
        return interface

# 创建并启动界面
def launch_ui(agent):
    ui = TravelAgentUI(agent)
    interface = ui.create_interface()
    interface.launch(share=True)