import gradio as gr
import torch
from typing import Dict, Tuple, List, Optional
from ..models.model import TravelAgent
from .mindmap import generate_mindmap
import re
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TravelAgentUI")

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
    
    def generate_mindmap_using_chatbot(self) -> Optional[str]:
        """
        生成思维导图，使用优化后的参数设置
        
        Returns:
            生成的思维导图图片路径，失败时返回None
        """
        try:
            content = self.merge_history_into_mindmap()
            
            if not content.strip():
                logger.warning("聊天历史为空，无法生成思维导图")
                return None
            
            logger.info("开始生成思维导图")
            
            # 使用优化后的参数调用generate_mindmap
            # 限制思维导图层级为3，每层最多5个项目
            img_path = generate_mindmap(
                topic=content,
                level_num=3,
                item_num=5,
                use_cache=True
            )
            
            logger.info(f"思维导图生成完成，路径: {img_path}")
            return img_path
            
        except Exception as e:
            logger.error(f"生成思维导图时出错: {str(e)}")
            return None
        
    def respond(
        self,
        message: str,
        history: List[Dict],
        temperature: float,
        top_p: float
    ) -> List[Dict]:
        """处理用户输入并生成回复"""
        if not message.strip():
            return history
            
        # 构建提示词
        formatted_chat_history = self._format_chat_history()
        prompt = f"{formatted_chat_history}{message}\nAssistant:"
        
        try:
            # 生成回复
            response = self.agent.generate_response(
                prompt=prompt,
                max_length=1024,
                temperature=temperature,
                top_p=top_p
            )
            
            # 更新内部聊天历史
            self.chat_history.append({"role": "user", "content": message})  
            self.chat_history.append({"role": "assistant", "content": response})  
            
            # 返回更新后的聊天历史
            return self.chat_history
        except Exception as e:
            # 错误处理
            error_message = f"抱歉，处理您的请求时发生错误：{str(e)}"
            self.chat_history.append({"role": "user", "content": message})  
            self.chat_history.append({"role": "assistant", "content": error_message})
            return self.chat_history
    
    def create_interface(self):
        """创建Gradio界面，采用现代AI聊天界面设计"""
        # 定义自定义CSS样式
        custom_css = """
        .gradio-container {max-width: 1200px !important;}
        #chatbot {border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);}
        .message.user {background-color: #e3f2fd !important; border-radius: 12px 12px 0 12px;}
        .message.assistant {background-color: #f5f5f5 !important; border-radius: 12px 12px 12px 0;}
        .gr-button-primary {background-color: #2196f3 !important; font-weight: bold;}
        .example-btn {margin: 4px; white-space: normal; word-wrap: break-word;}
        .tab-content {padding: 15px; background-color: #fafafa; border-radius: 8px;}
        """
        
        with gr.Blocks(css=custom_css) as interface:
            # 头部区域
            with gr.Row():
                gr.Markdown("# 🌍 AI Travel Agent", elem_id="header")
                
            # 主体区域使用选项卡布局
            with gr.Tabs():
                # 聊天选项卡
                with gr.Tab("💬 对话"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            # 左侧聊天区域
                            chatbot = gr.Chatbot(
                                [],
                                type="messages",
                                elem_id="chatbot",
                                height=600,
                                avatar_images=(
                                    "https://i.pravatar.cc/150?img=68",  # 用户头像
                                    "https://i.pravatar.cc/150?img=33"   # AI头像
                                )
                            )
                            
                            # 输入区域
                            with gr.Row():
                                message = gr.Textbox(
                                    show_label=False,
                                    placeholder="输入您的旅行相关问题...",
                                    container=True,
                                    lines=3,
                                    scale=10
                                )
                                
                                submit = gr.Button("发送", variant="primary", scale=1)
                            
                            # 快捷操作按钮
                            with gr.Row():
                                mindmap_button = gr.Button("📊 生成思维导图")
                                clear_chat = gr.Button("🗑️ 清空对话")
                        
                        with gr.Column(scale=1):
                            # 右侧面板
                            with gr.Accordion("⚙️ 设置", open=True):
                                temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1,
                                    label="创意度 (Temperature)",
                                    info="较高的值产生更多样化的回答"
                                )
                                top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.1,
                                    label="精确度 (Top P)",
                                    info="控制生成内容的多样性"
                                )
                            
                            # 示例问题
                            with gr.Accordion("💡 示例问题", open=True):
                                example_buttons = []
                                for example in self.example_prompts:
                                    btn = gr.Button(
                                        example, 
                                        size="sm",
                                        elem_classes=["example-btn"]
                                    )
                                    example_buttons.append(btn)
                                    btn.click(
                                        fn=self.set_example_text,
                                        inputs=[btn],
                                        outputs=[message]
                                    )
                
                # 思维导图选项卡
                with gr.Tab("📊 思维导图"):
                    with gr.Row():
                        with gr.Column():
                            mindmap_output = gr.Image(
                                label="旅行计划思维导图",
                                height=600,
                                show_download_button=True,
                                interactive=True
                            )
                        
                        with gr.Column(scale=1):
                            generate_mindmap_btn = gr.Button("🔄 重新生成思维导图", variant="primary")
                            
                            gr.Markdown("""
                            ### 📋 功能说明
                            - 基于对话历史生成旅行计划思维导图
                            - 可视化展示旅行规划的层次结构
                            - 点击重新生成按钮可更新思维导图
                            """)
            
            # 清空对话函数
            def clear_conversation():
                self.chat_history = []
                return []
            
            # 绑定事件
            # 发送消息按钮点击事件
            submit.click(
                self.respond,
                inputs=[message, chatbot, temperature, top_p],
                outputs=[chatbot]
            )
            
            # 回车发送消息
            message.submit(
                self.respond,
                inputs=[message, chatbot, temperature, top_p],
                outputs=[chatbot]
            )
            
            # 清空对话按钮
            clear_chat.click(
                fn=clear_conversation,
                inputs=[],
                outputs=[chatbot]
            )
            
            # 生成思维导图按钮（对话选项卡中）
            def on_generate_mindmap():
                img_path = self.generate_mindmap_using_chatbot()
                if img_path:
                    return img_path
                else:
                    # 如果生成失败，返回一个简单的提示
                    return "没有足够的对话内容来生成思维导图，请先进行一些旅行相关的对话。"
            
            mindmap_button.click(
                fn=on_generate_mindmap,
                inputs=[],
                outputs=[mindmap_output]
            )
            
            # 重新生成思维导图按钮（思维导图选项卡中）
            generate_mindmap_btn.click(
                fn=on_generate_mindmap,
                inputs=[],
                outputs=[mindmap_output]
            )
            
            # 改进的选项卡切换功能 - 添加一个按钮供用户手动触发更新
            with gr.Row(), gr.Column():
                update_on_tab_switch = gr.Button("🔄 同步最新对话到思维导图")
                update_on_tab_switch.click(
                    fn=on_generate_mindmap,
                    inputs=[],
                    outputs=[mindmap_output]
                )
            
            gr.Markdown("💡 **提示**: 切换到思维导图选项卡后，点击上方按钮更新为最新对话内容。")
            
        return interface

# 创建并启动界面
def launch_ui(agent):
    """
    创建并启动Travel Agent UI界面
    
    Args:
        agent: TravelAgent实例
    """
    try:
        logger.info("初始化Travel Agent UI")
        ui = TravelAgentUI(agent)
        interface = ui.create_interface()
        
        logger.info("启动Gradio界面")
        interface.launch(
            share=True,
            debug=False,
            server_name="0.0.0.0",
            server_port=None
        )
        
    except Exception as e:
        logger.error(f"启动UI时出错: {str(e)}")
        raise RuntimeError(f"无法启动Travel Agent UI: {str(e)}") from e