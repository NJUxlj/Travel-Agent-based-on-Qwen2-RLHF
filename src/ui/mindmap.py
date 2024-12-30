import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List
import re
import graphviz
from pathlib import Path
import tempfile
import os
import uuid

import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
from configs.config import MODEL_CONFIG

model_path = MODEL_CONFIG['model']['name']  


import subprocess
def check_graphviz_installed():  
    """检查系统是否安装了graphviz"""  
    try:  
        # 运行命令 dot -V，其中 dot 是 Graphviz 的命令行工具，-V 用于打印版本信息。
        # capture_output=True 表示捕获命令的输出，check=True 表示如果命令返回非零退出状态，则抛出异常。
        subprocess.run(['dot', '-V'], capture_output=True, check=True)  
        return True  
    except (subprocess.SubprocessError, FileNotFoundError):  
        return False



def clean_text(text: str) -> str:  
    """  
    清理文本，移除或替换可能导致graphviz语法错误的字符  
    """  
    # 移除或替换特殊字符  
    text = re.sub(r'[^\w\s-]', '_', text)  
    # 确保文本不为空  
    text = text.strip() or "node"  
    # 限制文本长度  
    return text[:50]  

class MindMapGenerator:
    def __init__(self, model_name: str = model_path):
        """
        初始化思维导图生成器
        Args:
            model_name: Hugging Face模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate_mindmap_content(self, topic: str) -> str:
        """
        使用大模型生成思维导图内容
        Args:
            topic: 用户输入的主题
        Returns:
            生成的思维导图内容（层级列表格式）
        """
        prompt = f"""Please create a detailed mind map about "{topic}". 
        The output should be in a hierarchical format with main topics and subtopics.
        Format the output as a list with proper indentation using - for each level.
        Keep it concise but informative. Generate no more than 3 levels and 12 total items.
        
        Example format:
        - Main Topic
          - Subtopic 1
            - Detail 1
            - Detail 2
          - Subtopic 2
            - Detail 3
            - Detail 4
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取生成的列表部分
        content = response.split("\n\n")[-1]
        return content

    def parse_hierarchy(self, content: str) -> List[tuple]:
        """
        解析层级列表内容为节点关系
        Args:
            content: 生成的层级列表内容
        Returns:
            节点关系列表 [(parent, child, level)]
        """
        lines = content.strip().split('\n')
        nodes = []
        previous_nodes = [''] * 10  # 存储每个层级的前一个节点
        
        for line in lines:
            # 计算缩进级别
            indent_level = len(re.match(r'^\s*', line).group()) // 2
            # 提取文本内容
            text = line.strip().strip('- ')
            
            if indent_level == 0:
                nodes.append(('ROOT', text, indent_level))
            else:
                parent = previous_nodes[indent_level - 1]
                nodes.append((parent, text, indent_level))
            
            previous_nodes[indent_level] = text
            
        return nodes

    def create_mindmap(self, topic: str, nodes: List[tuple]) -> str:
        """
        使用graphviz创建思维导图
        Args:
            topic: 主题
            nodes: 节点关系列表
        Returns:
            生成的图片路径
        """
        if not check_graphviz_installed():  
            raise RuntimeError(  
                "Graphviz not found. Please install it first:\n"  
                "Ubuntu/Debian: sudo apt-get install graphviz\n"  
                "CentOS: sudo yum install graphviz\n"  
                "MacOS: brew install graphviz\n\n"
                "pip install graphviz"  
            )  
            
        # 创建有向图
        # 通过创建这个有向图对象dot，后续可以通过添加节点、边等操作来构建具体的有向图内容
        dot = graphviz.Digraph(
                comment='MindMap',
                format='jpg',
                engine='dot' # dot是graphviz中用于布局和渲染图形的引擎之一。
            )
        dot.attr(rankdir='LR')  # 从左到右布局
        # 设置图形属性
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        # 添加根节点（主题）
        root_id = 'root'
        clean_topic = clean_text(topic)
        dot.node(root_id, clean_topic, fillcolor='lightblue')
        
        # 用于存储已创建的节点ID  
        created_nodes = {root_id} 
        
        # 添加所有其他节点和边
        for parent, child, level in nodes:
            # 为每个节点生成唯一ID  
            node_id = f"node_{uuid.uuid4().hex[:8]}"  
            
            # 清理节点文本  
            clean_child = clean_text(child) 
            
            # 根据层级设置不同的颜色  
            colors = ['lightblue', 'lightgreen', 'lightyellow']  
            color = colors[min(level, len(colors)-1)]  
            # 添加节点  
            dot.node(node_id, clean_child, fillcolor=color)  
            
            # if parent == 'ROOT':
            #     parent = topic
            
            # # 根据层级设置不同的颜色
            # colors = ['lightblue', 'lightgreen', 'lightyellow']
            # color = colors[min(level, len(colors)-1)]
            
            # # 添加节点和边
            # dot.node(child, child, fillcolor=color)
            # dot.edge(parent, child)
            
            
            if level==0:
                dot.edge(root_id, node_id)
            else:
                # 找到父节点的ID  
                parent_nodes = [n for n in created_nodes if clean_text(parent) in dot.body]  
                if parent_nodes:  
                    dot.edge(parent_nodes[-1], node_id)  
            created_nodes.add(node_id)  
            
            
        # 创建临时文件夹来保存图片
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'mindmap.png')
        
        # 渲染并保存图片
        dot.render(os.path.join(temp_dir, 'mindmap'), format='png', cleanup=True)
        
        return output_path

def generate_mindmap(topic: str) -> str:
    """
    Gradio接口函数
    Args:
        topic: 用户输入的主题
    Returns:
        生成的思维导图图片路径
    """
    generator = MindMapGenerator()
    content = generator.generate_mindmap_content(topic)
    nodes = generator.parse_hierarchy(content)
    image_path = generator.create_mindmap(topic, nodes)
    
    return image_path

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("""
    # AI思维导图生成器
    输入一个主题，AI将为您生成相应的思维导图。
    """)
    
    with gr.Row():
        topic_input = gr.Textbox(label="输入主题", placeholder="例如：人工智能、机器学习、Python编程...")
        generate_btn = gr.Button("生成思维导图")
    
    with gr.Row():
        # 使用Image组件显示生成的思维导图
        mindmap_output = gr.Image(label="生成的思维导图")
    
    generate_btn.click(
        fn=generate_mindmap,
        inputs=[topic_input],
        outputs=[mindmap_output]
    )

# if __name__ == "__main__":
#     demo.launch()