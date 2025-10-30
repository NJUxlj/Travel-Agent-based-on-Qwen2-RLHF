import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Optional, Tuple
import re
import graphviz
from pathlib import Path
import tempfile
import os
import uuid
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
try:
    from configs.config import MODEL_CONFIG
    model_path = MODEL_CONFIG['model']['name']
except ImportError:
    logger.warning("未找到配置文件，使用默认模型")
    model_path = "Qwen/Qwen2-7B-Chat"  # 默认模型路径


import subprocess
def check_graphviz_installed() -> bool:
    """
    检查系统是否安装了graphviz
    Returns:
        bool: 如果安装了graphviz返回True，否则返回False
    """
    try:
        # 运行命令 dot -V，检查是否安装了graphviz
        result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
        # 即使返回非零状态码也可能表示已安装但版本命令有问题
        logger.info(f"Graphviz检查结果: {result.returncode}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"未检测到Graphviz安装: {str(e)}")
        return False


def clean_text(text: str, truncate_length: int = 100) -> str:
    """
    清理文本，移除或替换可能导致graphviz语法错误的字符
    
    Args:
        text: 需要清理的文本
        truncate_length: 文本截断长度
    
    Returns:
        str: 清理后的文本
    """
    if not text or not isinstance(text, str):
        return "node"
    
    # 移除或替换可能导致graphviz语法错误的特殊字符
    # 保留字母、数字、空格、下划线和连字符
    text = re.sub(r'[\\/:*?"<>|\[\]{}()]', '_', text)
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    
    # 确保文本不为空
    text = text.strip() or "node"
    
    # 限制文本长度
    if len(text) > truncate_length:
        text = text[:truncate_length-3] + "..."
    
    return text

class MindMapGenerator:
    def __init__(
        self, 
        model_name: str = model_path,
        level_num: int = 3, 
        item_num: int = 15, 
        max_new_tokens: int = 1024,
        use_cache: bool = True
    ):
        """
        初始化思维导图生成器
        
        Args:
            model_name: Hugging Face模型名称
            level_num: 思维导图最大层级数
            item_num: 思维导图最大项目数
            max_new_tokens: 生成新token的最大数量
            use_cache: 是否使用模型缓存以提高性能
        """
        try:
            # 加载模型和分词器
            logger.info(f"正在加载模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                use_cache=use_cache
            )
            self.model.eval()  # 设置为评估模式
            logger.info(f"模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise RuntimeError(f"无法加载模型: {str(e)}")
        
        # 配置参数
        self.level_num = max(1, min(5, level_num))  # 限制层级在1-5之间
        self.item_num = max(5, min(50, item_num))   # 限制项目数在5-50之间
        self.max_new_tokens = max(512, min(2048, max_new_tokens))  # 限制token数在512-2048之间
        self.use_cache = use_cache
        
    def generate_mindmap_content(self, topic: str) -> str:
        """
        使用大模型生成思维导图内容
        
        Args:
            topic: 用户输入的主题
            
        Returns:
            str: 生成的思维导图内容（层级列表格式）
        """
        if not topic or not isinstance(topic, str) or not topic.strip():
            logger.error("无效的主题输入")
            return f"- 错误：无效的主题输入\n  - 请提供有效的主题"
        
        # 优化的提示词设计
        prompt = f"""请为主题："{topic}" 创建一个详细的思维导图。
        
        输出格式要求：
        1. 使用层级结构，每个层级使用缩进来表示
        2. 每个节点前使用减号(-)标记
        3. 第一层节点不缩进，第二层缩进2个空格，第三层缩进4个空格，以此类推
        4. 请确保内容逻辑清晰，层次分明
        5. 限制为最多{self.level_num}个层级，总共不超过{self.item_num}个项目
        
        示例格式：
        - 主题
          - 主要分类1
            - 详细信息1
            - 详细信息2
          - 主要分类2
            - 详细信息3
        
        请开始生成思维导图："""
        
        try:
            logger.info(f"开始生成思维导图内容，主题: {topic}")
            
            # 准备输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 生成响应
            with torch.no_grad():  # 减少内存使用
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.6,  # 略微降低随机性
                    top_p=0.85,       # 略微提高确定性
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # 减少重复内容
                )
            
            # 解码响应
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()  # 移除提示词部分
            
            logger.debug(f"原始模型响应: {response[:200]}...")
            
            # 提取层级列表内容
            # 尝试多种提取方法以提高鲁棒性
            
            # 方法1: 使用正则表达式提取层级列表
            pattern = r'(?:^|\n)(-\s+[^\n]+(?:\n\s+-\s+[^\n]+)*)'  
            matches = list(re.finditer(pattern, response, re.MULTILINE))
            
            if matches:
                content = matches[-1].group(0)
            else:
                # 方法2: 查找以减号开头的行集合
                lines = response.split('\n')
                mindmap_lines = []
                in_mindmap = False
                
                for line in lines:
                    if re.match(r'^\s*-\s+', line):
                        mindmap_lines.append(line)
                        in_mindmap = True
                    elif in_mindmap and line.strip() == '':
                        # 允许空行在思维导图内部
                        continue
                    elif in_mindmap and not re.match(r'^\s*$', line):
                        # 如果不在继续思维导图格式，停止收集
                        break
                
                if mindmap_lines:
                    content = '\n'.join(mindmap_lines)
                else:
                    # 如果没有找到有效的思维导图格式，创建一个简单的默认结构
                    logger.warning("无法从模型响应中提取有效的思维导图格式，使用默认结构")
                    content = f"- {topic}\n  - 主要分类1\n  - 主要分类2\n  - 主要分类3"
            
            # 验证和清理内容
            content = content.strip()
            
            logger.info(f"思维导图内容生成完成，共 {content.count('\n') + 1} 行")
            return content
            
        except Exception as e:
            logger.error(f"生成思维导图内容时出错: {str(e)}")
            # 返回错误状态的思维导图
            return f"- {topic}\n  - 错误: 生成内容时发生错误\n  - 详细信息: {str(e)}"

    def parse_hierarchy(self, content: str) -> List[tuple]:
        """
        解析层级列表内容为节点关系
        
        Args:
            content: 生成的层级列表内容
            
        Returns:
            List[tuple]: 节点关系列表 [(parent, child, level)]
        """
        if not content or not isinstance(content, str):
            logger.error("无效的内容输入")
            return []
        
        lines = content.strip().split('\n')
        nodes = []
        # 使用字典存储每个层级的最后一个节点，更加灵活
        level_nodes = {}
        
        logger.info(f"开始解析层级结构，共 {len(lines)} 行内容")
        
        for line_idx, line in enumerate(lines):
            # 跳过空行
            if not line.strip():
                continue
            
            try:
                # 计算缩进级别（每2个空格为一级）
                leading_spaces = len(re.match(r'^\s*', line).group())
                indent_level = leading_spaces // 2
                
                # 提取文本内容（移除减号和周围空格）
                text_match = re.search(r'-\s*(.+)', line)
                if text_match:
                    text = text_match.group(1).strip()
                else:
                    logger.warning(f"第{line_idx+1}行格式不正确，跳过: {line}")
                    continue
                
                # 限制层级深度
                if indent_level > self.level_num - 1:
                    logger.warning(f"第{line_idx+1}行层级过深，调整为最大层级: {indent_level} -> {self.level_num-1}")
                    indent_level = self.level_num - 1
                
                # 确定父节点
                if indent_level == 0:
                    # 根节点下的一级节点
                    parent = 'ROOT'
                else:
                    # 查找上一级的最后一个节点作为父节点
                    parent_level = indent_level - 1
                    # 如果找不到父级节点，使用最近的可用层级
                    while parent_level >= 0 and parent_level not in level_nodes:
                        parent_level -= 1
                    
                    if parent_level >= 0:
                        parent = level_nodes[parent_level]
                    else:
                        # 如果完全找不到父级，默认为根节点下的节点
                        parent = 'ROOT'
                        indent_level = 0
                
                # 添加节点关系
                nodes.append((parent, text, indent_level))
                
                # 更新当前层级的最后一个节点
                level_nodes[indent_level] = text
                
                logger.debug(f"解析节点: 父='{parent}', 子='{text}', 层级={indent_level}")
                
                # 限制总节点数
                if len(nodes) >= self.item_num:
                    logger.warning(f"节点数已达到限制 ({self.item_num})，停止解析")
                    break
                    
            except Exception as e:
                logger.error(f"解析第{line_idx+1}行时出错: {str(e)}, 行内容: {line}")
                continue
        
        logger.info(f"层级解析完成，共解析 {len(nodes)} 个节点")
        return nodes

    def create_mindmap(self, topic: str, nodes: List[tuple]) -> str:
        """
        使用graphviz创建思维导图
        
        Args:
            topic: 主题
            nodes: 节点关系列表
            
        Returns:
            str: 生成的图片路径
            
        Raises:
            RuntimeError: 当Graphviz未安装或渲染失败时
        """
        # 检查Graphviz是否安装
        if not check_graphviz_installed():  
            logger.error("未检测到Graphviz安装")
            # 创建一个备用的错误提示图片路径
            error_img_path = self._create_error_image("请安装Graphviz以生成思维导图")
            return error_img_path
        
        try:
            logger.info(f"开始创建思维导图，主题: {topic}")
            
            # 创建有向图，使用更美观的布局引擎
            dot = graphviz.Digraph(
                comment='MindMap',
                format='png',  # 使用PNG格式
                engine='dot',  # 可以尝试'neato', 'fdp'等其他引擎
                graph_attr={
                    'rankdir': 'LR',  # 从左到右布局
                    'nodesep': '0.5',  # 节点间距
                    'ranksep': '1.2',  # 层级间距
                    'fontname': 'SimHei, Arial Unicode MS',  # 支持中文的字体
                    'fontsize': '14'
                }
            )
            
            # 设置节点默认样式
            dot.attr('node', 
                     shape='box', 
                     style='rounded,filled', 
                     fillcolor='lightblue',
                     fontname='SimHei, Arial Unicode MS',
                     fontsize='12',
                     height='0.5',
                     width='1.5'
                    )
            
            # 设置边的样式
            dot.attr('edge', 
                     arrowhead='normal',
                     penwidth='1.0'
                    )
            
            # 添加根节点（主题）
            root_id = 'root'
            clean_topic = clean_text(topic)
            dot.node(root_id, 
                     clean_topic, 
                     fillcolor='#a1caf1',  # 根节点使用较深的蓝色
                     fontsize='16',
                     fontweight='bold',
                     style='rounded,filled,bold'
                    )
            
            # 用于存储节点ID映射（文本 -> ID）
            node_map = {root_id: root_id}
            text_to_id = {clean_topic: root_id}
            
            # 使用更丰富的颜色方案
            colors = [
                '#a1caf1',  # 浅蓝色 (根节点)
                '#a5d6a7',  # 浅绿色 (一级节点)
                '#fff59d',  # 浅黄色 (二级节点)
                '#ffcc80',  # 浅橙色 (三级节点)
                '#ce93d8'   # 浅紫色 (四级节点)
            ]
            
            # 添加所有其他节点和边
            for parent_text, child_text, level in nodes:
                try:
                    # 为每个节点生成唯一ID
                    child_id = f"node_{uuid.uuid4().hex[:8]}"
                    clean_child = clean_text(child_text)
                    
                    # 根据层级选择颜色
                    color_idx = min(level + 1, len(colors) - 1)  # +1 因为根节点已经用了colors[0]
                    node_color = colors[color_idx]
                    
                    # 添加子节点
                    dot.node(
                        child_id, 
                        clean_child, 
                        fillcolor=node_color,
                        style='rounded,filled'
                    )
                    
                    # 记录节点映射
                    node_map[child_id] = child_id
                    text_to_id[clean_child] = child_id
                    
                    # 确定父节点ID
                    if parent_text == 'ROOT':
                        parent_id = root_id
                    else:
                        # 尝试通过文本查找父节点ID
                        clean_parent = clean_text(parent_text)
                        # 如果直接找不到，尝试在已创建的节点中查找相似文本
                        parent_id = text_to_id.get(clean_parent)
                        
                        # 如果还是找不到，使用根节点作为父节点
                        if not parent_id:
                            logger.warning(f"找不到父节点: {parent_text}，使用根节点作为父节点")
                            parent_id = root_id
                    
                    # 添加边
                    dot.edge(parent_id, child_id)
                    
                    logger.debug(f"添加节点: {clean_child}，父节点: {parent_text if parent_text != 'ROOT' else 'root'}")
                    
                except Exception as e:
                    logger.error(f"添加节点时出错: {str(e)}, 节点: {child_text}")
                    continue
            
            # 创建唯一的临时文件名，避免冲突
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.mkdtemp(prefix=f"mindmap_{timestamp}_")
            base_filename = os.path.join(temp_dir, f"mindmap_{timestamp}")
            output_path = f"{base_filename}.png"
            
            logger.info(f"正在渲染思维导图到: {output_path}")
            
            # 渲染并保存图片
            # 使用quiet=True减少输出，format参数确保输出为PNG
            dot.render(base_filename, format='png', cleanup=True, quiet=True)
            
            # 验证文件是否生成成功
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"思维导图文件生成失败: {output_path}")
            
            logger.info(f"思维导图生成成功，文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
            return output_path
            
        except Exception as e:
            logger.error(f"创建思维导图时出错: {str(e)}")
            # 创建错误提示图片
            error_img_path = self._create_error_image(f"生成思维导图失败: {str(e)}")
            return error_img_path
    
    def _create_error_image(self, error_message: str) -> str:
        """
        创建错误提示图片
        
        Args:
            error_message: 错误信息
            
        Returns:
            str: 错误图片路径
        """
        try:
            # 创建一个简单的错误提示图形
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.mkdtemp(prefix=f"error_{timestamp}_")
            base_filename = os.path.join(temp_dir, f"error_{timestamp}")
            output_path = f"{base_filename}.png"
            
            dot = graphviz.Digraph(format='png')
            dot.attr('node', shape='box', style='filled', fillcolor='#ffcdd2')
            dot.node('error', f"错误\n{clean_text(error_message, 100)}")
            
            dot.render(base_filename, format='png', cleanup=True, quiet=True)
            return output_path
        except:
            # 如果连错误图片都生成不了，返回一个空的临时文件
            fd, path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
            return path

def generate_mindmap(topic: str, 
                    model_name: Optional[str] = None, 
                    level_num: Optional[int] = None, 
                    item_num: Optional[int] = None, 
                    use_cache: bool = True) -> str:
    """
    生成思维导图的公共接口
    
    Args:
        topic: 思维导图主题
        model_name: 可选，使用的模型名称
        level_num: 可选，思维导图层级数量
        item_num: 可选，每层的项目数量
        use_cache: 是否使用缓存，默认为True
        
    Returns:
        str: 生成的思维导图图片路径
        
    Example:
        >>> img_path = generate_mindmap("旅行规划", level_num=3, item_num=5)
    """
    logger.info(f"调用公共接口生成思维导图，主题: {topic}")
    
    # 验证主题输入
    if not topic or not isinstance(topic, str) or len(topic.strip()) == 0:
        logger.error("无效的主题输入：主题不能为空且必须为字符串")
        # 创建错误提示图片
        error_img_path = _create_error_image_for_public("错误：请提供有效的思维导图主题")
        return error_img_path
    
    try:
        # 初始化思维导图生成器
        generator_kwargs = {}
        if model_name is not None:
            generator_kwargs['model_name'] = model_name
        if level_num is not None:
            generator_kwargs['level_num'] = level_num
        if item_num is not None:
            generator_kwargs['item_num'] = item_num
        generator_kwargs['use_cache'] = use_cache
        generator_kwargs['max_new_tokens'] = 1024  # 保留原有的token数量设置
        
        generator = MindMapGenerator(**generator_kwargs)
        logger.info(f"思维导图生成器初始化完成，参数: {generator_kwargs}")
        
        # 生成思维导图内容
        logger.info("开始生成思维导图内容")
        content = generator.generate_mindmap_content(topic)
        
        if not content:
            raise ValueError("未能生成有效的思维导图内容")
        
        # 解析层级结构
        logger.info("开始解析思维导图层级结构")
        nodes = generator.parse_hierarchy(content)
        
        if not nodes:
            raise ValueError("未能解析出有效的节点层级结构")
        
        # 创建思维导图图片
        logger.info("开始创建思维导图图片")
        image_path = generator.create_mindmap(topic, nodes)
        
        logger.info(f"思维导图生成完成，图片路径: {image_path}")
        return image_path
        
    except Exception as e:
        logger.error(f"生成思维导图时发生错误: {str(e)}")
        # 创建错误提示图片
        error_img_path = _create_error_image_for_public(f"生成思维导图失败: {str(e)}")
        return error_img_path


def _create_error_image_for_public(error_message: str) -> str:
    """
    为公共接口创建错误提示图片
    
    Args:
        error_message: 错误信息
        
    Returns:
        str: 错误图片路径
    """
    try:
        # 创建一个简单的错误提示图形
        import tempfile
        import os
        from datetime import datetime
        import graphviz
        import uuid
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.mkdtemp(prefix=f"error_public_{timestamp}_")
        base_filename = os.path.join(temp_dir, f"error_public_{timestamp}")
        output_path = f"{base_filename}.png"
        
        # 清理错误消息
        def simple_clean_text(text, max_length=120):
            """简单的文本清理函数"""
            import re
            # 移除特殊字符，只保留基本的中文、英文、数字和常见标点
            cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.!?，。！？]', '', text)
            # 截断过长文本
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length] + '...'
            return cleaned
        
        # 创建错误提示图形
        dot = graphviz.Digraph(format='png')
        dot.attr(
            'node', 
            shape='box', 
            style='filled', 
            fillcolor='#ffcdd2',
            fontname='SimHei, Arial Unicode MS',
            fontsize='12'
        )
        
        # 分割错误信息为多行，每行最多30个字符
        cleaned_message = simple_clean_text(error_message)
        lines = []
        current_line = ""
        
        for char in cleaned_message:
            if len(current_line) >= 30:
                lines.append(current_line)
                current_line = char
            else:
                current_line += char
        
        if current_line:
            lines.append(current_line)
        
        # 最多显示5行
        display_lines = lines[:5]
        if len(lines) > 5:
            display_lines[-1] += '...'
        
        # 创建多行错误消息
        multiline_message = "\n".join(display_lines)
        
        dot.node('error', f"错误\n{multiline_message}")
        
        # 渲染图片
        dot.render(base_filename, format='png', cleanup=True, quiet=True)
        
        # 验证文件是否生成
        if os.path.exists(output_path):
            return output_path
        
    except Exception as e:
        logger.error(f"创建错误提示图片失败: {str(e)}")
    
    # 如果所有方法都失败，返回一个空的临时文件
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    return path

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