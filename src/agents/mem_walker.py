import os
from typing import List, Optional, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.llm import LLMChain
from zhipuai import ZhipuAI

class MemoryTreeNode:
    def __init__(self, content: str, level: int, children: List['MemoryTreeNode'] = None):
        self.content = content
        self.level = level
        self.children = children or []
        self.parent = None

class MemoryTreeBuilder:
    def __init__(self, chunk_size=1000, max_children=5):
        self.chunk_size = chunk_size
        self.max_children = max_children
        
    async def build_tree(self, text: str, model_type: str) -> MemoryTreeNode:
        # 文本分块
        chunks = self._chunk_text(text)
        
        # 构建叶子节点
        leaf_nodes = await self._create_leaf_nodes(chunks, model_type)
        
        # 递归构建上层节点
        current_level = leaf_nodes
        level = 1
        while len(current_level) > 1:
            parent_nodes = []
            for i in range(0, len(current_level), self.max_children):
                children = current_level[i:i+self.max_children]
                parent_content = await self._summarize_nodes(children, model_type)
                parent_node = MemoryTreeNode(parent_content, level, children)
                for child in children:
                    child.parent = parent_node
                parent_nodes.append(parent_node)
            current_level = parent_nodes
            level += 1
        return current_level[0]

    def _chunk_text(self, text: str) -> List[str]:
        # 简化的文本分块实现
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    async def _create_leaf_nodes(self, chunks: List[str], model_type: str) -> List[MemoryTreeNode]:
        nodes = []
        for chunk in chunks:
            summary = await self._call_model(
                prompt_template="leaf_summary_prompt",
                inputs={"text": chunk},
                model_type=model_type
            )
            nodes.append(MemoryTreeNode(summary, level=0))
        return nodes

    async def _summarize_nodes(self, nodes: List[MemoryTreeNode], model_type: str) -> str:
        combined = "\n\n".join([n.content for n in nodes])
        return await self._call_model(
            prompt_template="parent_summary_prompt",
            inputs={"summaries": combined},
            model_type=model_type
        )

class Navigator:
    def __init__(self, model_type: str = "api"):
        self.model_type = model_type
        self.working_memory = []
        
        # 初始化提示模板
        self.triage_prompt = ChatPromptTemplate.from_template("""
        The following are summaries of different text parts:
        {summaries}
        
        Query: {query}
        Which summary is MOST relevant? First reason, then choose action.
        Format: Reasoning:... Action: [number]
        """)
        
        self.leaf_prompt = ChatPromptTemplate.from_template("""
        [Working Memory] {working_memory}
        
        Text: {text}
        Can this answer the query? {query}
        Format: Reasoning:... Action: -1/-2 Answer: [optional]
        """)

    async def navigate(self, root: MemoryTreeNode, query: str) -> str:
        current_node = root
        while True:
            if current_node.level == 0:  # Leaf node
                response = await self._handle_leaf(current_node, query)
                if "Answer:" in response:
                    return response.split("Answer:")[-1].strip()
                else:
                    current_node = current_node.parent
            else:  # Non-leaf node
                current_node = await self._handle_triage(current_node, query)

    async def _handle_triage(self, node: MemoryTreeNode, query: str) -> MemoryTreeNode:
        summaries = {i: child.content for i, child in enumerate(node.children)}
        response = await self._call_model(
            prompt=self.triage_prompt,
            inputs={"summaries": "\n".join([f"Summary {i}: {s}" for i, s in summaries.items()]), "query": query},
            model_type=self.model_type
        )
        # 解析响应
        action = int(response.split("Action:")[-1].strip())
        self.working_memory.append(node.children[action].content[:200])  # 保存工作记忆
        return node.children[action]

    async def _handle_leaf(self, node: MemoryTreeNode, query: str) -> str:
        response = await self._call_model(
            prompt=self.leaf_prompt,
            inputs={
                "working_memory": "\n".join(self.working_memory[-3:]),  # 保留最近3条记忆
                "text": node.content,
                "query": query
            },
            model_type=self.model_type
        )
        return response

    async def _call_model(self, prompt: ChatPromptTemplate, inputs: Dict, model_type: str) -> str:
        if model_type == "api":
            client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
            response = client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": prompt.format(**inputs)}]
            )
            return response.choices[0].message.content
        elif model_type == "huggingface":
            llm = HuggingFacePipeline.from_model_id(
                model_id="meta-llama/Llama-2-70b-chat-hf",
                task="text-generation"
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            return await chain.arun(inputs)
        else:
            raise ValueError("Invalid model_type")

# 使用示例
async def main():
    # 初始化构建器
    builder = MemoryTreeBuilder()
    
    # 构建内存树
    with open("long_text.txt") as f:
        text = f.read()
    root = await builder.build_tree(text, model_type="api")
    
    # 执行导航
    navigator = Navigator(model_type="api")
    answer = await navigator.navigate(root, "用户的问题")
    print("Final Answer:", answer)
    
    
    '''
    async关键字用于定义异步函数（也称为协程）。在您提供的代码中，async def build_tree定义了一个异步方法。它的主要作用包括：

        非阻塞执行：异步函数允许程序在等待I/O操作（如网络请求、文件读写等）时，可以暂停当前任务并执行其他任务，而不是阻塞整个程序。

        提高效率：在处理大量I/O密集型任务时，异步编程可以显著提高程序的执行效率，因为它可以同时处理多个任务，而不是顺序执行。

        与await配合使用：在异步函数内部，可以使用await关键字来等待其他异步操作完成。例如在build_tree方法中，await self._create_leaf_nodes和await self._summarize_nodes就是等待这些异步操作完成。
    
    '''