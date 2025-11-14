import asyncio
from agent_framework import ChatAgent, MCPStdioTool, MCPStreamableHTTPTool, MCPWebsocketTool
from agent_framework.openai import OpenAIChatClient
from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.llm_config import OpenAIChatConfig

class LocalMCPServer:
    '''
    功能:
        本地 MCP 服务器类。
        - 他实际上只是个服务器包装类，用于包装真正的 mcp_server.py 脚本。
        - 他的主要作用是：
            - 启动 mcp_server.py 脚本，使得 agent-framework 可以完美适配 anthropic 的 MCP 库。
            - 提供一个异步接口，用于与 ChatAgent 进行交互。
    '''
    def __init__(
        self, 
        server_file_path = None,
        llm_config: OpenAIChatConfig = OpenAIChatConfig()):
        self.llm_config = llm_config
        self.server_file_path = server_file_path

        self.chat_client = OpenAIChatClient(
            model_id=self.llm_config.model_id,
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.endpoint,
            )
        
        self.mcp_server = MCPStdioTool(
            name="mcp_server of getting weather information within the UnitedStates", 
            command="python", 
            args=[self.server_file_path]
        )

        self.chat_agent = ChatAgent(
            chat_client=self.chat_client,
            name="WeatherForecastAgent",
            instructions="You are a helpful weather forecast assistant that can provide weather information within the United States.",
        )


    async def chat_with_mcp(self, query: str):
        """Chat with the MCP server using the given query."""

        # 使用异步上下文管理器来安全地管理MCP服务器和聊天代理的资源生命周期
        # async with语句确保：
        # 1. self.mcp_server (MCPStdioTool) 和 self.chat_agent (ChatAgent) 的资源在使用前被正确初始化
        #    - 自动调用各对象的__aenter__方法进行资源获取和初始化
        #    - MCPStdioTool负责建立与外部工具的连接和通信管道
        #    - ChatAgent负责初始化对话状态和模型接口
        # 2. 在代码块执行完毕后自动释放这些资源，防止资源泄漏
        #    - 自动调用各对象的__aexit__方法进行清理和资源释放
        #    - 确保即使在发生异常的情况下也能正确清理资源
        # 3. 两个对象都实现了__aenter__和__aexit__方法，支持异步上下文管理协议
        # 4. 在同一个async with语句中同时管理多个异步上下文管理器，提供简洁的资源管理方式
        async with (self.mcp_server, self.chat_agent):
            result = await self.chat_agent.run(
                query, 
                tools = self.mcp_server
            )

            print(result)




    


class SSEMCPServer:
    def __init__(
        self,
        server_file_path = None,
        llm_config: OpenAIChatConfig = OpenAIChatConfig()):
        self.llm_config = llm_config
        self.server_file_path = server_file_path

        self.chat_client = OpenAIChatClient(
            model_id=self.llm_config.model_id,
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.endpoint,
            )
        
        self.mcp_server = MCPStdioTool(
            name="Documentation Server", 
            command="python", 
            args=[self.server_file_path]
        )

        self.chat_agent = ChatAgent(
            chat_client=self.chat_client,
            name="DocsAgent",
            instructions="You help with documentation questions for programming languages and frameworks.",
        )

    async def chat_with_mcp(self, query: str):
        """Chat with the MCP server using the given query."""
        async with (self.mcp_server, self.chat_agent):
            result = await self.chat_agent.run(
                query, 
                tools = self.mcp_server
            )

            print(result)



class WebSocketMCPServer:
    def __init__(
        self, 
        server_file_path = None,
        llm_config: OpenAIChatConfig = OpenAIChatConfig()):
        self.llm_config = llm_config
        self.server_file_path = server_file_path

        self.chat_client = OpenAIChatClient(
            model_id=self.llm_config.model_id,
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.endpoint,
            )
        
        self.chat_agent = ChatAgent(
            chat_client=self.chat_client,
            name="DataAgent",
            instructions="You provide real-time financial data insights including stock and cryptocurrency prices.",
        )

        self.mcp_server = MCPStdioTool(
            name="Real-time Data Server", 
            command="python", 
            args=[self.server_file_path]
        )

    async def chat_with_mcp(self, query: str):
        """Chat with the MCP server using the given query."""
        async with (self.mcp_server, self.chat_agent):
            result = await self.chat_agent.run(
                query, 
                tools = self.mcp_server
            )

            print(result)   




if __name__ == "__main__":
    # 获取各种 MCP 服务器的文件路径
    server_file_path = str(Path(__file__).parent.parent / "mcp" / "mcp_server.py")
    sse_server_file_path = str(Path(__file__).parent.parent / "mcp" / "sse_mcp_server.py")
    websocket_server_file_path = str(Path(__file__).parent.parent / "mcp" / "websocket_mcp_server.py")
    
    # 创建本地 MCP 服务器实例（天气服务）
    local_mcp_server = LocalMCPServer(
        server_file_path=server_file_path,
    )
    
    # 创建 SSE MCP 服务器实例（文档服务）
    sse_mcp_server = SSEMCPServer(
        server_file_path=sse_server_file_path,
    )
    
    # 创建 WebSocket MCP 服务器实例（实时数据服务）
    websocket_mcp_server = WebSocketMCPServer(
        server_file_path=websocket_server_file_path,
    )
    
    # 测试所有三个服务器
    print("=== Testing Local MCP Server (Weather Service) ===")
    asyncio.run(local_mcp_server.chat_with_mcp("What is the weather like in New York?"))
    
    print("\n=== Testing SSE MCP Server (Documentation Service) ===")
    asyncio.run(sse_mcp_server.chat_with_mcp("Tell me about Python documentation"))
    
    print("\n=== Testing WebSocket MCP Server (Real-time Data Service) ===")
    asyncio.run(websocket_mcp_server.chat_with_mcp("Get the latest stock price for AAPL."))