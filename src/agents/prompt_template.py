from typing import Dict, List, Optional  
from pydantic import BaseModel  

class ToolParameter(BaseModel):  
    """工具参数规范"""  
    name: str  
    type: str  
    description: str  
    required: bool = True  

class ToolTemplate:  
    def __init__(self,   
                 name: str,  
                 description: str,  
                 parameters: List[ToolParameter],  
                 call_template: str):  
        """  
        Args:  
            name: 工具名称（英文标识）  
            description: 自然语言描述  
            parameters: 参数列表  
            call_template: 调用模板，如：google_search(query)  
        """  
        self.name = name  
        self.description = description  
        self.parameters = parameters  
        self.call_template = call_template  

class PromptTemplate:  
    def __init__(self):  
        self.tools: Dict[str, ToolTemplate] = {
            
        }  
        self.base_prompt = """你是一个旅游路线规划助手，\
                        你的任务是使用合适的工具（例如：酒店预定、机票预定、地图上的最快交通路线）来获取旅行相关信息，并以此来规划最准确、舒适、快捷的旅游路线，你可以使用以下工具："""  
        
        # 预置常用工具  
        self.register_tool(self._build_hotel_template())  
        self.register_tool(self._build_plane_template())  
        self.register_tool(self._build_transportation_template())  
        self.register_tool(self._build_weather_template())
    
    def register_tool(self, tool: ToolTemplate):  
        """注册新工具"""  
        self.tools[tool.name] = tool  
    
    def generate_prompt(self, query: str, history: Optional[List] = None) -> str:  
        """生成完整提示"""  
        tools_desc = "\n".join(  
            [f"{tool.name}: {tool.description}\n参数: {[p.name for p in tool.parameters]}"  
             for tool in self.tools.values()]  
        )  
        
        return f"""{self.base_prompt}  
        
                可用工具列表：  
                {tools_desc}  

                当前对话历史：  
                {history if history else "无"}  

                用户问题：{query}  

                请按照以下格式响应：  
                <思考>分析问题并选择工具</思考>  
                <工具调用>{'{工具名称}'}(参数1=值1, 参数2=值2)</工具调用>"""

