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

class MyPromptTemplate:  
    def __init__(self):  
        self.tools: Dict[str, ToolTemplate] = {
            
        }  
        self.base_prompt = """你是一个旅游路线规划助手，你的任务是使用合适的工具来获取旅行相关信息，并规划最准确、舒适、快捷的旅游路线。请遵循以下原则：  
                        1. 按逻辑顺序使用工具（如先查天气再订酒店）  
                        2. 确保参数格式正确（日期使用YYYY-MM-DD格式）  
                        3. 优先使用最新数据  
                        4. 当需要用户确认时，用自然语言询问  

                        你可以使用以下工具："""  
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
                {(history[:500] if len(history) > 500 else history) if history else "无"}  

                用户问题：{query}  
                
                你只能在工具列表中选择一种工具进行相应，请务必遵守下面的工具调用格式。
                
                请按照以下格式响应：  
                <思考>分析问题并选择工具</思考>  
                <工具调用>{'{工具名称}'}(参数1=值1, 参数2=值2)</工具调用>
                
            
                """
                
    def _build_hotel_template(self):  
        return ToolTemplate(  
            name="book_hotel",  
            description="酒店预订服务，支持根据位置、预算和日期筛选",  
            parameters=[  
                ToolParameter(name="location", type="str", description="城市名称或具体地址"),  
                ToolParameter(name="check_in", type="str", description="入住日期（YYYY-MM-DD）"),  
                ToolParameter(name="check_out", type="str", description="退房日期（YYYY-MM-DD）"),  
                ToolParameter(name="budget", type="int", description="每晚预算（人民币）", required=False),  
                ToolParameter(name="room_type", type="str", description="房型要求，如大床房/双床房", required=False)  
            ],  
            call_template="book_hotel(location={location}, check_in={check_in}, check_out={check_out})"  
        )  

    def _build_plane_template(self):  
        return ToolTemplate(  
            name="book_flight",  
            description="机票预订服务，支持多城市查询和价格比较",  
            parameters=[  
                ToolParameter(name="departure", type="str", description="出发城市机场代码（如PEK）"),  
                ToolParameter(name="destination", type="str", description="到达城市机场代码（如SHA）"),  
                ToolParameter(name="date", type="str", description="出发日期（YYYY-MM-DD）"),  
                ToolParameter(name="seat_class", type="str", description="舱位等级：economy/business/first", required=False)  
            ],  
            call_template="book_flight(departure={departure}, destination={destination}, date={date})"  
        )  

    def _build_transportation_template(self):  
        return ToolTemplate(  
            name="find_fastest_route",  
            description="实时交通路线规划，支持多种交通方式",  
            parameters=[  
                ToolParameter(name="start", type="str", description="起点坐标或地址"),  
                ToolParameter(name="end", type="str", description="终点坐标或地址"),  
                ToolParameter(name="transport_type", type="str", description="交通方式：driving/walking/transit", required=False)  
            ],  
            call_template="find_fastest_route(start={start}, end={end})"  
        )  

    def _build_weather_template(self):  
        return ToolTemplate(  
            name="get_weather",  
            description="多日天气预报查询服务",  
            parameters=[  
                ToolParameter(name="location", type="str", description="城市名称或邮政编码"),  
                ToolParameter(name="date", type="str", description="查询日期（YYYY-MM-DD）")  
            ],  
            call_template="get_weather(location={location}, date={date})"  
        )  

