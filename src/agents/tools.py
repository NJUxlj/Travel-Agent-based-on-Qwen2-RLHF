import requests  
import json  
import re  
from typing import Dict, Any, List
import os

from bs4 import BeautifulSoup 
from baidusearch.baidusearch import search
from urllib.parse import quote_plus  
import time
import random

from prompt_template import PromptTemplate



# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'



class ToolDispatcher:  
    def __init__(self):  
        self.executors = {  
            "google_search": GoogleSearchExecutor(  
                api_key=CUSTOM_SEARCH_API_KEY,  
                search_engine_id=CUSTOM_SEARCH_ENGINE_ID  
            ),
        }  
        
        self.prompt_template = PromptTemplate()
    
    def parse_tool_call(self, tool_str: str) -> Dict:  
        """解析工具调用字符串"""  
        pattern = r"(\w+)\((.*)\)"  
        match = re.match(pattern, tool_str)  
        if not match:  
            return None  
        
        tool_name = match.group(1)  
        args_str = match.group(2)  
        
        # 解析参数键值对  
        args = {}  
        for pair in re.findall(r"(\w+)=([^,]+)", args_str):  
            key = pair[0]  
            value = pair[1].strip("'")
            if re.match(r'^-?\d+$', value):  # 支持负整数
                value = int(value)
            args[key] = value  
        
        return {"tool": tool_name, "args": args}  

    def execute(self, tool_call: str) -> Dict:  
        """执行工具调用"""  
        parsed = self.parse_tool_call(tool_call)  
        if not parsed:  
            return {"error": "Invalid tool format"}  
        
        executor = self.executors.get(parsed["tool"])  
        if not executor:  
            return {"error": "Tool not registered"}  
        
        # 获取工具参数规范  
        tool_template = self.prompt_template.tools.get(parsed["tool"])  
        if not tool_template:  
            return {"error": "Tool template not found"}  
        
        
        # 参数类型校验  
        for param in tool_template.parameters:  
            if param.required and param.name not in parsed["args"]:  
                return {"error": f"Missing required parameter: {param.name}"}  
            if param.name in parsed["args"]:  
                expected_type = param.type  
                actual_value = parsed["args"][param.name]  
                if not isinstance(actual_value, eval(expected_type)):  
                    return {"error": f"Type mismatch for {param.name}, expected {expected_type}"}  
        
        print( "parse_args = ", parsed["args"])
        
        # parsed["args"] = {"query":..., "max_results":...}
        return executor.execute(**parsed["args"]) 




class GoogleSearchExecutor:  
    def __init__(self, api_key: str, search_engine_id: str):  
        self.base_url = "https://www.googleapis.com/customsearch/v1"  
        self.api_key = api_key  
        self.search_engine_id = search_engine_id  

    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:  
        params = {  
            "key": self.api_key,  
            "cx": self.search_engine_id,  
            "q": query,  
            "num": max_results  
        }  
        
        try:  
            response = requests.get(self.base_url, params=params)  
            response.raise_for_status()  
            return self._parse_results(response.json())  
        except Exception as e:  
            return {"error": str(e)}  

    def _parse_results(self, data: Dict) -> Dict:  
        """解析Google API响应"""  
        return {  
            "items": [{  
                "title": item.get("title"),  
                "link": item.get("link"),  
                "snippet": item.get("snippet")  
            } for item in data.get("items", [])]  
        }  


class WeatherExecutor:
    pass


class HotelExecutor:
    pass



class PlaneTicketExecutor:
    pass



class TransportationExecutor:
    pass
