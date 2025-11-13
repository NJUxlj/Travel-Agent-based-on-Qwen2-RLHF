引言：超越传统提示的局限性
在早期的大语言模型（LLM）应用中，提示工程（Prompt Engineering）是连接用户意图与模型输出的核心手段。然而，纯提示驱动的方案在多步推理、工具调用和动态环境交互中常显不足。ReAct框架（Reasoning+Acting）应运而生，通过将链式推理（Reasoning） 与环境行动（Acting） 结合，构建出能主动思考、决策并执行复杂任务的智能体（Agent）。



1. 推理（Reasoning）模块

动态思考链（Chain-of-Thought） ：Agent在每一步生成自然语言推理逻辑，解释当前决策原因（如：“用户需要查天气，需先获取位置信息”）。
错误回溯机制：当行动失败时，Agent能分析原因并调整策略（如：“API返回错误，可能是参数格式问题，重试前需校验输入”）。

2. 行动（Acting）模块

工具集成（Tool Calling） ：调用外部API、数据库、计算器等（如：search_weather(location="Beijing")）。
环境状态感知：实时接收行动结果，作为下一步决策的输入（如：“获取到北京气温25°C，建议用户带薄外套”）。

3.交互闭环：Reasoning与Acting的循环
ReAct的执行流程形成自迭代循环：
sql 体验AI代码助手 代码解读复制代码Thought → Act → Observation → Thought → ... → Final Answer

每一次循环都将环境反馈纳入推理，实现动态适应复杂场景的能力。



二、技术优势：为什么ReAct优于传统Agent？

1. 解决LLM的“幻觉”问题
传统Agent可能盲目执行错误指令，而ReAct的显式推理步骤让决策过程可追溯，减少无依据输出。
案例：当用户问“爱因斯坦最近的推特说了什么？”，ReAct会先推理：“爱因斯坦已去世，需搜索历史资料库而非实时社交媒体”。
2. 支持长任务分解
复杂任务被拆解为原子化推理-行动对（ReAct Pair）执行：

```yaml
# 用户请求：“分析特斯拉过去一年的股价趋势并总结原因”
Steps:
1. Thought: 需要特斯拉股票代码 → Action: search_stock_symbol("Tesla")
2. Observation: 获得代码TSLA
3. Thought: 查询过去一年股价 → Action: get_stock_data("TSLA", period="1y")
4. Observation: 接收时间序列数据
5. Thought: 调用数据分析模型 → Action: analyze_trend(data)
```

3.无缝集成领域工具
通过工具注册机制（如LangChain Tools），Agent可灵活扩展能力边界：

```python
from langchain_community.tools import DuckDuckGoSearchResults
tools = [
DuckDuckGoSearchResults(name="web_search"),
PythonCalculatorTool() # 自定义Python计算器
]
agent = ReActAgent(llm=GPT4, tools=tools)
```




## 简易的 ReAct 循环实现
```python
from langchain.agents import ReActAgent, Tool
from langchain.llms import OpenAI

def web_search(query: str) -> str:
# 调用搜索API
    return f"Results about {query}"

tools = [
    Tool(name="Search", func=web_search, description="Search the web")
]
agent = ReActAgent(llm=OpenAI(temperature=0), tools=tools)

task = "巴黎埃菲尔铁塔高度是多少米？"

for step in range(3): # 最大迭代步数
    output = agent.generate_step(task)
    print(output)
    if "Final Answer" in output:
        print(output)
        break
    else:
        # 执行Action并更新环境反馈
        action_result = execute_action(output)
        task = task + f"\nObservation: {action_result}"

```