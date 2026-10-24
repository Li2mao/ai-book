# Day4

# 多智能体架构

## **1\. 为什么需要多智能体**

在之前的课程中，我们构建的智能体都是**单智能体**：一个智能体完成所有任务。

但真实场景中，很多任务需要**多个智能体协作**才能完成：

\- **复杂任务分解**：一个任务太大，需要分给多个专业智能体

\- **专业分工**：不同智能体有不同的专业知识（如代码专家、写作专家、数据分析专家）

\- **协作决策**：需要多个智能体共同讨论才能做出最佳决策

\- **竞争博弈**：多个智能体竞争完成同一任务

简单来说：

\- **单智能体**：一个人做所有事

\- **多智能体**：一个团队分工合作

## **2\. 多智能体的典型架构模式**

多智能体系统有几种经典的架构模式：

### **2\.1 监督者模式（Supervisor Pattern）**

一个监督者智能体负责协调多个工作智能体：

```Plain Text
监督者
            │
    ┌──────┼──────┬──────┐
    ▼      ▼      ▼      ▼
  智能体A 智能体B 智能体C 智能体D
```

**适用场景**：任务需要多种专业能力，监督者分配任务并汇总结果。

### **2\.2 协作模式（Collaborative Pattern）**

多个智能体平等协作，共同完成任务：

```Plain Text
智能体A ←→ 智能体B
     ↑          ↑
     │          │
  智能体C ←→ 智能体D
```

**适用场景**：任务需要多个智能体之间频繁交流和协同。

### **2\.3 竞争模式（Competitive Pattern）**

多个智能体竞争完成同一任务，胜者获得奖励：

```Plain Text
智能体A ─┐
           ├──→ 比较器 → 最佳结果
  智能体B ─┘
```

**适用场景**：需要从多个方案中选择最优解。

### **2\.4 流水线模式（Pipeline Pattern）**

多个智能体按顺序处理任务，像流水线一样：

```Plain Text
智能体A → 智能体B → 智能体C → 智能体D
```

**适用场景**：任务可以分解为多个顺序步骤。

### **2\.5 辩论模式（Debate Pattern）**

多个智能体就某个问题展开辩论，最终达成共识：

```Plain Text
正方智能体 ←→ 反方智能体
        │           │
        └────┬──────┘
             ▼
          裁判智能体
```

**适用场景**：需要深入分析某个问题的正反两面。

## **3\. 第一个例子：简单的协作智能体**

我们用一个简单的例子来演示多智能体协作。

场景：

- 用户提出一个问题

- 研究智能体负责收集信息

- 写作智能体负责整理成报告

- 监督者智能体负责协调整个流程

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

_  = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

class AgentState(TypedDict):
    user_question: str
    research_info: str
    report: str 
    current_agent: str

def supervisor(state: AgentState):
    if not state['research_info']:
        return {'current_agent': 'research'}
    elif not state['report']:
        return {'current_agent': 'writer'}
    else:
        return {'current_agent': 'finish'}

def research_agent(state: AgentState):
    prompts = ChatPromptTemplate.from_messages([
        ('system', '你是一个研究员，会研究用户给的问题，并且给出一个简洁的总结回答。'),
        ('user','{user_question}')
    ])

    chain = prompts | llm | StrOutputParser()
    research_info = chain.invoke({'user_question': state['user_question']})
    return {'research_info': research_info} 

def writer_agent(state: AgentState):
    prompts = ChatPromptTemplate.from_messages([
        ('system', '你是一个写作高手，会根据用户问题，以及相关的文字提示，完成1000字的小作文'),
        ('user', '用户问题: {user_question}, 相关提示: {research_info}')
    ])

    chain = prompts | llm | StrOutputParser()
    report = chain.invoke({'user_question': state['user_question'], 'research_info': state['research_info']})
    return {'report': report} 

def route(state: AgentState):
    if state['current_agent'] == 'research':
        return 'research'
    elif state['current_agent'] == 'writer':
        return 'writer'
    elif state['current_agent'] == 'finish':
        return END
    return 'supervisor'

builder = StateGraph(AgentState)
builder.add_node('supervisor', supervisor)
builder.add_node('research', research_agent)
builder.add_node('writer', writer_agent)

builder.add_edge(START, 'supervisor')
builder.add_conditional_edges(
    'supervisor',
    route,
    {
        'research': 'research',
        'writer': 'writer',
        END: END
    }
)
builder.add_edge('research', 'supervisor')
builder.add_edge('writer', 'supervisor')

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

while True:
    question = input('>>> ')
    result = graph.invoke({
        'user_question': question,
        'research_info': '',
        'report': '',
        'current_agent': ''
    })
    print(f'研究报告: {result["research_info"]}\n')
    print(f'最终报告: {result["report"]}\n')
```

### **代码解释**

1\. **环境配置**：

- 加载 `.env` 文件中的 API 密钥

- 创建 OpenAI 客户端

- 定义 `call_llm` 函数，封装大模型调用逻辑

2\. **AgentState**：定义状态包含四个字段：

- `user_question`：用户的问题

- `research_info`：研究智能体收集的信息

- `report`：写作智能体生成的报告

- `current_agent`：当前应该执行的智能体

3\. **supervisor**：监督者智能体

- 检查当前状态，决定下一步分配给哪个智能体

- 如果没有研究信息，分配给研究智能体

- 如果有研究信息但没有报告，分配给写作智能体

- 如果报告已完成，标记任务结束

4\. **research\_agent**：研究智能体（使用真实大模型）

- 构建系统提示词和用户提示词

- 调用大模型获取研究结果

- 将结果保存到 `research_info`

5\. **writer\_agent**：写作智能体（使用真实大模型）

- 构建系统提示词和用户提示词

- 调用大模型根据研究信息生成报告

- 将结果保存到 `report`

6\. **route**：路由函数

- 根据 `current_agent` 字段决定下一步执行哪个节点

7\. **图结构**：

- START \-\> supervisor

- supervisor 根据条件边分配给 research 或 writer

- research 和 writer 执行完后都回到 supervisor

- supervisor 判断完成后走到 END

流程图：

```Plain Text
START -> supervisor
           │
    ┌──────┴──────┐
    ▼             ▼
research      writer
    │             │
    └──────┬──────┘
           ▼
       supervisor -> END
```

## **4\. 流水线模式：多智能体顺序处理**

流水线模式是最直观的多智能体架构，智能体按顺序处理任务。

场景：

- 用户输入一段原始文本

- 翻译智能体将其翻译成英文

- 摘要智能体生成摘要

- 情感分析智能体分析情感倾向

## **5\. 竞争模式：多个智能体竞争**

竞争模式中，多个智能体同时处理同一个任务，然后比较结果选择最优解。

场景：

- 用户提出一个问题

- 多个回答智能体给出不同的答案

- 评估智能体选择最佳答案

```Python
from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

class AgentState(TypedDict):
    theme: str                     # 用户输入的创作方向
    poems: List[dict]              # 每个智能体创作的诗词 [{'agent': 名字, 'poem': 诗词}, ...]
    scores: List[dict]             # 评分 [{'agent': 名字, 'score': 分数, 'comment': 评语}, ...]
    current_agent: str             # 当前应该执行哪个智能体

# 三个智能体各有不同的创作风格，形成竞争
AGENT_STYLES = {
    'poet_1': '豪放派诗人，风格大气磅礴、意境开阔',
    'poet_2': '婉约派诗人，风格细腻婉转、情感真挚',
    'poet_3': '现代派诗人，风格新颖独特、语言灵动'
}

def supervisor(state: AgentState):
    # 依次调度三个智能体创作，全部创作完成后交给评委
    if len(state['poems']) < 3:
        return {'current_agent': 'poet_%d' % (len(state['poems']) + 1)}
    elif not state['scores']:
        return {'current_agent': 'judge'}
    return {'current_agent': 'finish'}

def make_poet(agent_name: str):
    def poet_agent(state: AgentState):
        prompts = ChatPromptTemplate.from_messages([
            ('system', f'你是一位{AGENT_STYLES[agent_name]}。请根据用户的创作方向，创作一首诗词，只输出诗词本身。'),
            ('user', '创作方向: {theme}')
        ])

        chain = prompts | llm | StrOutputParser()
        poem = chain.invoke({'theme': state['theme']})
        print(f'\n【{agent_name}（{AGENT_STYLES[agent_name]}）】创作完成。')

        return {'poems': state['poems'] + [{'agent': agent_name, 'poem': poem}]}
    return poet_agent

def judge_agent(state: AgentState):
    # 评委智能体：对所有参赛作品逐一评分
    scores = []
    for poem in state['poems']:
        prompts = ChatPromptTemplate.from_messages([
            ('system', '你是一位资深诗词评委。请对给定诗词从意境、韵律、文采、切题四个方面综合评分（0-100的整数），并给出一句简短点评。严格按以下格式输出，不要输出其他内容：\n分数: xx\n点评: xxx'),
            ('user', '创作方向: {theme}\n诗词:\n{poem}')
        ])

        chain = prompts | llm | StrOutputParser()
        result = chain.invoke({'theme': state['theme'], 'poem': poem['poem']})

        # 解析评委输出
        score, comment = 0, ''
        for line in result.splitlines():
            line = line.strip()
            if line.startswith('分数'):
                digits = ''.join(ch for ch in line if ch.isdigit())
                score = int(digits) if digits else 0
            elif line.startswith('点评'):
                comment = line.split(':', 1)[-1].strip()

        scores.append({'agent': poem['agent'], 'score': score, 'comment': comment})

    return {'scores': scores}

def route(state: AgentState):
    if state['current_agent'].startswith('poet'):
        return state['current_agent']
    elif state['current_agent'] == 'judge':
        return 'judge'
    elif state['current_agent'] == 'finish':
        return END
    return 'supervisor'

builder = StateGraph(AgentState)
builder.add_node('supervisor', supervisor)
builder.add_node('poet_1', make_poet('poet_1'))
builder.add_node('poet_2', make_poet('poet_2'))
builder.add_node('poet_3', make_poet('poet_3'))
builder.add_node('judge', judge_agent)

builder.add_edge(START, 'supervisor')
builder.add_conditional_edges(
    'supervisor',
    route,
    {
        'poet_1': 'poet_1',
        'poet_2': 'poet_2',
        'poet_3': 'poet_3',
        'judge': 'judge',
        END: END
    }
)
builder.add_edge('poet_1', 'supervisor')
builder.add_edge('poet_2', 'supervisor')
builder.add_edge('poet_3', 'supervisor')
builder.add_edge('judge', 'supervisor')

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

while True:
    theme = input('\n请输入诗词创作方向（如：秋日思乡）: ')
    if not theme.strip():
        break

    result = graph.invoke({
        'theme': theme,
        'poems': [],
        'scores': [],
        'current_agent': ''
    })

    # 输出所有智能体的诗词与评分
    print('\n' + '=' * 50)
    print('所有智能体的参赛作品：')
    print('=' * 50)
    poem_by_agent = {p['agent']: p['poem'] for p in result['poems']}
    for s in result['scores']:
        print(f"\n【{s['agent']}】  评分: {s['score']}分  点评: {s['comment']}")
        print('-' * 50)
        print(poem_by_agent[s['agent']])

    # 输出评分最高的诗词
    winner = max(result['scores'], key=lambda s: s['score'])
    print('\n' + '=' * 50)
    print(f"评分最高的作品来自【{winner['agent']}】，得分 {winner['score']} 分：")
    print('=' * 50)
    print(poem_by_agent[winner['agent']])

```

## **6\. 辩论模式：智能体之间的辩论**

辩论模式中，多个智能体就某个问题展开辩论，最终达成共识或由裁判做出裁决。

场景：

- 用户提出一个有争议的问题

- 正方智能体支持某个观点

- 反方智能体反对该观点

- 裁判智能体总结辩论结果

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.3, streaming=True)

class DebateState(TypedDict):
    question: str
    debate_history: list
    round: int
    conclusion: str

def stream_print(chain, inputs, label=""):
    """流式调用 chain，逐字打印并返回完整结果。"""
    full = ""
    if label:
        print(f"\n【{label}】", end="", flush=True)
    for chunk in chain.stream(inputs):
        print(chunk, end="", flush=True)
        full += chunk
    print()
    return full

def pro_agent(state: DebateState):
    prompts = ChatPromptTemplate.from_messages([
        ('system', '你是一个毒舌辩论专家，针对可能有的反方观点，需要支持用户的观点。需要用极简的语句，一针见血。'),
        ('user', '用户的观点是{text},这次的辩论记录是: {history}')
    ])

    chain = prompts | llm | StrOutputParser()
    result = stream_print(chain, {
        'text': state['question'],
        'history': state['debate_history']
    }, '正方')
    return {
        'debate_history': state['debate_history'] + [{'side': '正方', 'argument': result}],
        'round': state['round']
    }

def con_agent(state: DebateState):
    new_count = state['round'] + 1
    prompts = ChatPromptTemplate.from_messages([
        ('system', '你是一个毒舌辩论专家，针对可能有的正方观点，需要反对用户的观点。需要用极简的语句，一针见血。'),
        ('user', '用户的观点是{text},这次的辩论记录是: {history}')
    ])

    chain = prompts | llm | StrOutputParser()
    result = stream_print(chain, {
        'text': state['question'],
        'history': state['debate_history']
    }, '反方')
    return {
        'debate_history': state['debate_history'] + [{'side': '反方', 'argument': result}],
        'round': new_count
    }

def judge_agent(state: DebateState):
    prompts = ChatPromptTemplate.from_messages([
        ('system', '你是一个老好人裁判，需要给出理由，说明双方观点都有道理，谁也不得罪，字数控制在200字左右'),
        ('user', '这次的辩论记录是: {debate_history}')
    ])

    chain = prompts | llm | StrOutputParser()
    conclusion = stream_print(chain, {
        'debate_history': state['debate_history']
    }, '裁判')
    return {'conclusion': conclusion}

def after_con_route(state: DebateState):
    if state['round'] <= 2:
        return 'pro'
    return 'judge'

builder = StateGraph(DebateState)
builder.add_node('pro', pro_agent)
builder.add_node('con', con_agent)
builder.add_node('judge', judge_agent)

builder.add_edge(START, 'pro')
builder.add_edge('pro', 'con')
builder.add_conditional_edges(
    'con',
    after_con_route,
    {
        'pro': 'pro',
        'judge': 'judge'
    }
)
builder.add_edge('judge', END)

graph = builder.compile()

result = graph.invoke({
    'question': '人工智能是否应该取代人类工作？',
    'debate_history': [],
    'round': 1,
    'conclusion': ''
})

print()
print('辩论总结：')
print(result['conclusion'])
```

### **代码解释**



1\. **DebateState**：定义辩论状态，包含：

- `question`：辩论的问题

- `debate_history`：辩论历史记录

- `round`：当前辩论轮次

- `conclusion`：最终结论

2\. **pro\_agent**：正方智能体

- 根据当前轮次给出支持观点

- 将论点添加到辩论历史

3\. **con\_agent**：反方智能体

- 根据当前轮次给出反对观点

- 将论点添加到辩论历史

4\. **judge\_agent**：裁判智能体

- 总结所有辩论内容

- 给出最终结论

5\. **after\_con\_route**：反方发言后的路由

- 如果还没到第2轮，回到正方开始下一轮

- 如果已经第2轮结束，进入裁判

6\. **图结构**：

- START \-\> pro \-\> con

- con 根据轮次决定回到 pro 还是进入 judge

- judge \-\> END

流程图：

```Plain Text
START -> pro -> con
                 │
        ┌────────┴────────┐
        ▼                 ▼
       pro → con      judge -> END
```

这种模式的优点：

- 可以全面分析问题的正反两面

- 通过辩论深入探讨复杂问题

- 最终结论更加客观全面

# 对话代理

## **1\. 什么是对话代理**

对话代理可以理解为一个能够和用户连续交流的 AI 程序。

普通的大模型调用通常是这样的：

```Plain Text
用户输入 -> 模型回答 -> 结束
```

而对话代理更像这样：

```Plain Text
用户第 1 轮输入 -> 模型回答
用户第 2 轮输入 -> 模型结合上一轮继续回答
用户第 3 轮输入 -> 模型继续理解上下文
```

它和一次性问答最大的区别是：**对话代理需要记住前面说过什么。**

例如用户先说：

```Plain Text
我叫小王。
```

下一轮再问：

```Plain Text
我叫什么？
```

如果代理能回答‘你叫小王’，说明它具备了基本的对话记忆。

## **2\. 为什么对话代理要用\`messages\`**

在 LangChain 和 LangGraph 中，对话通常不是用一个普通字符串保存，而是用消息列表 `messages` 保存。

一个简单的消息列表可能长这样：

```Python
[
    HumanMessage(content='我叫小王'),
    AIMessage(content='你好，小王！'),
    HumanMessage(content='我叫什么？')
]
```

这个列表的好处是：

- 能保存多轮对话历史

- 能区分每句话是谁说的

- 能直接传给聊天模型

- 很适合放进 LangGraph 的状态里

可以把 `messages` 理解成一份‘聊天记录本’。每轮对话都会往记录本里追加新内容。

## **3\. 三种常见消息类型**

对话代理中最常见的消息类型有三种：

|消息类型|含义|常见用途|
|---|---|---|
|`SystemMessage`|系统消息|规定 AI 的角色、规则、风格|
|`HumanMessage`|用户消息|保存用户输入|
|`AIMessage`|AI 消息|保存模型回复|

举个例子：

- `SystemMessage`：你是一个耐心的 Python 老师

- `HumanMessage`：请解释什么是列表

- `AIMessage`：列表是 Python 中用于存放多个元素的数据结构

这三类消息合在一起，就能表达一段完整对话。

## 4\.智能体维护对话的案例\(自带上下文记忆\)

```Python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.3, streaming=True)

class ChatState(TypedDict):
    message: Annotated[list, add_messages]

def simple_reply(state: ChatState):
    history = state["message"]

    prompts = ChatPromptTemplate.from_messages([
        ('system', '你是一个简洁的助手'),
        MessagesPlaceholder(variable_name="message"),
    ])

    chain = prompts | llm | StrOutputParser()
    result = chain.invoke({"message": history})
    return {"message": AIMessage(content=result)}

builder = StateGraph(ChatState)
builder.add_node('simple_reply', simple_reply)

builder.add_edge(START, 'simple_reply')
builder.add_edge('simple_reply', END)

graph = builder.compile()

messages = []
result = None
while True:
    text = input('>>> ')
    if text.strip() == 'exit':
        break
    # 把本轮用户输入追进历史，再传完整历史给图
    messages.append(HumanMessage(content=text))
    result = graph.invoke({'message': messages})
    print(result['message'][-1].content)
    # 用图返回的最新历史，更新累加列表
    messages = result['message']

if result is not None:
    print(result['message'])
print("bye~")
```

# **任务调度型智能体**

## **1\. 什么是任务调度型智能体**

任务调度型智能体可以理解为一个“项目经理型”智能体。用户只提出一个目标，它不会马上盲目执行，而是先思考：

1. 这个目标可以拆成哪些步骤？

2. 哪些步骤需要先做，哪些步骤可以后做？

3. 每一步应该交给哪个工具、函数或智能体执行？

4. 每一步完成后，如何把结果合并成最终答案？

例如用户说：

> 帮我分析一个产品的用户反馈，并给出改进建议。
> 
> 

任务调度型智能体可能会拆成：

1. 收集用户反馈

2. 对反馈进行分类

3. 提取高频问题

4. 生成改进建议

5. 输出结构化报告

这类智能体的核心不是“单次回答”，而是“组织多个步骤完成复杂任务”。

## **2\. 任务调度的基本流程**

一个典型任务调度型智能体通常包含 4 个环节：

|环节|作用|通俗理解|
|---|---|---|
|任务输入|接收用户目标|用户告诉智能体要做什么|
|任务分解|把大任务拆成小任务|项目经理制定待办清单|
|任务执行|按顺序执行每个小任务|不同同事完成各自工作|
|结果汇总|整合所有执行结果|写成最终报告交给用户|

在 LangGraph 中，我们可以把每个环节看成一个 **节点**，节点之间通过 **边** 连接起来，形成一个工作流。



