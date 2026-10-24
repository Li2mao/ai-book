# Day2

# 使用字典转换的情况

```MATLAB
什么时候需要这么做
核心判断标准只有一个：当前一个 Runnable 的输出格式，跟下一个 Runnable 期待的输入格式不匹配时。典型场景：

字符串 → 字典（本例）：前一段用 StrOutputParser() 输出字符串，后一段是带 {占位符} 的模板或 RunnableParallel，需要字典。
字段改名 / 挑字段：上游输出了 {'translation': ...}，下游模板却叫 {text}，用 lambda d: {'text': d['translation']} 做映射。
多字段合并 / 重组：把两个 key 拼成一句话再喂给下一个 prompt。
LLM 输出做轻量后处理：去掉多余空格、截断、简单清洗、转大写等，在进入下一环前先规整。
RunnableBranch 的条件判断预处理（demo05 里 is_chinese 就是先接收输入再判断）。
```

# 提示词模板

## **为什么需要 Prompt 模板**

- 直接拼接字符串来构造 Prompt 会带来几个问题：

\- **可读性差**：Python 字符串拼接难以看清最终给模型的提示长什么样

\- **容易出错**：变量多的时候容易遗漏、顺序错乱

\- **难以复用**：同样的提示结构无法直接复用到不同输入

\- **难以管理**：无法集中管理和版本化 Prompt

**Prompt 模板** 就是解决这些问题的方案。它允许你：

- 把 Prompt 写成一个带占位符的模板

- 运行时传入变量自动填充

- 复用同一个结构处理不同输入

- 把提示逻辑与业务逻辑分离

## **ChatPromptTemplate：聊天消息模板**

现代大模型通常采用对话格式，每条消息都有角色（role）。`ChatPromptTemplate` 就是为这种场景设计的。

常见的消息类型：

|消息类型|作用|
|---|---|
|system|设定模型的身份、能力和行为规则|
|user / human|用户的输入问题|
|assistant / ai|模型的回复，可用于 Few\-Shot 示例|

## **PromptTemplate：基础字符串模板**

`PromptTemplate` 是最基础的模板类，适用于构造单条文本提示。它只有一个 `template` 字符串和若干 \{变量\}。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv
#调用环境变量
_ = load_dotenv()
#调用模型
llm = ChatOpenAI(model='glm-5.2', temperature=0.7)
#提示词模板（可复用）
prompt = ChatPromptTemplate.from_messages([
    ('system','你是一位{role},擅长于用{style}风格回答问题。'),
    ('user','{question}')
])
#专家模板
pro_vars= {
    'role': '相关领域专家',
    'style': '硬核专业有学术深度',
    'question': '什么是质能方程'
}
#老师模板
air_vars= {
    'role': '小学老师',
    'style': '通俗易懂擅长举生活中的例子',
    'question': '什么是质能方程'
}
#查看提示词
print('=========组装后的提示词=========')
for msg in prompt.format_messages(**pro_vars):
    print(f'{msg.type}:{msg.content}')
#输出（可以切换专家和老师）
chain = prompt | llm 
print('=========大模型的输出========')
print(chain.invoke(pro_vars).content)
```

## 使用消息模板对象构建模板

除了元组，也可以使用 LangChain 提供的**消息模板类**来构建模板。这种方式语义更清晰，而且同样支持变量替换。

**注意区分两类对象**：

- `SystemMessage` / `HumanMessage`：普通消息对象，内容固定，不会解析 `{变量}`

- `SystemMessagePromptTemplate` / `HumanMessagePromptTemplate`：消息模板对象，内容中的 `{变量}` 会被替换

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#这里导入了系统角色以及人类角色模板
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2',temperature=0.7)

#这里应用了两个模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template('你是一个耐心的老师'),
    HumanMessagePromptTemplate.from_template('请解释{concept},要求{requirement}')
])

input_vars = {
    'concept': '神经网络',
    'requirement': '让小白也能听懂'
}

print('===========组装后的Prompt============')
for msg in prompt.format_messages(**input_vars):
    print(f'{msg.type}: {msg.content}')

chain = prompt | llm
print('\n===========大模型输出============')
result = chain.invoke(input_vars)
print(result.content)
```

## 模板变量的高级用法

### 默认值和partial

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2',temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一位{role}'),
    ('user', '请{action}:{content}')
])

#先固定 role 变量，得到一个新的模板
translator_prompt = prompt.partial(role='专业翻译')

input_vars = {
    'action': '把以下中文翻译成英文',
    'content': '今天天气很好'
}

print('======组装后的 Prompt ======')
for msg in translator_prompt.format_messages(**input_vars):
    print(f'{msg.type}: {msg.content}')

#构建链并调用
chain = translator_prompt | llm
print('\n======大模型的输出===========')
result = chain.invoke(input_vars)
print(result.content)
```

### **多轮对话模板**

模板中可以预设多轮对话历史，常用于构建 Few\-Shot 示例或保持上下文：

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2',temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个情绪分析助手，只输出正面或负面。'),
    ('human', '这个产品太差了'),
    ('ai', '负面'),
    ('human', '这次的体验非常愉快'),
    ('ai', '正面'),
    ('human', '{text}')
])

inputs = {'text': '东西还不错，会回购的'}

messages = prompt.format_messages(**inputs)
for msg in messages:
    print(f'{msg.type}: {msg.content}')

chain = prompt | llm
print(chain.invoke(inputs).content)
```

## **Few\-Shot Prompt：少样本提示**

Few\-Shot Prompt 是 Prompt Engineering 中最有效的技巧之一。它通过在输入前给出若干「示例\-答案」对，让模型学习期望的输出格式和风格。

LangChain 提供了 `FewShotPromptTemplate` 和 `FewShotChatMessagePromptTemplate` 两种主要方式。

```Python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 准备示例数据
examples = [
    {'input': '苹果', 'output': 'apple'},
    {'input': '香蕉', 'output': 'banana'},
    {'input': '人工智能', 'output': 'artificial intelligence'},
]

# 定义每个示例的格式
example_prompt = ChatPromptTemplate.from_messages([
    ('human', '{input}'),
    ('ai', '{output}')
])

# 构建 Few-Shot 提示模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# 组合成完整提示
final_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个中英文翻译助手，请参考示例回答问题。'),
    few_shot_prompt,
    ('human', '{input}')
])

messages = final_prompt.format_messages(input='机器学习')
for msg in messages:
    print(f'{msg.type}: {msg.content}')
```

### **明确输出格式（Output Formatting）**

如果你希望模型按特定格式输出，要在 Prompt 中明确说明。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

# 把 JSON 格式示例放在 system 消息中
# 注意：{ 和 } 是 LangChain 的变量占位符，如果要显示字面量大括号，必须写成 {{ 和 }}
system_template = '''你是一个信息提取助手。请只输出 JSON 格式，不要包含任何解释。

输出格式如下：
{{
  "names": [...],
  "locations": [...],
  "time": ""
}}'''

prompt = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '文本：{text}')
])

chain = prompt | llm | JsonOutputParser()

input_vars = {
    'text': '2024年5月1日，李明和王芳一起去了北京故宫参观。'
}

print('===== 组装后的 Prompt =====')
for msg in prompt.format_messages(**input_vars):
    print(f'{msg.type}: {msg.content}')

print('\n===== 大模型输出 =====')
result = chain.invoke(input_vars)
print(result)
```

### **分步骤思考（Chain\-of\-Thought）**

对于推理类问题，让模型「一步步思考」往往能显著提升准确率。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.1)

# 直接提问
prompt1 = ChatPromptTemplate.from_messages([
    ('user', '问题：一个农场有鸡和兔共35只，脚共94只。鸡和兔各几只？请直接给出答案。')
])

# 分步思考
prompt2 = ChatPromptTemplate.from_messages([
    ('user', '问题：一个农场有鸡和兔共35只，脚共94只。鸡和兔各几只？请一步步思考并给出答案。')
])

print('=== 直接提问 ===')
print((prompt1 | llm).invoke({}).content)
print('\n=== 分步思考 ===')
print((prompt2 | llm).invoke({}).content)
```

## **Prompt 模板与链结合**

下面是一个综合示例：使用模板构建一个「学习助手」，根据学生年级和知识点生成适合难度的解释。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一位{grade}水平的优秀教师。'),
    ('user', '请用适合{grade}学生理解的方式解释 {topic}，并举一个生活中的例子。')
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    'grade': '小学三年级',
    'topic': '浮力'
})
print(result)
```

## 客服小助手

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

# 客服助手系统提示词：明确定义正面/负面两种策略
system_prompt = """你是"暖心客服小助手"，请遵循以下规则：
1. 先判断用户的回应是正面还是负面：
   - 正面回应（如夸奖、感谢、好评）：用谦虚的客套话回应，例如"感谢您的认可，这是我们应尽的本分，我们会继续努力"。
   - 负面回应（如投诉、差评、不满）：先诚恳致歉，再给出对应的补偿措施（如发红包、退换货、补发赠品等），并说明具体操作流程。
2. 结合对话历史中的上下文回答问题，保持回复连贯、语气一致。"""

# 上下文记忆：保存最近 20 条消息（约 10 轮对话）
history: list = []

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name='history'),
    ('user', '{msg}')
])

chain = prompt | llm

print("客服小助手已上线，输入内容开始对话（输入 exit 退出）：")
while True:
    msg = input(">>> ")
    if msg.strip().lower() in ('exit', 'quit', '退出'):
        break

    result = chain.invoke({'history': history, 'msg': msg})
    print(result.content)

    # 更新记忆，只保留最近 20 条
    history.extend([HumanMessage(content=msg), AIMessage(content=result.content)])
    history = history[-20:]
```

# maxkb使用

- 这是一个图形化的AI模型制造

- 使用docker compose使用下列指令

```YAML
services:
  maxkb:
    image: 1panel/maxkb:v2.10.4-lts
    container_name: maxkb_v2.10.4
    restart: unless-stopped
    ports:
      - 8080:8080
    volumes:
      - ./data:/opt/maxkb
```

# 输出解析器

## **为什么需要输出解析器**

大模型返回的通常是**自由文本**，但很多时候我们需要：

- 提取结构化的 JSON 数据

- 把文本转换成 Python 列表、字典或对象

- 验证输出是否符合预期的数据格式

- 过滤掉模型多余的解释，只保留关键信息

**输出解析器**就是链中负责把模型输出转换为结构化数据的组件。

在 LCEL 中，链的常见结构是：

```Plain Text
chain = prompt | llm | output_parser
```

其中 `output_parser` 接收 `AIMessage` 对象，输出我们需要的 Python 数据结构。

## **StrOutputParser：最简单的字符串解析器**

`StrOutputParser` 会把模型返回的 `AIMessage` 对象直接转换成纯字符串。

它适合的场景：

- 只需要文本回答

- 不想每次手动写 `.content`

- 作为链的最后一步统一输出类型

```Python
# 导入所需模块
from langchain_openai import ChatOpenAI                # Langchain封装的OpenAI兼容客户端，可调用智谱GLM等兼容接口的模型
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示词模板，用于构建system、user多轮消息结构
from langchain_core.output_parsers import StrOutputParser # 输出解析器，将模型返回消息对象转为纯字符串
from dotenv import load_dotenv                         # python-dotenv库，用于读取.env环境变量文件

_ = load_dotenv()  # 加载项目目录下.env文件中的环境变量（存放API_KEY、BASE_URL，避免密钥硬编码）
# _代表忽略返回值，仅执行加载动作

# 初始化大模型实例，调用智谱GLM‑4.5
llm = ChatOpenAI(model='glm-4.5', temperature=0.7)
# model='glm-4.5' 指定模型名称；temperature=0.7控制随机性，数值越高回答越富有创意

# 创建聊天提示模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个性格像Reze的助手'), # system系统提示，设定AI角色人设
    ('user', '{text}')                     # user用户消息，{text}为占位符，运行时填充真实用户输入
])

# LCEL链式管道 | 串联各个组件：提示模板→大模型→字符串解析器
chain = prompt | llm | StrOutputParser()
# 执行顺序：填充模板生成完整prompt → 请求大模型得到返回对象 → 提取文本字符串

# 调用链路，传入用户输入，打印AI回复
print(chain.invoke({'text': '今天天气不错'}))
# invoke启动整条链，字典键text匹配模板占位符，值为用户提问
```

- 自定义输入内容修改方向

```MATLAB
# ==========仅修改此处，实现自定义输入，前面代码完全原样保留==========
user_input = input("请输入提问内容：")   #控制台手动输入内容
print(chain.invoke({'text': user_input}))
```

## **JsonOutputParser：解析 JSON 输出**

`JsonOutputParser` 会自动把模型输出的 JSON 字符串解析成 Python 字典。

使用要点：

1. 在 Prompt 中明确要求模型输出 JSON

2. JSON 中的大括号 `{` 在 Prompt 模板中需要写成 `{{` 进行转义

3. 模型输出必须能被 `json.loads()` 解析

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from dotenv import load_dotenv
import json

_ = load_dotenv()

llm = ChatOpenAI(model='glm-4.5', temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个信息提取助手。将文字的关键信息提取， 请只输出JSON格式，不包含任何解释'),
    ('user', '{text}')
])

chain = prompt | llm | JsonOutputParser()
result = chain.invoke({'text': '2024年5月1日,李明和王芳一起去了北京故宫参观。'})
print(json.dumps(result,indent=2,ensure_ascii=False))
```

## **PydanticOutputParser：解析为 Pydantic 对象**

果你需要更严格的类型校验和结构化数据，可以使用 `PydanticOutputParser`。

Pydantic 是 Python 中非常流行的数据验证库。通过定义数据模型，可以：

- 明确每个字段的类型

- 自动验证数据格式

- 把模型输出转换成可操作的 Python 对象

- 这里没有的模块，自己在终端里面下载`pip install 模块名`

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-4.5', temperature=0.7)

#第一步： 定义Pydantic数据类型
class PersonInfo(BaseModel):
    name: str = Field(description='人物姓名')
    age: int = Field(description='人物年龄')
    city: str = Field(description='所在城市')
    hobbies: list[str] = Field(description='兴趣爱好列表')

#第二步： 创建解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)

#第三步： 在Prompt中嵌入格式说明
prompt = ChatPromptTemplate.from_messages([
    ('system', '''你是一个信息提取助手。请从文本中提取任务信息，按指定格式输出。\n{format_instructions}'''),
    ('user', '文本:{text}')
])

#format_instructions 会自动生成JSON Schema说明
prompt_with_format = prompt.partial(format_instructions=parser.get_format_instructions())

#构建链
chain = prompt_with_format | llm | parser 

result = chain.invoke({
    'text': '张三今年25岁，住在杭州，喜欢打篮球和编程。'
})

print('返回类型：', type(result))
print('返回对象: ', result)
print('姓名： ', result.name)
print('年龄：', result.age)
print('爱好： ', result.hobbies)
```

## **CommaSeparatedListOutputParser：解析逗号分隔列表**

当你需要模型返回一个列表时，可以使用 `CommaSeparatedListOutputParser`。

它适合的场景：

- 提取关键词

- 生成待办事项

- 输出多个选项

```Python
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

#创建解析器
parser = CommaSeparatedListOutputParser()

#获取格式说明
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个关键词提取助手。{format_instructions}'),
    ('user', '请从以下文本中提取3-5个关键词: {text}')
])

#预填充格式说明
prompt_with_format = prompt.partial(format_instructions=format_instructions)

chain = prompt_with_format | llm | parser 

result = chain.invoke({
    'text': '人工智能、机器学习和深度学习正在改变各个行业的运作方式。'
})

print('返回类型: ', type(result))
print('关键词列表: ', result)
print('第一个关键词: ', result[0])
```

## **处理格式异常：手动 try\-except \+ LLM 修复**

大模型有时不会严格按 JSON 格式输出，可能包含多余文字、缺失引号或注释。

`手动 try-except + LLM 修复` 可以包装另一个解析器，当第一次解析失败时，自动调用 LLM 修复输出格式。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import JsonOutputParser 
import json 
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

#定义一个JSON解析器
parser = JsonOutputParser()

#提示模型输出JSON，但不强制约束格式
prompt =  ChatPromptTemplate.from_messages([
    ('system', '请把用户信息转换成 JSON 格式输出'),
    ('user', '内容：{text}')
])

#先获取模型的原始文本输出
raw_chain = prompt | llm 
raw_output = raw_chain.invoke({'text': '姓名：李四，年龄：30，城市：上海'})

print('===========原始模型输出=============')
print(raw_output.content)

#尝试解析JSON
try:
    result = parser.invoke(raw_output)
    print('\n===========直接解析成功==========')
    print(result)
except Exception as e:
    print('\n===========解析失败，尝试用LLM修复=======')
    print('错误信息: ', e)

    #构造修复Prompt, 让模型把错误输出修正为合法JSON
    fix_prompt = ChatPromptTemplate.from_messages([
        ('system', '你是一个JSON修复专家。请把用户提供的内容修正为合法 JSON ，只输出JSON字符串，不需要任何解释'),
        ('user', '原始内容： \n{raw_output}\n\n错误信息:{error}')
    ])

    fix_chain = fix_prompt | llm 
    fixed_output = fix_chain.invoke({
        'raw_output': raw_output.content,
        'error': str(e)
    })

    #手动用json.loads解析修复后的内容
    result = json.loads(fixed_output.content)
    print('\n========修复后的结果=======')
    print(result)
```

## **自定义输出解析器**

如果内置解析器不能满足需求，你可以继承 `BaseOutputParser` 自己实现。

下面是一个简单的自定义解析器：把模型输出按指定分隔符切分成列表。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

# 自定义解析器：按换行符切分，并去除空行和序号
class NumberedListParser(BaseOutputParser[list[str]]):
    """把模型输出的编号列表解析成字符串列表。"""

    def parse(self, text: str) -> list[str]:
        # 按行切分
        lines = text.strip().split('\n')
        # 去除空行，并去掉每行开头的 1. 2. 3. 等序号
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去掉行首的数字和点，例如 "1. "
            if '. ' in line[:4]:
                line = line.split('. ', 1)[1]
            items.append(line)
        return items

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个清单生成助手。请输出编号列表，每行一个项目。'),
    ('user', '请列出学习{topic}的5个步骤')
])

chain = prompt | llm | NumberedListParser()

result = chain.invoke({'topic': 'Python 编程'})

print('返回类型：', type(result))
print('返回内容：', result)
print('步骤数量：', len(result))
```

## **示例：结构化数据提取**

下面是一个综合示例：从一段商品评价中同时提取评分、优点、缺点和关键词。

```Python
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 
from dotenv import load_dotenv

_  = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

# class PersonInfo(BaseModel):
#     name: str = Field(description='人物姓名')
#     age: int = Field(description='人物年龄')
#     city: str = Field(description='所在城市')
#     hobbies: list[str] = Field(description='兴趣爱好列表')

class Review(BaseModel):
    sentiment: str = Field(description='整体情感，正面/负面/中性')
    pros: list[str] = Field(description='优点列表')
    cons: list[str] = Field(description='缺点列表')
    keywords: list[str] = Field(json_schema_extra={'descriptions': '关键词列表'})
    
parser = PydanticOutputParser(pydantic_object=Review)

prompt  = ChatPromptTemplate.from_messages([
    ('system', '''你是一个商品评价分析助手。请从评价中提取结构化信息\n{format_instructions}\n只输出JSON，不要其他内容'''),
    ('user', '商品评价: {review}')
])

prompt_with_format = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt_with_format | llm | parser

review_text = '''这款手机外观很漂亮，拍照效果也不错。
但是电池续航一般，充电速度有点慢。
总体来说性价比还可以。'''

result = chain.invoke({'review': review_text})

print('情感: ', result.sentiment)
print('优点: ', result.pros)
print('缺点: ', result.cons)
print('关键词: ', result.keywords)
```

# 链式组合

## **什么是 LCEL 链式组合**

LCEL 是 LangChain 推荐的新一代链式组合语法。它用管道符 `|` 把各个组件串联起来，就像 Linux 命令行一样直观。

```Plain Text
chain = component_a | component_b | component_c
```

数据流：

```Plain Text
输入 -> component_a -> 中间结果 -> component_b -> 中间结果 -> component_c -> 输出
```

LCEL 中的每个组件都实现了 `Runnable` 接口，它们都有统一的方法签名。

## **使用 RunnableLambda 转换数据**

在组合链时，前一个组件的输出格式可能和后一个组件的输入格式不匹配。

例如：前一个链输出字符串，后一个链需要字典。这时可以用 `RunnableLambda` 或普通 lambda 函数做转换。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

#第一步：把中文翻译成英文
translate_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个翻译助手。'),
    ('user', '把以下中文翻译成英文: {text}')
])
translate_chain = translate_prompt | llm | StrOutputParser()

#第二步：把英文总结成一句话
summarize_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个总结助手。'),
    ('user', '请用一句话总结以下内容：{text}')
])
summarize_chain = summarize_prompt | llm | StrOutputParser()

#翻译链输出字符串，但总结链需要字典{text:...}
#使用RunnableLambda做格式转换
def to_dict(text):
    return{'text': text}

#组合链：翻译 -> 转字典 -> 总结
full_chain = translate_chain | RunnableLambda(to_dict) | summarize_chain

result = full_chain.invoke({'text': '大语言模型开发效率比较高，而且比较聪明，而且成本比较低，大家都在学习怎么使用，正在改变软件的开发方式。'})
print('第一步结果：', translate_chain.invoke({'text': '大语言模型开发效率比较高，而且比较聪明，而且成本比较低，大家都在学习怎么使用，正在改变软件的开发方式。'}))
print('最终结果：', result)
```

## **直接传递多个变量：RunnableParallel**

`RunnableParallel` 允许你同时运行多个子链，并把多个结果合并成一个字典。

它适合的场景：

- 同时让模型做翻译和总结

- 同时提取不同类型的信息

- 一个输入需要生成多个输出

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

#定义多个子链
translate_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个翻译助手。'),
    ('user', '把以下中文翻译成英文:{text}')
])
translate_chain = translate_prompt | llm | StrOutputParser()

summary_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是总结助手。'),
    ('user', '用一句话总结：{text}')
])
summary_chain = summary_prompt | llm | StrOutputParser()

keywords_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是关键词提取助手'),
    ('user', '提取3个关键词，用逗号分隔：{text}')
])
keywords_chain = keywords_prompt | llm | StrOutputParser()

#并行执行三个子链，输入都是{'text': '....'}
parallel_chain = RunnableParallel(
    translation=translate_chain,
    summary=summary_chain,
    keywords=keywords_chain
)

result = parallel_chain.invoke({'text': '人工智能正在改变教育、医疗和交通行业'})

print('返回类型：', type(result))
print('翻译结果：', result['translation'])
print('总结结果：', result['summary'])
print('关键词结果：', result['keywords'])
```

## **组合并行与串行**

`RunnableParallel` 的输出是一个字典，可以继续传给下一个组件使用。

例如：先并行做翻译和总结，再把结果合并成一个报告。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel 
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

#并行生成翻译和总结
parallel_chain = RunnableParallel(
    translation=(
        ChatPromptTemplate.from_messages([
            ('system', '翻译助手'),
            ('user', '把中文翻译成英文：{text}')
        ]) | llm | StrOutputParser()
    ),
    summary=(
        ChatPromptTemplate.from_messages([
            ('system', '总结助手'),
            ('user', '用一句话总结: {text}')
        ]) | llm | StrOutputParser()
    )
)

#把并行结果合并成最终报告
report_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个报告生成助手'),
    ('user', '''请根据以下翻译和总结生成一份简短报告：\n\n翻译：{translation}\n总结：{summary}\n\n请输出包含[原文概要]和[英文翻译]两部分的报告''')
])

full_chain = parallel_chain | report_prompt | llm | StrOutputParser()

result = full_chain.invoke({'text': '大语言模型开发效率比较高，而且比较聪明，而且成本比较低，大家都在学习怎么使用，正在改变软件的开发方式。'})
print(result)
```

## **条件分支：RunnableBranch**

`RunnableBranch` 可以根据条件选择不同的子链执行。

例如：根据输入语言类型选择不同的处理链。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

#定义两个处理链
chinese_chain = (
    ChatPromptTemplate.from_messages([
        ('system', '你是中文助手'),
        ('user', '请用中文解释：{text}')
    ]) | llm | StrOutputParser()
)

english_chain = (
    ChatPromptTemplate.from_messages([
        ('system', '你是英文助手'),
        ('user', '请用英文解释：{text}')
    ]) | llm | StrOutputParser()
)

#判断输入文本是否包含中文字符
def is_chinese(input_dict):
    text = input_dict['text']
    return any('\u4e00' <= char <= '\u9fff' for char in text)

#构建分支链
branch_chain = RunnableBranch(
    (is_chinese, chinese_chain),
    english_chain   #默认分支
)

print('中文输入结果：')
print(branch_chain.invoke({'text': '神经网络'})[:250])

print('\n英文输入结果：')
print(branch_chain.invoke({'text': 'neural network'})[:250])
```

## **查看链的结构**

复杂的链可以通过 `get_graph().print_ascii()` 方法查看结构。它会用 ASCII 字符打印出链中的节点和连接关系，非常直观。

> 需要安装依赖：`pip install grandalf`
> 
> 

\> 安装后请**重启 Jupyter 内核**再运行本单元格。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

chain = (
    ChatPromptTemplate.from_messages([
        ('user', '把以下内容翻译成英文：{text}')
    ])
    | llm
    | StrOutputParser()
    | (lambda x: {'text': x})
    | ChatPromptTemplate.from_messages([
        ('user', '用一句话总结: {text}')
    ])
    | llm 
    | StrOutputParser()
)

#打印链的ASCII结构图
chain.get_graph().print_ascii()
```

## **完整示例：智能客服工单处理**

下面是一个综合示例：用户输入一条投诉或咨询，系统同时完成情绪分析、分类、生成回复建议。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

#并行分析：情绪+分类+摘要
#注意：reply_prompt中还需要原始输入text，所以这里用lambda把text一起传下去
analysis_chain = RunnableParallel(
    text=lambda x: x['text'],
    sentiment=(
        ChatPromptTemplate.from_messages([
            ('system', '情绪分析助手，只输出：正面/负面/中性'),
            ('user', '{text}')
        ]) | llm | StrOutputParser()
    ),
    category=(
        ChatPromptTemplate.from_messages([
            ('system', '分类助手，只输出类别：售后/产品/物流/其他'),
            ('user', '{text}')
        ]) | llm | StrOutputParser()
    ),
    summary=(
        ChatPromptTemplate.from_messages([
            ('system', '摘要助手，用一句话概括'),
            ('user', '{text}')
        ]) | llm | StrOutputParser()
    )
)

#根据分析结果生成回复建议
reply_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是客服主管，请根据分析结果生成回复建议。'),
    ('user', '''用户反馈：{text}
    分析结果： 
    - 情绪：{text}
    - 类别：{category}
    - 摘要： {summary}

    请生成一段礼貌、专业的回复建议。
    ''')
])

full_chain = analysis_chain | reply_prompt | llm | StrOutputParser()

result = full_chain.invoke({
    'text': '你们的发货速度太慢了，说好三天到，结果等了一周，非常失望！'
})
print(result)
```

## 作业练习

- 要求：

```MATLAB
#开发一个写作的小助手，根据标题产生文章结构，根据文章结构产生文章大纲以及每章节的100字描述
#将文章结构，文章大纲，每节描述，组合到新的提示词中，然后并发写文章
#最终组装，需要有大纲和正文
#扩展需求，在组装之后，对全文进行评价打分，如果分数低于8/10、，那么将全文和修改建议一起带入下一轮loop
```

- 完成代码

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

#标题->文章结构
jiegou_chain = (ChatPromptTemplate.from_messages([
    ('system', '你是一个写作的小助手。'),
    ('user', '根据标题：{text}，产生文章结构。')
]) | llm | StrOutputParser()
)

#文章结构->文章大纲
dagang_chain = (
    RunnableLambda(lambda s: {'text': s})
    | ChatPromptTemplate.from_messages([
      ('system', '你是一个写作的小助手。'),
     ('user', '根据文章结构：{text}，产生文章大纲以及每章节100字的描述。')
]) | llm | StrOutputParser()
)

#大纲->每章节约100字描述
miaoshu_chain = (
    RunnableLambda(lambda s: {'text': s})
    | ChatPromptTemplate.from_messages([
    ('system', '你是一个写作的小助手。'),
    ('user', '根据文章大纲：{text}，生成每个章节约100字的描述。')
]) | llm | StrOutputParser()
)
#用lambda按顺序把材料备齐
def make_plan(text):
    jiegou = jiegou_chain.invoke({'text': text})
    dagang = dagang_chain.invoke(jiegou)
    miaoshu = miaoshu_chain.invoke(dagang)
    return{'text': text, 'jiegou': jiegou, 'dagang': dagang, 'miaoshu': miaoshu}

# 写作链：模板里加了 {feedback}，首轮为空字符串不影响
write_chain = (
    ChatPromptTemplate.from_messages([
        ('system', '你是一个写作的小助手。'),
        ('user', '''请根据以下材料写一篇完整文章正文：
标题：{text}
结构：{jiegou}
大纲与章节描述：{dagang}
{feedback}''')
    ]) | llm | StrOutputParser()
)

# 组装：直接用代码拼接大纲+正文，不再让 LLM 重写一遍
def assemble(dagang, zhengwen):
    return f'【大纲】\n{dagang}\n\n【正文】\n{zhengwen}'

# 评价打分链：输出结构化 JSON {"score": ..., "suggestions": ...}
# 打分用 temperature=0，结果更稳定
critic_chain = (
    ChatPromptTemplate.from_messages([
        ('system', '你是资深编辑，请从内容质量、逻辑结构、语言表达等角度打分（满分10分），并给出具体修改建议。'),
        ('user', '文章：{article}\n请只输出JSON格式：{{"score": 0到10的整数, "suggestions": "具体的修改建议"}}')
    ]) | ChatOpenAI(model='glm-5.2', temperature=0)
    | JsonOutputParser()
)

# 流式打印并累积返回完整文本，方便每轮观看
def stream_text(chain, **inputs):
    text = ''
    for chunk in chain.stream(inputs):
        print(chunk, end='', flush=True)
        text += chunk
    print()
    return text

MAX_ROUNDS = 3
title = '狼王梦'

# 结构与大纲只规划一次，循环里只"重写正文"
plan = make_plan(title)
feedback = ''   # 首轮没有修改建议

for round_no in range(1, MAX_ROUNDS + 1):
    print(f'\n===== 第 {round_no} 轮写作 =====')
    zhengwen = stream_text(
        write_chain,
        text=plan['text'], jiegou=plan['jiegou'],
        dagang=plan['dagang'], miaoshu=plan['miaoshu'],
        feedback=feedback,
    )

    article = assemble(plan['dagang'], zhengwen)   # ← 245行的"组装之后"

    print('\n===== 编辑评分中 =====')
    verdict = critic_chain.invoke({'article': article})
    score = float(verdict['score'])
    print(f'当前得分：{score}/10')

    if score >= 8:
        print('评分达标，结束循环。')
        break

    if round_no == MAX_ROUNDS:
        print(f'已达最大轮数 {MAX_ROUNDS}，输出最后一次结果。')
        break

    # ← 245行的"全文和修改建议一起带入下一轮"
    feedback = (f'这是上一版全文：\n{article}\n\n'
                f'编辑的修改建议：{verdict["suggestions"]}\n'
                '请结合修改建议重新写一版正文，直接输出正文，不要解释。')
```

# tools的定义

## **为什么需要工具**

大语言模型本身只能处理文本，它的知识截止于训练数据，也无法直接访问外部世界。

**工具（Tools）** 让模型能够：

- 执行数学计算（模型不擅长精确计算）

- 查询数据库、API、搜索引擎

- 读写文件、操作数据库

- 获取实时信息（天气、股价、新闻）

智能体（Agent）的核心工作就是：**根据用户问题选择合适的工具，调用工具，再根据结果回答问题**。

## **使用 @tool 装饰器定义第一个工具**

LangChain 提供了 `@tool` 装饰器，可以把一个普通 Python 函数变成智能体可调用的工具。

```Python
from langchain.tools import tool

# 使用 @tool 装饰器定义一个加法工具
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

# 查看工具信息
print('工具名称：', add.name)
print('工具描述：', add.description)
print('参数Schema：', add.args)
```

### **代码解释**

- `@tool`：把 `add` 函数注册为 LangChain 工具

- `a: int, b: int`：参数类型会被自动解析为工具的输入参数格式

- 函数文档字符串 `"""..."""`：会被自动提取为工具描述（description）

- `add.name`：工具名，默认使用函数名

- `add.description`：工具描述，来自函数的 docstring

- `add.args`：参数的 JSON Schema，模型根据它决定如何传参

## **手动调用工具**

工具定义好后，可以通过 `invoke` 方法或 `run` 方法调用。

注意：\`@tool\` 装饰后的对象是一个 \`StructuredTool\` 对象，**不能直接像普通函数那样加括号调用**（如 \`multiply\(3, 4\)\`），需要传入字典参数。

```Python
from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b

# 通过 invoke 调用，传入字典参数
print('invoke 调用：', multiply.invoke({'a': 3, 'b': 4}))

# 通过 run 调用（LangChain 工具的传统调用方式）
print('run 调用：', multiply.run({'a': 5, 'b': 6}))
```

## **自定义工具名称和描述**

默认情况下，工具名就是函数名，描述来自 docstring。你也可以通过 `@tool` 参数自定义。

```Python
from langchain.tools import tool

@tool('weather-query', description='查询指定城市的当前天气情况')
def get_weather(city: str) -> str:
    """
    查询天气。
    参数：
        city: 城市名称，例如 "北京"、"上海"
    """
    # 这里用模拟数据，实际可接入天气 API
    weather_data = {
        '北京': '晴，25°C',
        '上海': '多云，28°C',
        '广州': '小雨，30°C'
    }
    return weather_data.get(city, f'未找到 {city} 的天气信息')

print('工具名称：', get_weather.name)
print('工具描述：', get_weather.description)
print('调用结果：', get_weather.invoke({'city': '北京'}))
```

## **定义带多个参数的工具**

工具可以有多个参数，LangChain 会自动根据类型注解生成参数 Schema。

```Python
from langchain.tools import tool

@tool
def calculate_bmi(height: float, weight: float) -> str:
    """
    根据身高和体重计算 BMI 指数，并返回健康建议。
    参数：
        height: 身高，单位米
        weight: 体重，单位千克
    """
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        suggestion = '体重过轻，建议加强营养'
    elif bmi < 24:
        suggestion = '体重正常，请保持'
    elif bmi < 28:
        suggestion = '超重，建议适当运动'
    else:
        suggestion = '肥胖，建议咨询医生'
    return f'BMI: {bmi:.2f}，{suggestion}'

print(calculate_bmi.invoke({'height': 1.75, 'weight': 70}))
print('\n参数 Schema：')
print(calculate_bmi.args)
```

## **定义模拟 API 查询工具**

实际应用中，工具通常会调用外部 API。下面我们用一个模拟的「产品库存查询」工具来演示。

```Python
from langchain.tools import tool

@tool
def query_product_stock(product_name: str) -> str:
    """
    查询指定商品的库存数量。
    参数：
        product_name: 商品名称
    """
    # 模拟数据库查询
    stock_db = {
        '手机': 150,
        '笔记本电脑': 45,
        '耳机': 200,
        '充电宝': 80
    }
    stock = stock_db.get(product_name)
    if stock is None:
        return f'未找到商品 "{product_name}" 的库存信息'
    return f'商品 "{product_name}" 当前库存：{stock} 件'

# 测试工具
print(query_product_stock.invoke({'product_name': '手机'}))
print(query_product_stock.invoke({'product_name': '相机'}))
```

## **定义文件处理工具**

工具也可以读写文件。下面是读取文本文件和写入文本文件的工具示例。

```Python
from langchain.tools import tool

@tool
def read_text_file(file_path: str) -> str:
    """
    读取指定文本文件的内容。
    参数：
        file_path: 文件路径
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f'文件不存在：{file_path}'
    except Exception as e:
        return f'读取失败：{str(e)}'

@tool
def write_text_file(file_path: str, content: str) -> str:
    """
    将内容写入指定文本文件。
    参数：
        file_path: 文件路径
        content: 要写入的内容
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f'已成功写入文件：{file_path}'
    except Exception as e:
        return f'写入失败：{str(e)}'

# 测试写入和读取
write_text_file.invoke({'file_path': 'test_note.txt', 'content': '这是测试内容'})
print(read_text_file.invoke({'file_path': 'test_note.txt'}))
```

## **工具列表与工具选择**

在构建智能体时，通常会把多个工具放在一个列表里，交给模型选择使用。

```Python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b

@tool
def power(base: int, exponent: int) -> int:
    """计算 base 的 exponent 次方。"""
    return base ** exponent

# 把工具放入列表
tools = [add, multiply, power]

# 查看工具列表信息
for t in tools:
    print(f'名称：{t.name}，描述：{t.description}')
```

# tools的调用

## **工具调用的基本流程**

工具调用不是由模型直接执行代码，而是模型「决定」要调用哪个工具、传什么参数，然后由程序代为执行。

完整流程如下：

```Plain Text
用户提问 -> 模型分析 -> 输出工具调用请求 -> 程序执行工具 -> 结果返回模型 -> 模型生成最终回答
```

模型输出的是结构化的调用请求，包含：

- 工具名称（name）

- 工具参数（arguments）

程序根据请求执行对应函数，再把结果返回给模型。

## **准备工具集**

首先定义几个简单的工具，用于后续演示。

```Python
from langchain.tools import tool

@tool
def add(a: float, b: float) -> float:
    """计算两个数的和。"""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """计算两个数的乘积。"""
    return a * b

@tool
def query_weather(city: str) -> str:
    """
    查询指定城市的天气。
    参数：
        city: 城市名称，如北京、上海
    """
    weather_db = {'北京': '晴 25°C', '上海': '多云 28°C', '广州': '小雨 30°C'}
    return weather_db.get(city, f'未找到 {city} 的天气信息')

tools = [add, multiply, query_weather]

for t in tools:
    print(f'工具：{t.name}，描述：{t.description}')
```

## **手动执行工具调用**

在使用智能体之前，先手动走一遍工具调用流程，理解底层机制。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from langchain.tools import tool

_ = load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b

@tool
def power(base: int, exponent: int) -> int:
    """计算 base 的 exponent 次方。"""
    return base ** exponent

# 把工具放入列表
tools = [add, multiply, power]

# 查看工具列表信息
for t in tools:
    print(f'名称：{t.name}，描述：{t.description}')

llm = ChatOpenAI(model='glm-5.2', temperature=0.1)

# 把工具绑定到模型上
llm_with_tools = llm.bind_tools(tools)

# 用户提问
question = '北京和上海的气温相差多少度？'
messages = [HumanMessage(content=question)]

# 模型输出，可能包含工具调用请求
response = llm_with_tools.invoke(messages)

print('模型回复类型：', type(response))
print('工具调用请求：', response.tool_calls)
print('文本内容：', response.content)
```

## **手动解析并执行工具调用**

模型只负责「决定」调用什么工具，真正的执行需要程序完成。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from dotenv import load_dotenv

_  = load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b

@tool
def power(base: int, exponent: int) -> int:
    """计算 base 的 exponent 次方。"""
    return base ** exponent

@tool
def query_weather(city: str) -> str:
    """
    查询指定城市的天气。
    参数：
        city: 城市名称，如北京、上海
    """
    weather_db = {'北京': '晴 25°C', '上海': '多云 28°C', '广州': '小雨 30°C'}
    return weather_db.get(city, f'未找到 {city} 的天气信息')
#把工具放入列表
tools = [add, multiply, query_weather]

llm = ChatOpenAI(model='glm-5.2', temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

# 使用一个只需一次工具调用的简单问题
question = '北京今天的天气怎么样？'
messages = [HumanMessage(content=question)]

# 第一步：获取模型的工具调用请求
response = llm_with_tools.invoke(messages)
messages.append(response)

print('第一次模型输出：')
print('tool_calls:', response.tool_calls)
print('content:', response.content)

# 第二步：执行工具调用
for tool_call in response.tool_calls:
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    
    # 找到对应的工具并执行
    selected_tool = {t.name: t for t in tools}[tool_name]
    tool_result = selected_tool.invoke(tool_args)
    
    # 把工具执行结果加入对话历史
    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call['id']))
    
    print(f'\n执行工具：{tool_name}，参数：{tool_args}，结果：{tool_result}')

# 第三步：把工具结果返回模型，生成最终回答
final_response = llm_with_tools.invoke(messages)
print('\n最终回答：', final_response.content)
```

## **使用 create\_agent 构建智能体**

LangChain 1\.x 推荐使用 `create_agent` 构建工具调用智能体。它会自动处理「模型决定调用工具 \-\> 执行工具 \-\> 返回结果 \-\> 再次调用模型」的循环，并返回一个可执行的状态图。

`create_agent` 的核心参数：

|参数|说明|
|---|---|
|`model`|语言模型实例|
|`tools`|工具列表|
|`system_prompt`|系统提示词|
|`checkpointer`|可选，用于保存对话历史|

```Python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool 
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.1)

@tool 
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

@tool 
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b 

@tool 
def power(base: int, exponent: int) -> int:
    """计算base的exponse次方。"""
    return base ** exponent 

@tool 
def query_weather(city: str) -> str:
    """
    查询指定城市的天气。
    参数：
        city: 城市名称，如北京，上海
    """
    weather_data = {'北京': '晴 25℃', '上海': '多云 28℃', '广州': '小雨 30℃'}
    return weather_data.get(city, f'未找到{city}的天气信息')

#把工具放入列表
tools = [add, multiply, query_weather]

#创建工具调用智能体
#create_agent是Langchain 1.x 推荐的智能体构建方式，返回一个可执行的状态图
agent = create_agent(
    model = llm,
    tools=tools,
    system_prompt='你是一个helpful的数学和天气助手，可以使用工具帮助用户'
)

#运行智能体
result = agent.invoke({'messages': [('user', '上海天气怎么样？')]})
print('\n最终输出：', result['messages'][-1].content)
```

## **多轮对话：带记忆的智能体**

如果希望智能体能记住上下文，可以在 `create_agent` 中传入 `checkpointer`。这里使用 `MemorySaver` 作为内存型的 checkpointer。

```Python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool 
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

@tool 
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

@tool 
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b 

@tool 
def power(base: int, exponent: int) -> int:
    """计算base的exponse次方。"""
    return base ** exponent 

@tool 
def query_weather(city: str) -> str:
    """
    查询指定城市的天气。
    参数：
        city: 城市名称，如北京，上海
    """
    weather_data = {'北京': '晴 25℃', '上海': '多云 28℃', '广州': '小雨 30℃'}
    return weather_data.get(city, f'未找到{city}的天气信息')

#把工具放入列表
tools = [add, multiply, query_weather]

# 创建带记忆的智能体
# checkpointer 会自动保存和加载对话历史
memory = MemorySaver()
agent_with_history = create_agent(
    model=llm,
    tools=tools,
    system_prompt='你是一个 helpful 的数学和天气助手，可以使用工具帮助用户。',
    checkpointer=memory
)

# 第一轮对话
config = {'configurable': {'thread_id': 'user_001'}}
result1 = agent_with_history.invoke(
    {'messages': [('user', '我叫张三')]},
    config=config
)
print('第一轮：', result1['messages'][-1].content)

# 第二轮对话，测试是否记得名字
result2 = agent_with_history.invoke(
    {'messages': [('user', '我叫什么名字？')]},
    config=config
)
print('第二轮：', result2['messages'][-1].content)
```

### **代码解释**

- `MemorySaver()`：内存型 checkpointer，用于保存对话历史

- `checkpointer=memory`：把 checkpointer 传给 `create_agent`，让智能体具备记忆能力

- `config={'configurable': {'thread_id': 'user_001'}}`：通过 thread\_id 区分不同会话

- 同一个 `thread_id` 下的历史消息会被保留并传给模型

## **处理工具调用错误**

工具执行过程中可能出现异常，比如参数错误、网络超时、文件不存在等。好的智能体需要能优雅处理这些错误。

```Python
from langchain.tools import tool

@tool
def safe_divide(a: float, b: float) -> str:
    """安全除法，除数为0时返回友好提示。"""
    try:
        if b == 0:
            return '错误：除数不能为0'
        return str(a / b)
    except Exception as e:
        return f'计算出错：{str(e)}'

safe_tools = [safe_divide, add, multiply]

safe_agent = create_agent(
    model=llm,
    tools=safe_tools,
    system_prompt='你是一个 helpful 的数学助手，可以使用工具帮助用户。'
)

result = safe_agent.invoke({'messages': [('user', '10 除以 0 等于多少？')]})
print('\n最终输出：', result['messages'][-1].content)
```

## 调用linux命令行工具

```Python
# 为智能体写一个 tools，可以执行 Linux 命令行，并且可以用自然语言来配置 Linux
# 设计思路：
#   - tool 1 run_linux_command            ：直接执行一条 Linux shell 命令行
#   - tool 2 translate_request_to_commands：把用户"自然语言"描述的配置/管理需求
#                                            翻译成具体的 Linux 命令行（不执行）
#   - 智能体（create_agent）：收到自然语言指令时先翻译成命令，再执行，实现"用自然语言配置 Linux"
#
# 在 Windows 上运行本文件也能演示：优先调用 WSL 里的真 Linux bash，
# 若本机未安装 WSL 发行版，则自动回退到 Git Bash 自带的 bash（提供 Linux 风格命令）。
import shutil
import subprocess
import sys

# 让终端统一按 UTF-8 收发，避免 Windows GBK 控制台把 emoji / 中文 / BOM 读成乱码而报错
for stream_name in ('stdin', 'stdout', 'stderr'):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.1)

# ------------------------------- 查找可用的 Linux bash -------------------------------
_WSL_READY = None   # None=尚未检测；True/False=检测结果，只探测一次避免每次都拖慢

def _wsl_has_distro():
    '''检测 WSL 是否已安装发行版（只探测一次并缓存结果）。

    注意：wsl.exe 的文本输出是 UTF-16 编码，直接解析文本容易出乱码，
    因此这里只看退出码：`wsl bash -lc "echo probe"` 能返回 0，说明发行版可用。
    '''
    global _WSL_READY
    if _WSL_READY is not None:
        return _WSL_READY
    wsl = shutil.which('wsl')
    if not wsl:
        _WSL_READY = False
        return False
    try:
        proc = subprocess.run(
            [wsl, 'bash', '-lc', 'echo probe'],
            capture_output=True,
            timeout=15,
        )
        _WSL_READY = proc.returncode == 0
    except Exception:
        _WSL_READY = False
    return _WSL_READY

def _linux_bash_candidates():
    '''返回本机真正可用的 bash 解释器命令列表，按优先级排序'''
    if sys.platform == 'win32':
        candidates = []
        wsl = shutil.which('wsl')       # Windows 子系统 Linux
        if wsl and _wsl_has_distro():   # 只有装了发行版才用 WSL，否则直接给 Git Bash
            candidates.append([wsl, 'bash', '-lc'])
        gitbash = shutil.which('bash')  # Git Bash
        if gitbash:
            candidates.append([gitbash, '-lc'])
        return candidates
    return [['bash', '-lc']]            # Linux / macOS 直接用系统 bash

def _exec_linux_command(command):
    '''用候选 bash 执行命令，WSL 未安装发行版时自动回退到下一个候选'''
    last = None
    for shell in _linux_bash_candidates():
        try:
            proc = subprocess.run(
                shell + [command],
                capture_output=True,
                encoding='utf-8',   # WSL / Git Bash 输出为 UTF-8，避免 Windows 默认 GBK 解码报错
                errors='replace',
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return None, '命令执行超时（超过 30 秒，已终止）', ''
        except Exception as e:
            return None, f'启动 bash 失败：{e}', ''
        if proc.returncode == 0:
            return proc.returncode, proc.stdout, proc.stderr
        # WSL 存在但没装发行版时，错误信息含关键词，此时继续尝试下一个 bash
        if 'no installed distributions' in proc.stderr.lower() \
                or 'is not installed' in proc.stderr.lower():
            last = proc
            continue
        return proc.returncode, proc.stdout, proc.stderr
    if last:
        return last.returncode, last.stdout, last.stderr
    return None, '未找到可用的 bash（WSL 或 Git Bash）', ''

# ------------------------------- 工具 1：直接执行 Linux 命令行 -------------------------------
@tool
def run_linux_command(command: str) -> str:
    '''执行一条 Linux shell 命令，返回命令的标准输出和标准错误。

    适合查看系统信息、文件目录操作、进程管理、安装软件等一切 Linux 运维操作。

    参数：
        command: 需要执行的完整 Linux 命令行字符串，例如 "ls -la"、"df -h"、"uname -a"。
    '''
    rc, out, err = _exec_linux_command(command)
    if rc is None:
        return f'[无法执行] {out}'
    output = (out or err or '(命令执行成功，无输出)').strip()
    if rc != 0:
        return f'命令执行失败（退出码 {rc}）：\n{output}'
    return output

# ------------------------------- 工具 2：自然语言需求 -> Linux 命令行 -------------------------------
# 把"用中文说需求"翻译成"具体命令"这件事，同样交给大模型来做
translate_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名资深 Linux 系统管理员。请把用户的自然语言需求转换为一条或多条可直接执行的 Linux shell 命令。'
               '只输出命令本身，不要解释，不要 Markdown 代码块，多个命令用 && 连接。'),
    ('user', '{request}'),
])
translate_chain = translate_prompt | llm | StrOutputParser()

@tool
def translate_request_to_commands(request: str) -> str:
    '''把用户用自然语言描述的 Linux 配置/管理需求，翻译成具体可执行的 Linux shell 命令。

    注意：本工具只负责"翻译生成命令"，不会执行命令。

    参数：
        request: 用户的自然语言需求描述，例如 "创建一个新用户叫 demo"、"查看所有磁盘占用情况"。
    '''
    try:
        return translate_chain.invoke({'request': request}).strip()
    except Exception as e:
        return f'命令翻译失败：{e}'

# ------------------------------- 组装智能体 -------------------------------
tools = [run_linux_command, translate_request_to_commands]

# 带记忆的智能体，checkpointer 自动保存/加载多轮对话
memory = MemorySaver()
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        '你是一个 Linux 运维智能体，既能直接执行 Linux 命令行，也能听懂用自然语言描述的配置需求。'
        '工作规则：\n'
        '1. 用户直接给出命令行时，直接调用 run_linux_command 执行。\n'
        '2. 用户用自然语言提出配置/管理需求时，先调用 translate_request_to_commands '
        '把需求翻译成命令，再调用 run_linux_command 执行，最后把结果整理成中文回答。\n'
        '3. 涉及危险操作（如删除数据、改系统关键配置）前要先说明风险并征求用户同意。'
    ),
    checkpointer=memory,
)

config = {'configurable': {'thread_id': 'linux_admin_001'}}

# 退出关键词：输入这些内容后结束程序
EXIT_KEYWORDS = ('退出', 'exit', 'quit', 'q')

# 交互入口：启动后在终端输入自然语言提问，回答后可以继续追问，直到输入"退出"
if __name__ == '__main__':
    print('=== Linux 运维智能体已启动 ===')
    print('提示：可直接发 Linux 命令行，也可用自然语言描述需求；输入"退出"结束对话。\n')

    while True:
        try:
            question = input('我：').strip('\ufeff \t')   # 去掉可能的 BOM / 空白，再比较关键词
        except (EOFError, KeyboardInterrupt):   # 支持 Ctrl+C / Ctrl+Z 直接退出
            print('\n再见！')
            break

        if not question:                        # 空输入则重新等待
            continue
        if question.lower() in EXIT_KEYWORDS:   # 命中退出关键词
            print('再见！')
            break

        answer = agent.invoke(
            {'messages': [('user', question)]},
            config=config,                      # 同一个 thread_id，多轮记忆自动累积
        )
        print('\n智能体：', answer['messages'][-1].content, end='\n\n')

```



