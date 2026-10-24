# Day1

# autodl的使用

- 现在网上租一台算力服务器

![image\.png](图片和附件/image%2019.png)

- 登录codebuddy，在codebuddy上远程连接这个服务器

![image\.png](图片和附件/image%201.png)

- 创建对应文件夹

```MATLAB
cd /opt
mkdir qwenpaw
cd qwenpaw/
python3    ##这里查看python版本，输入exit()退出
python3.12 -m venv .venv  #创建好python虚拟环境，防止和本机python环境打架
```

- 修改源，安装qwenpaw

```MATLAB
source .venv/bin/activate    ##每次新开都要重新激活虚拟环境
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install qwenpaw
```

## 重新开机记得启动这条命令

- 查看是否存在nvidia显卡指令`nvidia-smi`

- 初始化以及启动命令

```MATLAB
qwenpaw init --defaults
/opt/qwenpaw/.venv/bin/qwenpaw app --host 0.0.0.0 --port 8088
###也可以采用后台启动，不会顶命令
nohup /opt/qwenpaw/.venv/bin/qwenpaw app --host 0.0.0.0 --port 8088 &   #这条命令
```

- 然后就可以登录界面了

![image\.png](图片和附件/image%207.png)

- 登录界面之后自定义供应商

![image\.png](图片和附件/image%204.png)

- 这里选择的供应商是阶跃星辰stepfun，找到他的api配置

![image\.png](图片和附件/image%205.png)

- 粘贴到刚才的模型中创建

- 创建成功之后。点击刚刚创建的模型，设置，输入APIkey（这个在软件里面找）

![image\.png](图片和附件/image%2015.png)

- 之后点击模型，如果自动出现模型，证明成功了。否则则是APIkey出现了问题

- 这里教学用的是step\-3\.7\-flash

![image\.png](图片和附件/image%2011.png)

- 注意，如果订阅失败可能是订阅的专用的APIkey，跟普通的APIkey地址不同

- 可以设置一下默认的API供应商

![image\.png](图片和附件/image.png)

# llama\.cpp运行大模型

- 让他下载modelscope方便自己下载大模型

- 完成之后开启无卡模式，下载大模型

## 下载llama\.cpp（适合自己的小机器，能跑就行）

- 国内选择魔塔社区

```MATLAB
modelscope download --model unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-IQ4_XS.gguf configuration.json mmproj-F16.gguf README.md --local_dir /dev/shm/Qwen3.6-27B-MTP-GGUF
```

- 将对应的安装包下载好，这里提前给好，如果自己搭建没有安装包，可以让AI生成（将安装包导入机器中然后输入这串指令）

- AI生成办法：使用刚刚的小龙虾输入：`使用modelscope下载模型unsloth/Qwen3.6-27B-MTP-GGUF放在/autodl-fs/data目录下，后台下载，方便我随时查询进度`

- 由于我们下载GGUF格式大模型，需要llamma，可以用qwenpaw自己给生成，这里采用安装包拖到机器里面，然后解压的形式

```MATLAB
dpkg -i llama-cpp-cuda-ui_0.15.3+git72874f5.cuda12.8.ui_amd64.deb
```

- 修复指令，给的安装包没有做好，有点小bug所以需要修复指令

```MATLAB
echo "/opt/llama-cpp-cuda-ui/bin" > /etc/ld.so.conf.d/llama-cpp-cuda-ui.conf
ldconfig
llama-server-cuda-ui
```

- 启用run\.sh指令，这里注意要提前创建好autodl\-fs文件（这个是自己的硬盘，每个人路径不一样）我这里是/dev/shm

- `mkdir -p /autodl-fs/data/`（同理）

- 编写在code下编写run\.sh文件

```SQL
llama-server-cuda-ui \
 --host 0.0.0.0 \
 --port 8080 \
 --ui \
 --path /opt/llama-cpp-cuda-ui/dist \
 -m /root/autodl-tmp/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_XS.gguf \
 -mm /root/autodl-tmp/Qwen3.6-27B-MTP-GGUF/mmproj-F16.gguf \
 --mmproj-offload \
 --spec-type draft-mtp \
 -ngl 99 \
 -c 32768 \
 --parallel 1 \
 --jinja \
 --alias qwen3.6-27b-mtp-gguf
```

- 启动run\.sh的时候要注意，需要`chmod +x run.sh`给予权限。

- 可以下载nvtop查看显卡使用情况

```MATLAB
apt -y install nvtop 
nvtop 
```

## 下载VLLM模型\(稳定\)

- 要想在全局使用modelscope下载模型，可以`pip install modelscope`

# 课件的打开方式

- 找一个文件夹下载git。`winget install --id Git.Git -e --source winget`

![image\.png](图片和附件/image%2017.png)

- 找到github上对应的课件，找到代码复制

![image\.png](图片和附件/image%2016.png)

- 命令行输入`git clone 代码 `

![image\.png](图片和附件/image%202.png)

- 下载好文件后通过VScode或Codebuddy，下载插件Jupyter，然后打开

# 大模型API调用

## 开发环境搭建

- 打开codebuddy或者vscode在里面打开文件夹

![image\.png](图片和附件/image%2012.png)

- 找到一个文件夹搭建环境

![image\.png](图片和附件/image%203.png)

- 找到终端，默认是PS即powershell，输入`cmd`切换到cmd终端

![image\.png](图片和附件/image%206.png)

---

- 一定要注意在虚拟环境下安装，否则本机python可能会冲突导致无法使用

---

- 开始环境安装

### 虚拟环境切换

```MATLAB
python -m venv .venv    #搭建python虚拟环境
.venv\Scripts\activate    #切换到虚拟环境
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple    #改用全局清华源下载
copy ..\ai-agent-dev\requirements.txt .    #下载项目依赖，如果是powershell则改用cp
pip install -r requirements.txt    #读取并安装项目依赖
```

## 大模型API密钥管理

## **大模型 API 概述**

大模型（Large Language Model, LLM）通常通过 **HTTP API** 提供服务。你发送一段文本（Prompt），模型返回生成的回复。

### **主流 API 提供商**

|提供商|代表模型|特点|
|---|---|---|

\| **OpenAI** \| GPT\-4o、GPT\-4o\-mini \| 生态最成熟，很多第三方平台兼容其格式 \|

\| **Anthropic** \| Claude 3\.5 Sonnet、Claude 3 Opus \| 长文本处理能力强，安全性设计突出 \|

\| **国内厂商** \| 文心一言、通义千问、智谱 GLM \| 中文优化好，无需翻墙 \|

\> 💡 **注意**：很多国内平台（如硅基流动、智谱 AI、DashScope）提供兼容 OpenAI 格式的 API，可以直接使用 OpenAI SDK 调用。

## **API 调用核心概念**

调用大模型 API 时，你需要了解以下几个核心要素：

\- **Base URL**：API 服务的基础地址，例如 \`https://api\.openai\.com/v1\` 或 \`https://api\.anthropic\.com\`

\- **API Key**：身份验证密钥，类似于密码，**绝对不能泄露**

\- **Model**：模型名称，本课程统一使用 \`qwen3\.6\-35b\-A3b\`

\- **Message / Prompt**：发送给模型的输入内容

\- **Temperature**：控制输出随机性（0\~2，越低越确定，越高越有创意）

\- **Max Tokens**：限制模型输出的最大长度

### 使用\.env文件管理密钥

- 创建\.env文件并在其中填写密钥信息

![image\.png](图片和附件/image%2013.png)

- 调用方式（此处为实例）

    ```Plain Text
    # 请使用老师提供的 base_url 和 api_key
    OPENAI_BASE_URL=https://your-openai-compatible-base-url.com/v1
    OPENAI_API_KEY=sk-your-openai-api-key
    
    ANTHROPIC_BASE_URL=https://api.anthropic.com
    ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key
    ```

- 创建demo01\.py检验，环境变量是否安装完全

![image\.png](图片和附件/image%2018.png)

- 在终端使用`python dmeo01.py`开启指令

```Python
# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
import os

load_dotenv()  # 默认加载当前目录下的 .env 文件

# 验证环境变量是否加载成功
openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

print("✅ OPENAI_BASE_URL 已设置" if openai_base_url else "❌ OPENAI_BASE_URL 未设置")
print("✅ OPENAI_API_KEY 已设置" if openai_api_key else "❌ OPENAI_API_KEY 未设置")
print("✅ ANTHROPIC_BASE_URL 已设置" if anthropic_base_url else "❌ ANTHROPIC_BASE_URL 未设置")
print("✅ ANTHROPIC_API_KEY 已设置" if anthropic_api_key else "❌ ANTHROPIC_API_KEY 未设置")
```

- 检验出缺少的模块

![image\.png](图片和附件/image%209.png)

- 发现缺少dotenv文件，需要下载`pip install dotenv`，因为我们使用的是\.env的形式使用API，所以需要这个dotenv

- 下载完成后再次检验，结果设置成功

![image\.png](图片和附件/image%208.png)

- 如果此时依旧显示未设置，可以采用别的APIKEY

- 此时可以调用大模型

```Python
from openai import OpenAI
from dotenv import load_dotenv
import os

#加一个_变量赋值，编译器认可
_ = load_dotenv()

# 初始化客户端（使用环境变量中的配置）
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# 发送对话请求
response = client.chat.completions.create(
    model="glm-5.2",  # 请根据老师提供的实际模型名称修改
    messages=[
        {"role": "system", "content": "你是一个 helpful 的 AI 助手。"},
        {"role": "user", "content": "请用一句话解释什么是大语言模型。"}
    ],
    temperature=0,
    max_tokens=512
)

# 输出模型回复
print("🤖 AI 回复：")
print(response.choices[0].message.content)
```

- 如果有回复且为空，可以试着调大max\_tokens参数

![image\.png](图片和附件/image%2010.png)

- while循环版本：

```Python
from openai import OpenAI
from dotenv import load_dotenv
import os

#加一个_变量赋值，编译器认可
_ = load_dotenv()

    # 初始化客户端（使用环境变量中的配置）
client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
while True:
    msg = input(">>> ")      
    # 发送对话请求
    response = client.chat.completions.create(
        model="glm-5.2",  # 请根据老师提供的实际模型名称修改
        messages=[
            {"role": "system", "content": "你是一个 helpful 的 AI 助手。"},
            {"role": "user", "content": msg}
        ],
        temperature=0,
        max_tokens=512
    )
    
    # 输出模型回复
    print("🤖 AI 回复：")
    print(response.choices[0].message.content)
```

## 多轮对话

- 多轮对话可以让AI理解上下文作出回答

![image\.png](图片和附件/image%2014.png)

- 实例代码

```Python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
)

#维护对话历史
messages = [
    {"role": "system", "content":"你是一个helpful的AI助手"}
]

#第一轮对话
messages.append({"role": "user", "content":"你好我叫李二毛。"})
response = client.chat.completions.create(
    model="glm-5.2",
    messages=messages,
    temperature=0.5
    )
reply = response.choices[0].message.content
print(f"机器人:{reply}")

#将模型回复加入历史,这里assistant就是AI的意思
messages.append({"role":"assistant","content":reply})

#第二轮对话（模型记得我叫小明）
messages.append({"role": "user", "content": "你还记得我的名字吗？"})
response = client.chat.completions.create(
    model="glm-5.2",
    messages=messages,
    temperature=0.5
)
reply = response.choices[0].message.content
print(f"机器人：{reply}")
```

- 实现while自循环

```Python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
)

#维护对话历史
messages = [
    {"role": "system", "content":"你是一个helpful的AI助手"}
]
while True:
    msg=input(">>> ")
    messages.append({"role": "user", "content":msg})
    response = client.chat.completions.create(
        model="glm-5.2",
        messages=messages,
        temperature=0.5
        )
    reply = response.choices[0].message.content
    print(f"机器人:{reply}")

    #将模型回复加入历史,这里assistant就是AI的意思
    messages.append({"role":"assistant","content":reply})
```

## 流体输出

- 让回复变成一个字一个字连续回复

```Python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
)

messages = [
    {"role": "system", "content":"你是一个helpful的AI助手"}
]
while True:
    msg=input(">>> ")
    messages.append({"role": "user", "content":msg})
    response = client.chat.completions.create(
        model="glm-5.2",
        messages=messages,
        stream=True,
        temperature=0.5
        )
    print(f"机器人: ", end="")
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
    messages.append({"role": "assistant","content": "reply"})
```

# LangChain概述

## **为什么需要 LangChain**

前两节课中，我们直接通过 openai 或 anthropic SDK 调用大模型 API。这种方式简单直接，但当应用变得复杂时，会遇到以下问题：

\- **Prompt 管理混乱**：硬编码在代码中，难以维护和复用

\- **代码重复**：每次调用都要写重复的 API 初始化、参数配置

\- **模型切换困难**：换一个模型提供商需要重写大量代码

\- **缺乏标准流程**：没有统一的输入处理、输出解析规范

**LangChain** 就是为了解决这些问题而生的框架。它将大模型应用的开发流程抽象为标准化组件，让开发者可以像搭积木一样构建复杂的 AI 应用。

## **LangChain 是什么**

LangChain 是一个用于构建基于大语言模型（LLM）应用的 Python/JS 框架，由 Harrison Chase 于 2022 年创建。

### **核心设计思想**

\- **组件化**：将大模型应用拆解为可复用的标准组件

\- **链式组合**：通过管道（\|）将多个组件串联成处理流程

\- **模型无关**：同一套代码可以切换不同厂商的模型

\- **生态丰富**：提供大量预集成工具（向量库、搜索引擎、数据库等）

## LangChain 核心架构

- LangChain 将大模型应用抽象为四大核心模块：

### 四大核心模块

|模块|作用|本课程对应章节|
|---|---|---|
|**Model I/O**|模型输入输出管理（Prompt、Model、Output Parser）|05\-07 节|
|**Retrieval**|检索增强生成（RAG）：文档加载、向量化、检索|11\-13 节|
|**Chains**|链式组合，将多个组件串联成工作流|05、08 节|
|**Agents**|智能体：让模型自主决策、调用工具完成任务|09\-10 节|

## 简单使用langchain

- 创建一个demo02\.py

```Python
# 1. 导入LangChain封装兼容OpenAI接口的聊天模型类（智谱GLM-5.2兼容OpenAI协议，所以用这个类）
from langchain_openai import ChatOpenAI

# 2. 导入聊天提示词模板，用来组装system、user消息
from langchain_core.prompts import ChatPromptTemplate

# 3. 导入环境变量加载工具，读取.env文件里的API_KEY、BASE_URL
from dotenv import load_dotenv

# 4. 加载当前目录.env文件中的环境变量；_代表接收返回值但不使用
_ = load_dotenv()

# 5. 实例化大模型对象，指定使用 glm-5.2 模型
llm = ChatOpenAI(model="glm-5.2")

# 6. 创建聊天提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个幽默的助手，总会让用户哈哈大笑"), # 系统提示：设定AI角色
    ("user","{input}") # 用户消息模板，{input}是占位符，后续填充用户输入
])

# 7. LCEL管道语法 | 构建执行链：先填充prompt → 传给大模型
chain = prompt | llm 

# 8. 执行链条：
# input(">>> ") 控制台接收用户输入
# chain.invoke({变量名:值}) 把用户文本填入{input}占位符，调用模型
# .content 取出AI返回消息里的文本内容，打印输出
print(chain.invoke({
    'input': input(">>> ")
}).content)
```

## LangChain核心组件速览

### **Model（模型）**

LangChain 封装了各大厂商的模型，统一接口：

```Python
from langchain_openai import ChatOpenAI      # OpenAI 兼容格式
from langchain_anthropic import ChatAnthropic # Anthropic Claude
from langchain_community.llms import Tongyi    # 阿里通义千问

llm = ChatOpenAI(model='qwen3.6-35b-A3b')
```

所有模型都支持 \.invoke\(\)、\.stream\(\)、\.batch\(\) 等统一方法。

### **Prompt（提示模板）**

将 Prompt 模板化，支持变量插入和复用：

```Python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个{role}。'),
    ('user', '请{task}：{content}')
])

prompt.format(role='翻译助手', task='翻译', content='Hello')
```

### **Output Parser（输出解析器）**

将模型输出的文本解析为结构化数据（JSON、列表、Pydantic 对象等）：

```Python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = prompt | llm | parser  # 模型输出自动解析为 JSON
```

### **Chain（链）**

链是 LangChain 的核心概念，表示数据的处理流程。使用 \| 运算符组合：

```Python
chain = prompt | llm | parser
result = chain.invoke({'input': '用户输入'})
```

数据流：用户输入 \-\> Prompt模板 \-\> LLM \-\> 输出解析 \-\> 最终结果

## LangChain 生态工具

|工具|作用|
|---|---|
|**LangGraph**|构建复杂状态图和多智能体工作流（本课程核心内容）|
|**LangServe**|将 LangChain 链部署为 REST API 服务|
|**LangSmith**|调试、监控和评估 LangChain 应用的可观测性平台|
|**LangChain Hub**|社区共享的 Prompt 模板市场|

# LLM链

## **什么是 LLMChain**

LLMChain 是 LangChain 中最基础、最常用的链类型。它表示一个简单的处理流程：

```Plain Text
用户输入 -> Prompt模板 -> 大模型 -> 输出结果
```

在 LangChain 的新版（LCEL）中，我们不再使用专门的 `LLMChain` 类，而是直接用 `|` 管道符将组件串联起来：

```Python
chain = prompt | llm
```

这种方式更直观、更灵活，也是本课程推荐的标准写法。

## **数据流详解**

当调用 `chain.invoke({'text': '人工智能正在改变世界'})` 时，数据在链中是这样流动的：

```Plain Text
输入: {'text': '人工智能正在改变世界'}
  |
  v
prompt: 将变量填入模板
  system: 你是一个专业的翻译助手，只输出翻译结果，不做解释。
  user: 请将以下文本翻译成英文：人工智能正在改变世界
  |
  v
llm: 调用大模型 API
  |
  v
输出: AIMessage(content='AI is changing the world.')
```

链的每个环节都是 **Runnable**（可运行对象），它们都实现了统一的接口：

- `.invoke(input)` — 单条调用

- `.batch(inputs)` — 批量调用

- `.stream(input)` — 流式输出

- `.ainvoke(input)` — 异步调用

## 多变量输入

Prompt模版中可以定义多个变量，invoke时传入字典即可

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.5)

#多变量 Prompt模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一位{style}作家。'),
    ('user', '请用{style}的风格写一段关于{topic}的{format},约{length}字')
])

chain = prompt | llm

result = chain.invoke({
    'style': '浪漫主义',
    'topic': '恋爱',
    'format':'散文',
    'length': '100'
})
print(result.content)
```

## 批量调用（batch\)

当需要处理多条输入时，使用 `.batch()` 可以一次性发送多个请求，效率更高：

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.5)

#多变量 Prompt模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个简洁的助手，只输出结果'),
    ('user', '将以下中文翻译成英文:{text}')
])

chain = prompt | llm
#本质是列表
results = chain.batch([   
    {'text': '西游记'},
    {'text': '红楼梦'},
    {'text': '水浒传'},
    {'text': '三国演义'},
])
#枚举打印输出
for i, result in enumerate(results):    
    print(f"{i+1}=>{result.content}")
```

## 流式输出

- 一部分一部分输出

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.5)

#多变量 Prompt模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '专业的作家，擅长根据用户给的提示词写散文'),
    ('user', '{text}')
])

chain = prompt | llm

print("开始生成: ", end="")    #清除空格

for chunk in chain.stream({'text': '荷塘月色'}):
    print(chunk.content, end="")

print()    #换行
```

## 链的重用与组合

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

_ = load_dotenv()
# 创建共享的 LLM 实例
llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

# ========== 1. 定义可复用的翻译链 ==========
# 先创建翻译用的 Prompt 模板
translate_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个翻译助手，将中文翻译成英文。'),
    ('user', '{text}')  # {text} 是输入变量
])

# 构建翻译链：Prompt -> LLM -> 字符串输出解析器
# StrOutputParser() 的作用：把模型返回的 AIMessage 对象直接转成 str
translate_chain = translate_prompt | llm | StrOutputParser()

# ========== 2. 链的复用 ==========
# 同一个 translate_chain 可以像函数一样多次调用
# 每次 invoke() 传入不同的 {'text': '...'}，得到对应的英文翻译
print(translate_chain.invoke({'text': '苹果'}))
print(translate_chain.invoke({'text': '香蕉'}))
print(translate_chain.invoke({'text': '桃子'}))

# ========== 3. 链的组合 ==========
# 再定义一个总结链，同样接收 {text} 变量
summarize_prompt = ChatPromptTemplate.from_messages([
    ('system', '请用中文将以下内容总结为一句话，只输出总结结果。'),
    ('user', '{text}')
])
summarize_chain = summarize_prompt | llm | StrOutputParser()

# 组合链：翻译后的英文传给总结链
# translate_chain 的输出是字符串，但 summarize_chain 需要字典 {'text': ...}
# 所以中间加一个 lambda 函数做格式转换
full_chain = translate_chain | (lambda x: {'text': x}) | summarize_chain

# 运行组合链：输入中文 -> 翻译成英文 -> 总结成中文
result = full_chain.invoke({'text': '大语言模型写代码的能力越来越强。使用ai开发代码让传统的软件开发方式被改变。'})
print(f'\n翻译+总结：{result}')
```

## 链的调试与查看中间结果

使用 `StrOutputParser()` 可以自动将模型输出转换为字符串，省去手动 `.content` 的步骤。

如果想查看链运行过程中的中间状态（比如格式化后的 Prompt 是什么样的），可以使用回调或调试工具：

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个助手。'),
    ('user', '你好，我叫{name}')
])

# 查看格式化后的 Prompt
formatted = prompt.format_messages(name='小明')
print('格式化后的 Prompt：')
for msg in formatted:
    print(f'  {msg.type}: {msg.content}')

chain = prompt | llm
result = chain.invoke({'name': '小明'})
print(f'\n模型回复：{result.content}')
```



