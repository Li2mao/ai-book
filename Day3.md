# Day3

# 检索增强（RAG）

## **为什么需要 RAG**

大语言模型有两个明显局限：

\- **知识过时**：模型训练数据有截止日期，不知道最新信息

\- **容易产生幻觉**：对未训练过的问题可能编造答案

**RAG** 通过在回答前先从外部知识库中检索相关信息，把检索到的内容作为上下文喂给模型，从而：

- 让模型基于最新、准确的资料回答

- 减少幻觉

- 实现私有知识库问答（如公司文档、个人笔记）

## **RAG 的核心流程**

一个完整的 RAG 系统通常包含以下步骤：

```Plain Text
1. 加载文档     -> DocumentLoader
2. 分割文档     -> TextSplitter
3. 文本向量化   -> Embeddings
4. 存入向量库   -> VectorStore
5. 用户提问     -> Query
6. 检索相关片段 -> Retriever
7. 拼接上下文   -> Context
8. 生成回答     -> LLM
```

其中第 1\-4 步是「离线」的索引阶段，第 5\-8 步是「在线」的查询阶段。

## **环境准备**

本节课会用到文档加载、文本分割组件。如果你的虚拟环境中还没有安装，请先执行：

```PowerShell
pip install langchain-community langchain-text-splitters -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果暂时不想安装，本课件也提供了用 Python 原生代码实现的备用方案。

## **准备示例文档**

先创建一个示例文本文件，后面会用它来演示 RAG 流程。

```Python
sample_text = """
人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能才能完成的任务的系统。
机器学习是人工智能的一个重要子领域。它通过数据训练模型，使计算机能够从经验中学习，而无需进行明确的编程。
深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。深度学习在图像识别、自然语言处理和语音识别等领域取得了显著成果。
自然语言处理（Natural Language Processing，简称 NLP）是人工智能和语言学的交叉领域，研究如何让计算机理解、解释和生成人类语言。
大语言模型（Large Language Model，简称 LLM）是基于深度学习的自然语言处理模型。它们通过在海量文本数据上进行预训练，能够生成连贯的文本、回答问题并完成多种语言任务。
"""

with open('sample_rag.txt', 'w', encoding='utf-8') as f:
    f.write(sample_text.strip())

print('示例文档已创建：sample_rag.txt')
```

### **使用 LangChain 的 TextLoader（推荐）**

`TextLoader` 可以自动处理编码、元数据等问题。

> 注意：`langchain-community` 在 LangChain 1\.x 中已被标记为弃用，但仍然可用。代码中用 `warnings.filterwarnings('ignore', category=DeprecationWarning)` 过滤了警告。
> 
> 

```Python
import warnings

# langchain-community 在 LangChain 1.x 中已被标记为弃用，但仍然可用
# 这里过滤掉 DeprecationWarning，避免输出中显示警告信息
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader

loader = TextLoader('sample_rag.txt', encoding='utf-8')
documents = loader.load()

print(f'加载了 {len(documents)} 个文档')
print(f'第一个文档长度：{len(documents[0].page_content)} 字符')
print(f'元数据：{documents[0].metadata}')
```

### **原生 Python 加载方式（备用）**

如果暂时不想安装 `langchain-community`，可以直接用 Python 读取文件。

```Python
from langchain_core.documents import Document

with open('sample_rag.txt', 'r', encoding='utf-8') as f:
    text = f.read()

documents = [Document(page_content=text, metadata={'source': 'sample_rag.txt'})]
print(f'加载了 {len(documents)} 个文档，长度：{len(documents[0].page_content)} 字符')
```

## **文档分割**

大模型一次能处理的文本长度有限，而且过长的文档会影响检索精度。因此需要把长文档切分成小块。

```Python
import warnings

# langchain-community 在 LangChain 1.x 中已被标记为弃用，但仍然可用
# 这里过滤掉 DeprecationWarning，避免输出中显示警告信息
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader

loader = TextLoader('sample_rag.txt', encoding='utf-8')
documents = loader.load()

print(f'加载了 {len(documents)} 个文档')
print(f'第一个文档长度：{len(documents[0].page_content)} 字符')
print(f'元数据：{documents[0].metadata}')

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建一个文本分割器
# chunk_size: 每个块的最大字符数
# chunk_overlap: 相邻块之间的重叠字符数，用于保持上下文连贯
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=['\n\n', '\n', '。', '，', ' ', '']
)

chunks = splitter.split_documents(documents)

print(f'分割成 {len(chunks)} 个文本块')
for i, chunk in enumerate(chunks[:3]):
    print(f'\n--- 第 {i+1} 块 ---')
    print(chunk.page_content[:80] + '...')
```

### **代码解释**

- `chunk_size`：每个文本块的最大长度

- `chunk_overlap`：相邻块重叠的字符数，避免关键信息被切分断掉

- `separators`：切分优先级，先按段落切，再按句子切，最后按字符切

- `RecursiveCharacterTextSplitter` 会尽量在语义边界处切分

## **文本向量化**

向量化是把文本转换成数值向量的过程。语义相近的文本，向量距离也更近。

本课件使用 `OpenAIEmbeddings`。如果你的 API 不支持 `qwen3-embedding`，请把 `model` 参数改成你实际可用的 embedding 模型名。

- 注意要使用支持embedding的模型，否则文本向量化不成功

- 去使用的大模型官网查找API模型，查看是否包含embedding模型

```Python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

_ = load_dotenv()

#创建embedding模型
#如果你的API提供的是其他embedding模型名称，请修改model参数
embeddings = OpenAIEmbeddings(model='Embedding-2')

#测试向量化
test_text = '人工智能是计算机科学的分支'
vector = embeddings.embed_query(test_text)

print(f'文本：{test_text}')
print(f'向量维度：{len(vector)}')
print(f'向量前5个值：{vector[:5]}')
```

## **向量存储与检索**

把分割后的文本块存入向量数据库。这里使用 `InMemoryVectorStore`，它是一个内存型向量存储，适合学习和测试。

```Python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
from langchain_core.vectorstores import InMemoryVectorStore 

_ = load_dotenv()

#创建embedding模型
#如果你的API提供的是其他embedding模型名称，请修改model参数
embeddings = OpenAIEmbeddings(model='Embedding-2')

# langchain-community 在 LangChain 1.x 中已被标记为弃用，但仍然可用
# 这里过滤掉 DeprecationWarning，避免输出中显示警告信息
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader

loader = TextLoader('sample_rag.txt', encoding='utf-8')
documents = loader.load()

# 创建一个文本分割器
# chunk_size: 每个块的最大字符数
# chunk_overlap: 相邻块之间的重叠字符数，用于保持上下文连贯
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=['\n\n', '\n', '。', '，', ' ', '']
)

chunks = splitter.split_documents(documents)

#创建内存向量库
vectorstore =  InMemoryVectorStore(embeddings)

#把文本块加入向量库
vectorstore.add_documents(chunks)

#创建检索器，k=2表示返回最相关的2个字段
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

#测试检索
query = '什么是深度学习'
retrieved_docs = retriever.invoke(query)

print(f'检索到{len(retrieved_docs)}个相关片段：\n')
for i, doc in enumerate(retrieved_docs):
    print(f'--- 片段 {i+1} ---')
    print(doc.page_content)
    print(f'来源：{doc.metadata}\n')
```

### **代码解释**

- `InMemoryVectorStore(embeddings)`：创建内存向量库，使用指定的 embedding 模型

- `add_documents(chunks)`：把文本块存入向量库

- `as_retriever(search_kwargs={'k': 2})`：把向量库转成检索器，每次返回前 2 个最相关结果

- `retriever.invoke(query)`：根据用户问题检索相关文本块

## **构建 RAG 问答链**

把检索器和 LLM 组合起来，构建一个完整的 RAG 链。流程是：

1. 接收用户问题

2. 用检索器找到相关文档

3. 把文档内容拼接到 Prompt 中

4. 让 LLM 基于上下文回答

```Python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

warnings.filterwarnings('ignore', category=DeprecationWarning)

_ = load_dotenv()

llm = ChatOpenAI(model='glm-4.6v', temperature=0.7)

# 创建 embedding 模型（chunk_size=64 是分批大小，需 ≤64 条/次，见前面注释）
embeddings = OpenAIEmbeddings(model='Embedding-2', chunk_size=64)

loader = TextLoader('sample_rag.txt', encoding='utf-8')
documents = loader.load()

#创建一个文本分割器
#chunk_size：每个块的最大字符数
#chunk_overlap：相邻块之间的重叠字符数，用于保持上下文连贯
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=20,
    separators=['\n\n','\n','。',',','，']
)
chunks = splitter.split_documents(documents)

#创建内存向量库
vectorstore = InMemoryVectorStore(embeddings)

#把文本块加入向量库
vectorstore.add_documents(chunks)

#创建检索器, k=2 表示返回最相关的两个片段
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

#定义RAG Prompt模板
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个知识库回答助手。请根据下面的上下文回答问题，如果上下文没有相关信息，请说【我不知道】。'),
    ('user', '''上下文：\n{context}\n\n问题：{question}''')
])

#把检索到的文档拼接成字符串
def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

#构建RAG链
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | rag_prompt
    | llm 
    | StrOutputParser()
)

#测试
question = '深度学习和大语言模型有什么关系？'
answer = rag_chain.invoke(question)
print('问题：', question)
print('\n回答：', answer)

```

# 向量数据库

## **为什么需要向量数据库**

上一节课我们用 `InMemoryVectorStore` 把向量存在内存中，它适合学习和测试，但有几个明显缺点：

\- **数据无法持久化**：程序关闭后向量就消失了

\- **无法增量更新**：不能方便地添加、删除文档

\- **不支持复杂查询**：无法按元数据过滤

\- **不适合大规模数据**：内存容量有限

**向量数据库** 就是为解决这些问题而生的专用数据库。

## **向量数据库 vs 传统数据库**

|特性|传统数据库（MySQL）|向量数据库（ChromaDB）|
|---|---|---|
|存储内容|结构化数据|向量 \+ 原始文本 \+ 元数据|
|查询方式|精确匹配、范围查询|相似度搜索|
|索引类型|B\+ 树|HNSW、IVF 等近似最近邻索引|
|典型应用|订单、用户管理|语义搜索、RAG、推荐|

向量数据库的核心能力：**根据语义相似度快速找到最相关的向量**。

## **核心概念**

### **向量（Vector）**

文本经过 Embedding 模型编码后变成的高维数组。例如一个 2560 维的浮点数数组。

### **相似度（Similarity）**

两个向量之间的距离。常用度量方式：

\- **余弦相似度（Cosine Similarity）**：衡量方向是否一致，范围 \-1 到 1

\- **欧氏距离（Euclidean Distance）**：衡量向量空间中的直线距离

### **索引（Index）**

为了加速海量向量的相似度搜索，向量数据库会构建专门的索引结构，如 HNSW。

## **常见向量数据库**

|数据库|特点|适用场景|
|---|---|---|

\| **ChromaDB** \| 轻量、易用、支持持久化 \| 本地开发、中小规模 \|

\| **FAISS** \| Meta 开源、高性能 \| 大规模向量检索 \|

\| **Milvus** \| 企业级、分布式 \| 生产环境、海量数据 \|

\| **Pinecone** \| 云端托管、无需运维 \| 快速上线、Serverless \|

\| **Weaviate** \| 支持多模态、GraphQL \| 复杂查询、多模态 \|

本节课以 **ChromaDB** 为例，因为它安装简单、接口友好，非常适合学习。

## **安装 ChromaDB**

在虚拟环境中执行：

```PowerShell
pip install chromadb -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装完成后需要**重启 Jupyter 内核**。

## **使用 ChromaDB 创建向量库**

下面演示如何把文档存入 ChromaDB 并进行检索。

```Python
import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

# 创建 embedding 模型
embeddings = OpenAIEmbeddings(
    model='Embedding-2',
    openai_api_base=os.environ.get('OPENAI_BASE_URL'),
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

# 准备文档
documents = [
    Document(page_content='人工智能是计算机科学的一个分支。', metadata={'source': 'ai_book', 'category': 'AI'}),
    Document(page_content='深度学习使用多层神经网络。', metadata={'source': 'ai_book', 'category': 'AI'}),
    Document(page_content='机器学习让计算机从数据中学习。', metadata={'source': 'ai_book', 'category': 'AI'}),
    Document(page_content='香蕉是一种黄色的水果。', metadata={'source': 'food_book', 'category': 'food'}),
    Document(page_content='苹果通常是红色或绿色的。', metadata={'source': 'food_book', 'category': 'food'})
]

# 从文档创建 ChromaDB 向量库
# persist_directory 指定数据保存目录
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory='./chroma_db_demo'
)

print('向量库创建成功！')
print(f'文档数量：{vectorstore._collection.count()}')
```

### **代码解释**

- `Chroma.from_documents()`：从 Document 列表直接创建向量库

- `embedding=embeddings`：指定向量化模型

- `persist_directory`：数据持久化目录，程序关闭后数据仍存在

## **持久化加载\+相似度检索**

ChromaDB 的数据已经保存在 `./chroma_db_demo` 目录中。下次启动程序时，可以直接加载，不需要重新向量化。

```Python
import warnings
from dotenv import load_dotenv

_ = load_dotenv()
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(
    model='Embedding-2',
    openai_api_base=os.environ.get('OPENAI_BASE_URL'),
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

#从已有目录加载向量库
loaded_vectorstore = Chroma(
    persist_directory='./chroma_db_demo',
    embedding_function=embeddings
)

loaded_retriever = loaded_vectorstore.as_retriever(search_kwargs={'k': 2})
results = loaded_retriever.invoke('深度学习和神经网络的关系')

print('从持久化数据库加载后检索：')
for doc in results:
    print(f'- {doc.page_content}')

```

## **元数据过滤**

元数据过滤可以让我们在检索时只搜索特定类别的文档。例如只搜索 `category='AI'` 的文档。

```Python
import warnings
from dotenv import load_dotenv

_ = load_dotenv()
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(
    model='Embedding-2',
    openai_api_base=os.environ.get('OPENAI_BASE_URL'),
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

#从已有目录加载向量库
loaded_vectorstore = Chroma(
    persist_directory='./chroma_db_demo',
    embedding_function=embeddings
)

#只检索category为AI的文档
filtered_results = loaded_vectorstore.similarity_search(
    '水果的颜色',    #语义上匹配食物类
    filter={'category': 'AI'},    #但是实际只在AI类里面搜索
    k=3    #最终只会在AI类里面返回3个答案
)

print('过滤后只搜索AI类文档：')
for doc in filtered_results:
    print(f'-[{doc.metadata["category"]}] {doc.page_content}')

#只检索category为food的文档
food_results = loaded_vectorstore.similarity_search(
    '人工智能',
    filter={'category': 'food'},
    k=3
)

print('\n过滤后只搜索food类文档：')
for doc in food_results:
    print(f'-[{doc.metadata["category"]}] {doc.page_content}')
```

### **代码解释**

- `filter={'category': 'AI'}`：只返回 category 字段等于 AI 的文档

- 即使查询词是「水果的颜色」，由于过滤条件限制，也只会返回 AI 类文档

- 元数据过滤常用于多租户、多知识库场景

## **增量更新：添加和删除文档**

向量数据库支持动态增删文档，不需要每次重新构建整个库。

```Python
import warnings
from dotenv import load_dotenv

_ = load_dotenv()
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(
    model='Embedding-2',
    openai_api_base=os.environ.get('OPENAI_BASE_URL'),
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

#从已有目录加载向量库
loaded_vectorstore = Chroma(
    persist_directory='./chroma_db_demo',
    embedding_function=embeddings
)

# 添加新文档
new_docs = [
    Document(page_content='自然语言处理让计算机理解人类语言。', metadata={'source': 'ai_book', 'category': 'AI'})
]

loaded_vectorstore.add_documents(new_docs)
print(f'添加后文档数量：{loaded_vectorstore._collection.count()}')

#查询刚添加的文档
results = loaded_vectorstore.similarity_search('NLP是什么', k=2)
for doc in results:
    print(f'- {doc.page_content}')
```

## 多轮检索示例

```Python
import warnings
from dotenv import load_dotenv

_ = load_dotenv()
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(
    model='Embedding-2',
    openai_api_base=os.environ.get('OPENAI_BASE_URL'),
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

#从已有目录加载向量库
loaded_vectorstore = Chroma(
    persist_directory='./chroma_db_demo',
    embedding_function=embeddings
)

#使用持久化加载的向量库创建检索器
retriever = loaded_vectorstore.as_retriever(search_kwargs={'k': 2})

#多轮检索示例
queries = [
    '人工智能是什么？',
    '机器学习和深度学习的关系',
    '水果有哪些颜色？'
]

for query in queries:
    print(f'\n===查询：{query}===')
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f'{i+1}.[{doc.metadata["category"]}] {doc.page_content}')
```

# RAG构建

## **1\. RAG 应用整体架构**

一个可落地的 RAG 应用通常包含以下模块：

```Plain Text
数据准备阶段：
  原始文档 -> 文档加载器 -> 文本分割器 -> 文本块

索引阶段：
  文本块 -> Embedding 模型 -> 向量 -> 向量数据库

查询阶段：
  用户问题 -> Embedding 模型 -> 向量检索 -> 相关文本块
  相关文本块 + 用户问题 -> Prompt -> LLM -> 最终回答
```

本节课将以「公司产品手册问答系统」为例，完整演示上述流程。

## **2\. 准备知识库文档**

先创建两份示例文档，模拟真实的产品手册。

```Python
import os

# 创建产品手册目录
os.makedirs('product_docs', exist_ok=True)

# 文档 1：智能手表使用手册
watch_manual = """
智能手表使用手册

1. 开机与配对
长按右侧电源键 3 秒开机。首次使用需在手机上下载 HealthApp，打开蓝牙后搜索设备并完成配对。

2. 心率监测
手表背面配备 PPG 心率传感器。在表盘界面下滑进入功能菜单，点击「心率」即可开始测量。测量时请保持手腕静止。

3. 睡眠监测
佩戴手表入睡后，系统会自动记录睡眠数据。次日可在 HealthApp 中查看深睡、浅睡和 REM 睡眠时长。

4. 防水说明
本手表支持 5ATM 防水，可佩戴游泳，但不适用于潜水、热水淋浴或桑拿。
"""

# 文档 2：无线耳机使用手册
earphone_manual = """
无线耳机使用手册

1. 开机与配对
打开充电盒，耳机自动开机并进入配对模式。在手机蓝牙设置中选择 SoundPod 即可完成配对。

2. 触控操作
单击左耳或右耳可播放/暂停音乐。双击右耳切换下一首，双击左耳切换上一首。长按 2 秒唤醒语音助手。

3. 充电说明
耳机单次续航 6 小时，充电盒可提供额外 24 小时续航。使用 Type-C 线缆为充电盒充电，约 1.5 小时充满。

4. 降噪功能
耳机支持主动降噪和环境音模式。在连接状态下，长按右耳 3 秒可在两种模式间切换。
"""

with open('product_docs/watch_manual.txt', 'w', encoding='utf-8') as f:
    f.write(watch_manual.strip())

with open('product_docs/earphone_manual.txt', 'w', encoding='utf-8') as f:
    f.write(earphone_manual.strip())

print('产品手册已创建：')
for f in os.listdir('product_docs'):
    print(f' - {f}')
```

## **3\. 完整 RAG 构建流程**

下面按照「加载 \-\> 分割 \-\> 向量化 \-\> 存储 \-\> 检索 \-\> 生成」的顺序，一步步构建系统。

```Python
import warnings
from dotenv import load_dotenv
import os

load_dotenv()

warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

#1.加载目录下所有txt文件
loader = DirectoryLoader(
    'product_docs',
    glob='*.txt',
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()
print(f'加载文档数：{len(documents)}')

#2.分隔文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=['\n\n', '\n', '。', '']
)
chunks = splitter.split_documents(documents)
print(f'分隔后文本块数：{len(chunks)}')

#3.创建embedding模型
embeddings = OpenAIEmbeddings(
    model='Embedding-2',
    openai_api_base=os.environ.get('OPENAI_BASE_URL'),
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

#4.存入ChromDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='./product_knowledge_db'
)

print('向量数据库创建成功')
```

### **代码解释**

- `DirectoryLoader`：批量加载目录下所有匹配文件

- `glob='*.txt'`：只加载 txt 文件

- `loader_cls=TextLoader`：指定使用 TextLoader 加载每个文件

- `RecursiveCharacterTextSplitter`：按语义边界切分文档

- `Chroma.from_documents`：一次性完成向量化和存储

## **4\. 构建问答链**

使用 LCEL 把检索器和 LLM 组合起来。

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import warnings
import os 

warnings.filterwarnings('ignore', category=DeprecationWarning)

_ = load_dotenv()

llm = ChatOpenAI(model='glm-5.2', temperature=0.1)

prompt = ChatPromptTemplate.from_messages([
    ('system', '''你是公司产品客服助手。请严格根据提供的产品手册内容回答用户问题。
如果手册中没有相关信息，请明确告知「根据现有手册，我无法回答这个问题」，不要编造。'''),
    ('user', '''相关手册内容：\n{context}\n\n用户问题：{question}''')
])

def format_docs(docs):
    return '\n\n'.join([f'[来自{doc.metadata["source"]}] {doc.page_content}' for doc in docs])

embeddings = OpenAIEmbeddings(
    model='Embedding-2',
    openai_api_base=os.environ.get('OPENAI_BASE_URL'),
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory='./product_knowledge_db'
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

qa_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#测试
questions = [
    '智能手表怎么测心率？',
    '耳机的降噪功能怎么用？',
    '这款手表支持潜水吗？',
    '手机怎么连接打印机？'  #不在手册中
]

for q in questions:
    print(f'Q: {q}')
    print(f'A: {qa_chain.invoke(q)}\n')
```

## **5\. 模块化封装**

把 RAG 系统封装成一个类，便于在项目中复用。创建rag\.py

```Python
import os
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ProductQASystem:
    """基于 ChromaDB 的产品手册问答系统。"""

    def __init__(self, persist_dir, llm_model='glm-5.2', embedding_model='Embedding-2'):
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=os.environ.get('OPENAI_BASE_URL'),
            openai_api_key=os.environ.get('OPENAI_API_KEY')
        )
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.chain = self._build_chain()

    def _format_docs(self, docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ('system', '你是产品客服助手，请根据手册内容回答问题。'),
            ('user', '''手册内容：\n{context}\n\n问题：{question}''')
        ])
        retriever = self.vectorstore.as_retriever(search_kwargs={'k': 3})
        return (
            {'context': retriever | self._format_docs, 'question': RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        """回答用户问题。"""
        return self.chain.invoke(question)

    def add_documents(self, file_paths):
        """增量添加新文档。"""
        loaders = [TextLoader(path, encoding='utf-8') for path in file_paths]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
        chunks = splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        return f'已添加 {len(chunks)} 个文本块'

# 使用封装类
if __name__ == "__main__":
    qa_system = ProductQASystem('./product_knowledge_db')
    print(qa_system.ask('耳机续航多久？'))
```

# RAG实战

## 文件检索系统

- 运行的主文件

```Python
import rag

#rag添加文档
system_prompt = "请根据提供的参考内容，逐一回答用户问题。若参考内容中能找到答案，结合原文和问题给出直接回答。"
kd_rag = rag.ProductQASystem('product_knowledge_db', system_prompt)

print(kd_rag.add_documents(['背影.txt']))
while True:
    print(kd_rag.ask(input(">>> ")))
```

- 可复用rag\.py文件

```Python
import os
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
warnings.filterwarnings('ignore', category=DeprecationWarning)

class ProductQASystem:
    """基于 ChromaDB 的产品手册问答系统。"""

    def __init__(self, persist_dir, system_prompt, llm_model='glm-5.2', embedding_model='Embedding-2'):
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=os.environ.get('OPENAI_BASE_URL'),
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            chunk_size=64,
            max_retries=3,
            request_timeout=60
        )
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.system_prompt = system_prompt
        self.chain = self._build_chain()

    def _format_docs(self, docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ('system', self.system_prompt),
            ('user', '''参考的内容是：\n{context}\n\n问题：{question}''')
        ])
        retriever = self.vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 5, 'fetch_k': 10}
        )
        return (
            {'context': retriever | self._format_docs, 'question': RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        """回答用户问题。"""
        return self.chain.invoke(question)

    def add_documents(self, file_paths):
        """增量添加新文档。"""
        loaders = [TextLoader(path, encoding='utf-8') for path in file_paths]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=['\n\n', '\n', '。', '！', '？', '']
        )
        chunks = splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        return f'已添加 {len(chunks)} 个文本块'

# 使用封装类
if __name__ == "__main__":
    qa_system = ProductQASystem('./product_knowledge_db')
    print(qa_system.ask())
```

# **LangGraph 概述**

## **1\. 为什么需要 LangGraph**



前面我们用 LangChain 构建了链式应用和简单的工具调用智能体。但当智能体流程变得复杂时，会遇到一些问题：

\- **流程不可控**：标准 Agent 的黑盒决策难以调试和约束

\- **循环和分支困难**：难以实现多轮循环、条件分支、人机协作

\- **状态管理复杂**：多步骤之间的状态传递不清晰

\- **缺乏持久化**：无法暂停、恢复或检查执行历史

**LangGraph** 就是为了解决这些问题而设计的。它把智能体流程建模为一个**状态图（StateGraph）**，让开发者精确控制每个步骤。

## **2\. LangGraph 是什么**

LangGraph 是 LangChain 生态中的一个库，用于构建基于**状态图**的智能体应用。它的核心思想是：

- 把应用看作一个图（Graph）

- 图中的每个节点（Node）是一个函数或操作

- 边（Edge）控制流程走向

- 状态（State）在节点之间传递

LangGraph 基于 LangChain 的 Runnable 接口构建，因此可以无缝使用 LangChain 的模型、工具、Prompt 等组件。

## **3\. LangGraph 与 LangChain 的关系**

|对比项|LangChain|LangGraph|
|---|---|---|
|核心抽象|Chain / Agent|StateGraph|
|流程控制|线性或简单循环|任意图结构、条件分支、循环|
|状态管理|隐式传递|显式 State 对象|
|适用场景|简单流水线、单次调用|复杂多智能体、多轮交互|
|关系|基础组件库|基于 LangChain 构建的流程编排层|

可以把 LangChain 理解为「零件库」，LangGraph 理解为「装配线控制系统」。

## **4\. 核心概念**

### **4\.1 State（状态）**

状态是整个图共享的数据结构。每个节点读取状态、修改状态，然后把状态传递给下一个节点。

### **4\.2 StateGraph（状态图）**

状态图是 LangGraph 最核心的类。它负责定义节点、边和状态类型。

### **4\.3 Node（节点）**

节点是图中的执行单元，通常是一个 Python 函数。节点接收当前状态，返回对状态的更新。

### **4\.4 Edge（边）**

边定义节点之间的连接关系。LangGraph 支持：

- 普通边：固定从一个节点到另一个节点

- 条件边：根据状态决定下一个节点

- START / END：图的起点和终点

## **5\. 第一个 LangGraph 程序**

下面构建一个最简单的图：两个节点 `node_a` 和 `node_b`，`node_a` 给状态加 1，`node_b` 再给状态乘 2。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

#1.定义状态类型
class MyState(TypedDict):
    """图中的共享状态"""
    value: int

#2.定义节点函数
def node_a(state: MyState):
    """把value加1"""
    print(f'Node A 接收：{state["value"]}')
    return {'value': state['value'] + 1}

def node_b(state: MyState):
    """"把value乘2"""
    print(f'Node B 接收：{state["value"]}')
    return {'value': state['value'] * 2}

#3.构建状态图
builder = StateGraph(MyState)
builder.add_node('node_a', node_a)
builder.add_node('node_b', node_b)

#4.添加边
builder.add_edge(START, 'node_a')
builder.add_edge('node_a', 'node_b')
builder.add_edge('node_b', END)

#5.编译图
graph = builder.compile()

#6.运行图
result = graph.invoke({'value': 3})
print(f'\n最终结果：{result}')
```

### **代码解释**

- `TypedDict` 定义状态结构，告诉 LangGraph 图中有哪些字段

- `StateGraph(MyState)` 创建状态图实例

- `add_node(name, func)` 添加节点

- `add_edge(START, 'node_a')` 设置从图起点到 node\_a 的边

- `compile()` 编译图，生成可执行的 Runnable

- `invoke({'value': 3})` 运行图，传入初始状态

这个例子的执行流程是：`START -> node_a(value=3 -> 4) -> node_b(value=4 -> 8) -> END`，最终结果是 8。

## **6\. LangGraph 执行流程可视化**

LangGraph 提供了 `get_graph()` 方法，可以查看图的结构。

在线查看Mermaid 图：https://mermaid\.live/edit

```Python
# 打印图的节点和边
print(graph.get_graph().nodes)
print(graph.get_graph().edges)

# 使用 mermaid 格式可视化（在支持 mermaid 的 Markdown 查看器中可显示）
print('\nMermaid 图：')
print(graph.get_graph().draw_mermaid())
#mermaid输出的内容在上面的网站上粘贴，可以看到详细的流程图
```

## **7\. 在 LangGraph 中使用 LangChain 组件**

LangGraph 的节点可以使用任何 LangChain 组件。下面是一个使用 LLM 和 Prompt 的节点示例。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages 
from dotenv import load_dotenv

load_dotenv()

#定义带消息历史的状态
class ChatState(TypedDict):
    messages: list  #对话历史

llm = ChatOpenAI(model='glm-5.2', temperature=0.1)

def chat_node(state: ChatState):
    """调用LLM回复用户"""
    prompt = ChatPromptTemplate.from_messages([
        ('system', '你是友好的AI助手。'),
        *state['messages']
    ])
    response = (prompt | llm).invoke({})
    return {'messages': [response]}

builder = StateGraph(ChatState)
builder.add_node('chat', chat_node)
builder.add_edge(START, 'chat')
builder.add_edge('chat', END)

chat_graph = builder.compile()

result = chat_graph.invoke({
    'messages': [('user', '请用一句话介绍LangGraph')]
})

print(result['messages'][-1].content)

# 使用 mermaid 格式可视化（在支持 mermaid 的 Markdown 查看器中可显示）
print('\nMermaid 图：')
print(chat_graph.get_graph().draw_mermaid())
```

# 图结构

## **1\. 为什么需要图结构**

在前面的课程中，我们已经学过链（Chain）：

```Plain Text
输入 -> Prompt -> 模型 -> 输出
```

这种结构很适合**线性流程**，也就是每一步都按固定顺序往下走。

但是在真实应用中，流程往往没有这么简单。例如：

- 如果用户的问题不完整，要先追问再继续

- 如果模型判断需要调用工具，要进入工具节点

- 如果工具返回的信息还不够，要再循环一次

- 如果满足某个条件，要走 A 路线，否则走 B 路线

这时，单纯的链就不够用了。我们需要一种更灵活的结构，让流程能够：

\- **分支**：根据条件走不同路线

\- **回路**：重复执行直到满足条件

\- **汇合**：多条路径最后回到同一个结果

这就是**图结构（Graph）**的价值。

## **2\. 图结构的核心组成**



在 LangGraph 中，一个图通常由三部分组成：



### **2\.1 State（状态）**

状态可以理解为‘流程执行时共享的一份数据’。



例如一个状态可能长这样：



```Python
{'value': 3, 'result': '通过', 'step': 2}
```



图中的每个节点都可以读取这份状态，也可以返回对状态的更新。



### **2\.2 Node（节点）**

节点就是图中的一个处理步骤。



你可以把节点理解成‘流水线上的一个工位’。每个工位做一件事，例如：



- 给数字加 1

- 判断成绩是否及格

- 调用大模型生成回复

- 调用搜索工具获取资料

### **2\.3 Edge（边）**

边表示‘从哪个节点走到哪个节点’。



边决定了执行顺序。常见有两种：



\- **普通边**：固定走向下一个节点

\- **条件边**：根据状态判断下一步走哪条路



可以把图结构想象成一张地铁路线图：



- 状态 = 你当前身上的信息

- 节点 = 你经过的站点

- 边 = 站点之间的路线

## **3\. 第一个图：顺序执行**

先从最简单的情况开始：

```Plain Text
START -> 加一 -> 乘二 -> END
```

## **4\. 查看图结构**

除了运行图，我们还可以把图的结构打印出来，帮助自己检查流程是否连对了。

```Python
print('节点：')
print(graph.get_graph().nodes)

print('\n边：')
print(graph.get_graph().edges)

print('\nMermaid 图：')
print(graph.get_graph().draw_mermaid())
```

## **5\. 条件分支图**

前面的图是固定顺序执行。现在我们来做一个更像真实业务的例子：**根据分数决定是否通过**。

流程如下：

```Plain Text
START -> 判断分数 -> 合格 / 不合格 -> END
```

如果分数大于等于 60，就走‘通过’节点；否则走‘补考’节点。

这就是典型的**条件分支**。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class ScoreState(TypedDict):
    score: int
    result: str

def check_score(state: ScoreState):
    print(f'当前分数：{state["score"]}')
    return {}

def pass_node(state: ScoreState):
    return {'result': '成绩合格，顺利通过'}

def fail_node(state: ScoreState):
    return {'result': '成绩不合格，需要补考'}

def route_by_score(state: ScoreState):
    if state['score'] >= 60:
        return 'pass_node'
    return 'fail_node'

builder = StateGraph(ScoreState)
builder.add_node('check_score', check_score)
builder.add_node('pass_node', pass_node)
builder.add_node('fail_node', fail_node)

builder.add_edge(START, 'check_score')
builder.add_conditional_edges(
    'check_score',
    route_by_score,
    {
        'pass_node': 'pass_node',
        'fail_node': 'fail_node'
    }
)
builder.add_edge('pass_node', END)
builder.add_edge('fail_node', END)

score_graph = builder.compile()

print('分数 85 的结果：')
print(score_graph.invoke({'score': 85, 'result': ''}))

print('\n分数 45 的结果：')
print(score_graph.invoke({'score': 45, 'result': ''}))
```

### **代码解释**

这个例子第一次出现了**条件边**，所以需要重点掌握。

#### **状态结构**

这里的状态包含两个字段：

- `score`：输入分数

- `result`：最终判断结果

也就是说，这张图的目标是：根据 `score` 计算出 `result`。

**\#\#\#\# \`check\_score\` 节点**

这个节点本身没有修改状态，只是把当前分数打印出来。

它返回 `{}`，表示‘这个节点不更新任何字段’。

这说明一个节点不一定非要做数据修改，它也可以只承担‘中转’或‘判断前准备’的作用。

**\#\#\#\# \`pass\_node\` 和 \`fail\_node\`**

这两个节点分别代表两条不同路径的处理结果：

- 如果通过，更新 `result='成绩合格，顺利通过'`

- 如果不通过，更新 `result='成绩不合格，需要补考'`

这两个节点就像流程图里分叉后的两条支路。

**\#\#\#\# \`route\_by\_score\`：路由函数**

这是本例最核心的部分。

它不是真正的业务节点，而是一个‘路线选择器’。它根据当前状态判断：

- `score >= 60` 返回 `'pass_node'`

- 否则返回 `'fail_node'`

返回值必须和后面条件边映射表中的 key 对应上。

**\#\#\#\# \`add\_conditional\_edges\(\.\.\.\)\`**

这段代码的意思是：

从 `check_score` 节点出来后，不是固定走下一步，而是调用 `route_by_score(state)`。

如果返回 `'pass_node'`，就走到 `pass_node`；如果返回 `'fail_node'`，就走到 `fail_node`。

你可以把它理解成：

- 普通边 = 固定导航

- 条件边 = 智能导航

#### **最终执行效果**

当输入 `score=85` 时，流程是：

```Plain Text
START -> check_score -> pass_node -> END
```

当输入 `score=45` 时，流程是：

```Plain Text
START -> check_score -> fail_node -> END
```

这就是图结构中‘根据条件走不同路线’的最基本模式。

## **6\. 循环图**

除了分支，图结构还有一个非常重要的能力：**循环**。

在很多智能体任务中，我们都需要重复执行某些步骤，比如：

- 检索不到足够信息就继续搜索

- 模型输出格式不对就重新生成

- 工具调用结果不完整就继续补充

下面我们用一个简单例子说明循环：

让 `count` 从 0 开始，每次加 1，直到加到 3 为止。

流程如下：

```Plain Text
START -> increment -> 判断是否结束
                 ↘ 未结束，回到 increment
                 ↘ 已结束，进入 END
```

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class LoopState(TypedDict):
    count: int

def increment(state: LoopState):
    new_count = state['count'] + 1
    print(f'当前 count 从 {state["count"]} 变成 {new_count}')
    return {'count': new_count}

def should_continue(state: LoopState):
    if state['count'] < 3:
        return 'continue'
    return 'stop'

builder = StateGraph(LoopState)
builder.add_node('increment', increment)

builder.add_edge(START, 'increment')
builder.add_conditional_edges(
    'increment',
    should_continue,
    {
        'continue': 'increment',
        'stop': END
    }
)

loop_graph = builder.compile()
result = loop_graph.invoke({'count': 0})
print('最终状态：', result)
```

```Markdown
**### 代码解释**

这个例子看起来简单，但它已经具备了很多 Agent 工作流的核心思想。

**#### `increment` 节点**
它的职责非常明确：每次把 `count` 加 1。

例如：

- 输入 `count=0`，输出 `count=1`
- 输入 `count=1`，输出 `count=2`
- 输入 `count=2`，输出 `count=3`

**#### `should_continue` 路由函数**
它负责判断是继续循环，还是结束流程。

规则很简单：

- 如果 `count < 3`，返回 `'continue'`
- 否则返回 `'stop'`

**#### 条件边的映射**
这里最关键的一点是：

- `'continue'` 映射到 `'increment'`
- `'stop'` 映射到 `END`

这就意味着：

- 只要还没达到条件，就重新回到同一个节点
- 达到条件后，就退出图

这就是循环结构的本质。

**#### 执行过程**
如果初始状态是 `{'count': 0}`，执行路径如下：

1. 进入 `increment`，变成 `count=1`
2. 判断 `1 < 3`，继续
3. 再次进入 `increment`，变成 `count=2`
4. 判断 `2 < 3`，继续
5. 再次进入 `increment`，变成 `count=3`
6. 判断 `3 < 3` 不成立，停止
7. 进入 `END`

也就是说，图结构中的循环并不是 `for` 或 `while` 语法写出来的，而是通过‘节点 + 条件边回跳’实现的。

这也是 LangGraph 很强大的地方：它把‘流程控制’变成了图本身的一部分。
```

## **7\. 图结构和普通 Python 流程控制有什么区别**

你可能会问：

‘这些分支和循环，我直接用 `if`、`while` 不也能写吗？’

当然可以。但图结构有几个明显优势：

|对比点|普通 Python 代码|图结构|
|---|---|---|
|流程表达方式|写在代码逻辑里|显式画成节点和边|
|可视化能力|弱|强|
|分支/回路可读性|复杂时容易乱|更直观|
|与 Agent / 工作流结合|需要手动组织|天然适合|
|状态传递|自己管理变量|统一用 State 管理|

最核心的一点是：

**图结构不是为了替代 Python，而是为了让复杂流程更容易设计、理解、调试和扩展。**

# 状态管理

## **1\. 什么是状态管理**

在 LangGraph 中，**状态（State）** 可以理解为“流程运行时一直随身携带的一份数据”。

前一节我们已经接触过状态，例如：

```Python
{'value': 3}
```

或者：

```Python
{'score': 85, 'result': '成绩合格，顺利通过'}
```

这些数据会在图的不同节点之间传来传去。

状态管理要解决的问题，其实很简单：

- 当前流程已经走到哪一步了？

- 前面节点产出的结果，后面节点怎么接着用？

- 某个节点更新了数据，后续节点如何读取最新值？

你可以把状态想象成一张‘流程记录卡’：

- 每经过一个节点，就在卡上补充信息

- 后面的节点只需要看这张卡，就知道前面发生了什么

所以，**状态管理的本质，就是管理这张共享记录卡。**

## **2\. 为什么状态很重要**

如果没有状态，节点之间就很难协作。

例如一个流程要完成下面几步：

1. 接收用户输入

2. 提取关键词

3. 根据关键词生成结论

那么问题来了：

- 第二步提取出来的关键词，怎么交给第三步？

- 第一节点记录的原始问题，第三步还能不能访问？

- 如果流程还要继续加步骤，数据还能不能统一管理？

答案就是：都放进状态里。

状态的几个核心价值：

\- **统一传递数据**：所有节点都从同一个地方读取信息

\- **减少参数混乱**：不用手动给每个函数传很多单独变量

\- **便于扩展流程**：后面增加新节点时，只要继续读写状态即可

\- **便于调试**：看状态就能知道流程执行到了什么程度

## **3\. 用 TypedDict 定义状态**

在 LangGraph 中，我们通常用 `TypedDict` 来定义状态结构。

这样做有两个好处：

1. 让自己清楚状态里有哪些字段

2. 让代码更容易维护和理解

下面先看一个最简单的例子。

```Python
from typing_extensions import TypedDict

class UserState(TypedDict):
    name: str
    age: int
    city: str

example_state: UserState = {
    'name': '张三',
    'age': 20,
    'city': '北京'
}

print(example_state)
```

### **代码解释**

这段代码本身还没有进入 LangGraph，只是在说明状态应该怎样定义。

**\#\#\#\# \`TypedDict\` 是什么**

`TypedDict` 可以理解为‘带字段说明的字典’。

普通字典也能写成：

```Python
{'name': '张三', 'age': 20, 'city': '北京'}
```

但如果项目变复杂，单纯靠大家记忆“这个字典里应该有哪些字段”会很容易出错。

`TypedDict` 的作用就是把这种约定明确写出来。

**\#\#\#\# \`class UserState\(TypedDict\)\`**

这表示我们定义了一种状态结构，名字叫 `UserState`。

它要求状态中有三个字段：

- `name`：字符串

- `age`：整数

- `city`：字符串

**\#\#\#\# \`example\_state: UserState = \{\.\.\.\}\`**

这表示我们创建了一个符合 `UserState` 结构的状态对象。

从教学角度看，这一步很重要，因为它让你意识到：

- 状态本质上仍然是字典

- `TypedDict` 只是帮助我们把字典结构写清楚

在 LangGraph 中，节点之间传递的就是这种结构化字典。

## **4\. 读取状态：节点如何使用已有数据**

定义完状态之后，下一步就是在节点中读取它。

下面这个例子演示：节点如何从状态里取出数据并生成一句自我介绍。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class ProfileState(TypedDict):
    name: str
    age: int
    city: str
    intro: str 

def create_intro(state: ProfileState):
    intro_text = f'大家好，我叫{state["name"]}，今年{state["age"]}岁，来自{state["city"]}。'
    return {'intro': intro_text}

builder = StateGraph(ProfileState)
builder.add_node('create_intro', create_intro)
builder.add_edge(START, 'create_intro')
builder.add_edge('create_intro', END)

graph = builder.compile()
result = graph.invoke({
    'name': '李雷',
    'age': 18,
    'city': '上海',
    'intro': ''
})

print(result)
print(result['intro'])
```

### **代码解释**

这个例子重点不是图结构本身，而是“节点如何读取状态中的已有字段”。

#### **状态结构**

这里的状态有四个字段：

- `name`

- `age`

- `city`

- `intro`

前三个字段是输入信息，最后一个字段是我们希望在流程中生成的新结果。

**\#\#\#\# \`create\_intro\(state\)\`**

这个节点做的事情很直观：

1. 从状态中读取 `name`

2. 从状态中读取 `age`

3. 从状态中读取 `city`

4. 组装成一句完整介绍

5. 把结果写回 `intro` 字段

也就是说，这个节点没有改变原来的 `name`、`age`、`city`，只是新增或更新了 `intro`。

#### **节点返回值为什么只写\`\{'intro': intro\_text\}\`**

这是 LangGraph 状态管理里非常重要的一点：

**节点只需要返回自己负责修改的字段。**

原来的字段不会凭空消失，而是继续保留在状态里。

所以最终 `result` 中既有原始输入，也有新生成的 `intro`。

#### **这个例子的意义**

它展示了状态管理的最基础工作方式：

- 输入状态中先有一部分信息

- 节点读取这些信息进行加工

- 再把加工结果写回状态

这就是后面所有复杂流程的基础。

## **5\. 更新状态：后面的节点继续接着用**

状态管理真正强大的地方，不在于‘一个节点能读写状态’，而在于**前一个节点更新的内容，后一个节点可以继续使用**。

下面这个例子演示两个节点接力处理状态：

1. 第一个节点生成问候语

2. 第二个节点在问候语后面再补一句欢迎词

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class GreetingState(TypedDict):
    name: str
    greeting: str
    final_message: str

def make_greeting(state: GreetingState):
    return {'greeting': f'你好，{state["name"]}！'}

def make_final_message(state: GreetingState):
    final_text = state['greeting'] + ' 欢迎来到状态管理课程。'
    return {'final_message': final_text}

builder = StateGraph(GreetingState)
builder.add_node('make_greeting', make_greeting)
builder.add_node('make_final_message', make_final_message)

builder.add_edge(START, 'make_greeting')
builder.add_edge('make_greeting', 'make_final_message')
builder.add_edge('make_final_message', END)

graph = builder.compile()
result = graph.invoke({
    'name': '小王',
    'greeting': '',
    'final_message': ''
})

print(result)
print(result['final_message'])
```

### **代码解释**

这个例子体现了状态在多个节点之间的‘接力传递’。

#### **第一个节点\`make\_greeting\`**

它只做一件事：根据 `name` 生成一句问候语，写入 `greeting`。

例如：

```Python
{'name': '小王'}
```

会生成：

```Python
{'greeting': '你好，小王！'}
```

#### **第二个节点\`make\_final\_message\`**

它并不重新根据 `name` 生成问候，而是直接读取上一个节点已经写入状态的 `greeting`。

然后在它后面拼上一句：

```Plain Text
欢迎来到状态管理课程。
```

最后写入 `final_message`。

#### **这说明了什么**

这说明状态不仅能存放“输入数据”，还能存放“中间结果”。

这是状态管理特别重要的一点：

- 输入数据：用户最初给的信息

- 中间结果：某个节点处理出来的阶段性结果

- 最终结果：流程结束时输出的内容

如果没有状态，你就得手动把这些内容一级一级传下去；有了状态，流程自然就串起来了。

## **6\. 多字段状态：让流程更接近真实业务**

真实项目里的状态，通常不会只有 1 个或 2 个字段。

例如一个订单处理流程，可能需要同时记录：

- 用户姓名

- 商品名称

- 数量

- 总价

- 订单状态

下面我们用一个简单的订单示例，演示多字段状态的管理方式。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class OrderState(TypedDict):
    customer_name: str
    product_name: str
    price: float
    quantity: int
    total_price: float
    order_status: str

def calculate_total(state: OrderState):
    total = state['price'] * state['quantity']
    return {'total_price': total}

def confirm_order(state: OrderState):
    status = f'订单已确认：{state["customer_name"]} 购买了 {state["quantity"]} 件 {state["product_name"]}，总价 {state["total_price"]} 元。'
    return {'order_status': status}

builder = StateGraph(OrderState)
builder.add_node('calculate_total', calculate_total)
builder.add_node('confirm_order', confirm_order)

builder.add_edge(START, 'calculate_total')
builder.add_edge('calculate_total', 'confirm_order')
builder.add_edge('confirm_order', END)

graph = builder.compile()
result = graph.invoke({
    'customer_name': '王芳',
    'product_name': '机械键盘',
    'price': 299.0,
    'quantity': 2,
    'total_price': 0.0,
    'order_status': ''
})

print(result)
print(result['order_status'])
```

### **代码解释**

这个例子很适合用来理解‘多字段状态为什么有必要’。

#### **状态里有哪些信息**

这里的状态已经不只是一个简单变量，而是一组完整业务数据：

- 谁在下单：`customer_name`

- 买的是什么：`product_name`

- 单价是多少：`price`

- 买了几件：`quantity`

- 总价是多少：`total_price`

- 当前订单描述：`order_status`

**\#\#\#\# \`calculate\_total\` 节点**

这个节点只负责计算总价：

```Python
总价 = 单价 * 数量
```

它返回 `{'total_price': total}`，把结果写回状态。

**\#\#\#\# \`confirm\_order\` 节点**

这个节点读取多个字段：

- `customer_name`

- `quantity`

- `product_name`

- `total_price`

然后把这些信息拼成完整订单说明，再写入 `order_status`。

#### **这个例子传达的核心思想**

状态不是只给“一个节点”准备的，而是给“整个流程”准备的。

每个节点只关心自己需要的部分字段，但所有字段加起来，构成了完整的业务上下文。

这也是为什么状态设计要尽量清晰：因为它决定了整个流程的数据组织方式。

## **7\. 状态设计的常见原则**

写状态时，建议遵循下面几个原则。

### **7\.1 字段名要清楚**

不要用太模糊的名字，例如：

- `data`

- `info`

- `result1`

更好的方式是：

- `user_question`

- `retrieved_docs`

- `final_answer`

字段名越清楚，后面越不容易乱。

### **7\.2 一个字段只表达一种含义**

例如不要今天把 `result` 用来存字符串，明天又用来存字典。

字段含义最好稳定，不要随流程变化得太厉害。

### **7\.3 把中间结果保留下来**

有些同学会只保留最终结果，把中间结果全部覆盖掉。这样虽然表面简洁，但调试时会非常痛苦。

如果中间结果后面可能还会用到，或者你希望排查流程问题，就应该保留。

### **7\.4 不要把无关信息全塞进去**

状态不是越大越好。

只保留这个流程真正需要的数据，避免状态越来越臃肿。

## **8\. 一个常见误区：以为节点要返回完整状态**

很多初学者会误以为：每个节点都必须返回整份状态，例如：

```Python
return {
    'name': state['name'],
    'age': state['age'],
    'city': state['city'],
    'intro': intro_text
}
```

其实在很多情况下，这么写没有必要。

更简洁的方式通常是：

```Python
return {'intro': intro_text}
```

原因是：LangGraph 会自动把更新结果合并回原状态。

所以，除非你确实要同时改很多字段，否则通常只返回你真正改动的那部分。

这会让代码更短，也更不容易出错。

## **9\. 状态管理和后续 Agent 的关系**

后面学 Agent、RAG、多轮工具调用时，你会越来越频繁地用到状态。

例如一个复杂 Agent 的状态里，可能会放这些字段：

- 用户问题 `user_query`

- 历史消息 `messages`

- 工具调用结果 `tool_result`

- 检索到的资料 `retrieved_docs`

- 当前决策 `decision`

- 最终回答 `final_answer`

所以可以说：

**图结构解决的是“流程怎么走”，状态管理解决的是“数据怎么跟着流程走”。**

这两者是 LangGraph 最核心的两根主线。

# 节点与边

## **1\. 为什么要单独讲节点与边**

在前两节中，我们已经接触过图结构和状态管理。现在要进一步回答一个更基础的问题：

**一张图到底是怎么拼出来的？**

答案就是：

\- 用 **节点（Node）** 表示每一步做什么

\- 用 **边（Edge）** 表示下一步往哪里走

也就是说：

- 节点负责‘干活’

- 边负责‘指路’

你可以把它想象成一张旅游路线图：

- 景点 = 节点

- 景点之间的道路 = 边

- 你游览的先后顺序 = 图的执行顺序

如果没有节点，图里就没有真正执行的步骤；如果没有边，这些步骤之间就连不起来。

## **2\. 节点是什么**

在 LangGraph 中，节点通常就是一个 Python 函数。

这个函数会：

1. 接收当前状态 `state`

2. 处理数据

3. 返回要更新的状态字段

例如一个节点可以做这些事：

- 给数字加 1

- 拼接一段字符串

- 计算总价

- 调用大模型

- 调用工具

所以，节点本质上就是流程中的‘处理步骤’。

一个图里可以有一个节点，也可以有很多节点。节点越多，流程通常越细。

## **3\. 边是什么**

边表示节点和节点之间的连接关系。

如果说节点回答的是：

> 这一步做什么？
> 
> 

那么边回答的就是：

> 做完这一步以后，下一步去哪？
> 
> 

例如：

```Plain Text
START -> node_a -> node_b -> END
```

这里就有三条边：

- `START -> node_a`

- `node_a -> node_b`

- `node_b -> END`

所以，边并不负责处理数据，它只负责规定执行顺序。

# 条件边

## **1\. 为什么需要条件边**

前一节我们学习了普通边。普通边的特点是：**下一步固定不变**。

例如：

```Plain Text
START -> node_a -> node_b -> END
```

这条流程里，不管输入是什么，`node_a` 做完以后都会进入 `node_b`。

但真实业务中，流程往往不是固定直走的，而是会根据情况分支。比如：

- 分数及格，就进入‘通过’节点

- 分数不及格，就进入‘补考’节点

- 用户问题完整，就直接回答

- 用户问题不完整，就先追问

- 检索结果足够，就结束

- 检索结果不够，就继续查

这类‘根据当前状态决定下一步去哪’的场景，就要用到**条件边**。

## **2\. 条件边和普通边的区别**

可以把两种边对比着理解：

|类型|决定方式|典型场景|
|---|---|---|
|普通边 `add_edge`|固定写死|顺序执行流程|
|条件边 `add_conditional_edges`|根据状态动态判断|分支、回路、路由控制|

最核心的区别只有一句话：

- 普通边：下一步是提前写死的

- 条件边：下一步是运行时判断出来的

也就是说，条件边让图开始‘有判断力’。

## **3\. 条件边的基本思路**

使用条件边时，通常有三样东西要配合起来：

1\. **一个普通节点**：先处理当前状态

2\. **一个路由函数**：根据状态判断下一步去哪

3\. **一个分支映射表**：告诉图不同返回值分别对应哪个节点

写法通常像这样：

```Python
builder.add_conditional_edges(
    '某个节点',
    route_function,
    {
        '路径A': 'node_a',
        '路径B': 'node_b'
    }
)
```

它的意思是：

- 先执行‘某个节点’

- 执行完以后，不直接写死下一步

- 而是调用 `route_function(state)` 看返回什么

- 如果返回 `'路径A'`，就走到 `node_a`

- 如果返回 `'路径B'`，就走到 `node_b`

## **4\. 第一个例子：根据分数决定是否通过**

先从最直观的例子开始。

流程目标：

- 如果分数大于等于 60，进入通过节点

- 如果分数小于 60，进入补考节点

流程图如下：

```Plain Text
START -> check_score -> pass_node / fail_node -> END
```

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class ScoreState(TypedDict):
    score: int 
    result: str

def check_score(state: ScoreState):
    print(f'当前分数：{state["score"]}')
    return {}

def pass_node(state: ScoreState):
    return{'result': '成绩合格，顺利通过'}

def fail_node(state: ScoreState):
    return{'result': '成绩不合格，需要补考'}

def route_by_score(state: ScoreState):
    if state['score'] >= 60:
        return 'pass'
    return 'fail'

builder = StateGraph(ScoreState)
builder.add_node('check_score', check_score)
builder.add_node('pass_node', pass_node)
builder.add_node('fail_node', fail_node)

builder.add_edge(START, 'check_score')
builder.add_conditional_edges(
    'check_score',
    route_by_score,
    {
        'pass': 'pass_node',
        'fail': 'fail_node'
    }
)

builder.add_edge('pass_node', END)
builder.add_edge('fail_node', END)

graph = builder.compile()

print('分数85：')
print(graph.invoke({'score': 85, 'result': ''}))
print('\n分数45：')
print(graph.invoke({'score': 45, 'result': ''}))
```

### **代码解释**

这个例子非常典型，几乎把条件边最核心的要点都展示出来了。

#### **状态结构**

状态里有两个字段：

- `score`：输入分数

- `result`：最终判断结果

**\#\#\#\# \`check\_score\` 节点**

这个节点本身没有修改状态，只是打印当前分数。

它返回 `{}`，表示‘这个节点不更新任何字段’。

这说明一个节点不一定非要负责生成结果，它也可以只是一个‘分支前的准备步骤’。

**\#\#\#\# \`pass\_node\` 和 \`fail\_node\`**

这两个节点分别代表分支后的两条路径：

- 通过路径：写入‘成绩合格，顺利通过’

- 失败路径：写入‘成绩不合格，需要补考’

**\#\#\#\# \`route\_by\_score\(state\)\`**

这是本例的路由函数。它的任务不是修改状态，而是‘判断下一步去哪’。

判断逻辑很简单：

- 如果 `score >= 60`，返回 `'pass'`

- 否则返回 `'fail'`

注意：这里返回的不是节点函数本身，而是一个**分支标记**。

**\#\#\#\# \`add\_conditional\_edges\(\.\.\.\)\`**

这一段可以拆成三层意思：

1. 从 `check_score` 节点出来以后，不直接写死下一步

2. 调用 `route_by_score(state)` 得到一个返回值

3. 根据映射表决定真正跳去哪个节点

也就是说：

- 返回 `'pass'` \-\> 去 `pass_node`

- 返回 `'fail'` \-\> 去 `fail_node`

#### **执行路径是怎么变化的**

当输入 `score=85` 时：

```Plain Text
START -> check_score -> pass_node -> END
```

当输入 `score=45` 时

```Plain Text
START -> check_score -> fail_node -> END
```

这就是条件边最基础的价值：**同一张图，面对不同状态时，可以自动走不同路线。**

## **5\. 第二个例子：问题完整就直接回答，不完整就先追问**

下面我们做一个更贴近 AI 助手的例子。

假设用户提问时，有的问题已经很清楚，有的问题信息还不够。

流程目标：

- 如果问题完整，直接进入回答节点

- 如果问题不完整，先进入追问节点

这个例子可以帮助你理解：条件边不仅能处理分数、数字，也能处理业务判断。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class QuestionState(TypedDict):
    user_question: str
    is_complete: bool
    response: str

def analyze_question(state: QuestionState):
    print(f'收到问题：{state["user_question"]}')
    return {}

def ask_more(state: QuestionState):
    return {'response': '你的问题还不够完整，请补充更多背景信息。'}

def answer_directly(state: QuestionState):
    return {'response': '问题信息已经足够，我现在直接为你回答。'}

def route_question(state: QuestionState):
    if state['is_complete']:
        return 'answer'
    return 'ask_more'

builder = StateGraph(QuestionState)
builder.add_node('analyze_question', analyze_question)
builder.add_node('ask_more', ask_more)
builder.add_node('answer_directly', answer_directly)

builder.add_edge(START, 'analyze_question')
builder.add_conditional_edges(
    'analyze_question',
    route_question,
    {
        'answer': 'answer_directly',
        'ask_more': 'ask_more'
    }
)
builder.add_edge('ask_more', END)
builder.add_edge('answer_directly', END)

graph = builder.compile()

print('完整问题：')
print(graph.invoke({
    'user_question': '请解释什么是条件边，并举一个例子',
    'is_complete': True,
    'response': ''
}))

print('\n不完整问题：')
print(graph.invoke({
    'user_question': '帮我看看这个',
    'is_complete': False,
    'response': ''
}))
```

### **代码解释**

这个例子和前面的‘分数判断’相比，更接近真实 AI 产品中的流程控制。

#### **状态字段的意义**

这里的状态有三个字段：

- `user_question`：用户输入的问题

- `is_complete`：问题是否完整

- `response`：最终回复

**\#\#\#\# \`analyze\_question\` 节点**

这个节点先接收问题并打印出来，相当于‘进入分析阶段’。

它没有直接修改状态，而是把路由判断交给后面的路由函数。

**\#\#\#\# \`route\_question\(state\)\`**

这个函数根据 `is_complete` 的真假做判断：

- `True` \-\> 返回 `'answer'`

- `False` \-\> 返回 `'ask_more'`

这意味着流程下一步是动态的，而不是提前写死的。

#### **两条分支的业务含义**

- `answer_directly`：说明信息足够，可以直接处理

- `ask_more`：说明信息不足，需要先追问

#### **为什么这个例子很重要**

它让你看到：条件边并不只是‘程序里的 if 判断’，更像是业务流程里的‘路由器’。

很多 Agent 系统都会有类似逻辑：

- 信息够不够？

- 要不要调用工具？

- 要不要继续搜索？

- 能不能直接输出答案？

而这些判断，往往都可以通过条件边表达出来。

## **6\. 第三个例子：条件边也可以用于继续或结束**

条件边不仅可以做左右分支，还经常用来做‘继续/停止’的判断。

这类场景非常常见，例如：

- 检索结果不够，就继续检索

- 输出格式不对，就继续生成

- 数量还没到要求，就继续累加

下面用一个简单例子说明：如果 `count < 3`，就继续执行；否则结束。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class LoopState(TypedDict):
    count: int

def add_one(state: LoopState):
    new_count = state['count'] + 1
    print(f'当前 count 从 {state["count"]} 变成 {new_count}')
    return {'count': new_count}

def route_loop(state: LoopState):
    if state['count'] < 3:
        return 'continue'
    return 'stop'

builder = StateGraph(LoopState)
builder.add_node('add_one', add_one)

builder.add_edge(START, 'add_one')
builder.add_conditional_edges(
    'add_one',
    route_loop,
    {
        'continue': 'add_one',
        'stop': END
    }
)

graph = builder.compile()
result = graph.invoke({'count': 0})
print('最终结果：', result)
```

### **代码解释**

这个例子非常重要，因为它说明：**条件边不仅能分叉，也能回跳。**

**\#\#\#\# \`add\_one\` 节点**

这个节点每次把 `count` 加 1。

也就是说，它是一个不断重复执行的工作步骤。

**\#\#\#\# \`route\_loop\(state\)\`**

这个路由函数负责判断接下来是继续，还是停止：

- 如果 `count < 3`，返回 `'continue'`

- 否则返回 `'stop'`

#### **条件边映射**

这里映射关系很关键：

- `'continue'` \-\> `'add_one'`

- `'stop'` \-\> `END`

也就是说：

- 还没满足条件，就重新回到原节点

- 满足条件后，就离开流程

#### **执行过程**

如果初始值是 `count=0`，路径会是：

1. `0 -> 1`，继续

2. `1 -> 2`，继续

3. `2 -> 3`，停止

4. 进入 `END`

从这个例子你可以看到，条件边不只是“二选一分支”，它还是循环流程的基础。

后面很多 Agent 的‘反复尝试直到成功’流程，本质上都是这种写法。

## **7\. 路由函数应该怎么理解**

很多初学者最容易混淆的地方，就是把‘节点函数’和‘路由函数’混在一起。

你可以这样区分：

|类型|主要职责|是否负责业务处理|
|---|---|---|
|节点函数|读写状态、执行处理逻辑|是|
|路由函数|判断下一步走哪条路|通常不是重点|

换句话说：

- 节点函数更像‘工人’

- 路由函数更像‘调度员’

工人负责真正干活，调度员负责决定下一站去哪。

理解这一点以后，条件边的结构就会非常清晰。

## **8\. 使用条件边时的常见错误**

### **8\.1 路由函数返回值和映射表对不上**

例如路由函数返回 `'pass'`，但映射表里只有 `'ok'` 和 `'fail'`。

这样图就不知道该往哪里走。

### **8\.2 把节点名字和路由标记混为一谈**

路由函数返回的可以是一个标记，例如 `'continue'`、`'stop'`、`'pass'`。

这些标记不一定要和节点名完全相同，但必须和映射表中的 key 对应。

### **8\.3 以为条件边只能做两条分支**

其实不止两条。

你完全可以写三条、四条甚至更多路线，只要业务场景需要。

### **8\.4 忘记给分支节点连到\`END\` 或后续节点**

条件边只负责从当前节点分出去，不代表后面的路径已经自动补全。

分支出去以后，该怎么结束、怎么汇合，仍然要继续写边。

## **9\. 条件边在 Agent 场景中的意义**

到了这里，你可以把条件边理解成 LangGraph 里的‘流程决策器’。

在 Agent 场景中，它特别常见，因为 Agent 天生就不是固定直线流程。

例如：

- 如果用户问题明确，直接回答

- 如果用户问题不明确，先追问

- 如果需要外部信息，调用工具

- 如果工具结果不够，再查一次

- 如果结果足够，就进入最终回答

你会发现，这些逻辑本质上都属于：

**根据当前状态，动态决定下一步去哪。**

这正是条件边最核心的价值。

# 循环与记忆

## **1\. 为什么需要循环和记忆**

在之前的课程中，我们构建的图流程都是**单次执行**的：

```Plain Text
START -> node_a -> node_b -> END
```

这种流程跑一次就结束了。但真实场景中，很多需求需要**多次循环**和**记忆状态**：

\- **聊天机器人**：需要记住之前说过的话，才能进行多轮对话

\- **任务分解**：可能需要反复调用工具，直到任务完成

\- **信息收集**：需要逐步收集用户的信息，直到完整

\- **多用户系统**：每个用户的对话历史需要独立保存

简单来说：

\- **循环**：让流程可以重复执行某些节点

\- **记忆**：让状态可以在多次执行之间保持

## **2\. 循环的实现方式**

在LangGraph中，循环主要通过**条件边**来实现。关键思路是：

1. 执行某个节点

2. 判断是否需要继续循环

3. 如果需要，回到之前的节点重新执行

4. 如果不需要，走向结束

流程图如下：

```Plain Text
START -> work_node -> 判断节点
                        |
              ┌─────────┴─────────┐
              ▼                   ▼
           继续循环             结束
              |                   |
              └───────► work_node ◄──┘
```

用条件边实现就是：`判断节点` 根据状态返回 `'loop'` 或 `'end'`，然后：

- `'loop'` → 回到 `work_node`

- `'end'` → 走到 `END`

## **3\. 第一个例子：简单的计数器循环**

我们用一个计数器来演示循环的基本实现。

流程目标：

- 从0开始计数

- 每次加1

- 当计数达到5时停止

- 每次循环都打印当前计数值

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class CounterState(TypedDict):
    count: int 

def increment_node(state: CounterState):
    new_count = state['count'] + 1
    print(f'当前计数：{new_count}')
    return {'count': new_count}

def should_continue(state: CounterState):
    if state['count'] < 5:
        return 'loop'
    return 'end'

builder = StateGraph(CounterState)
builder.add_node('increment', increment_node)

builder.add_edge(START, 'increment')
builder.add_conditional_edges(
    'increment',
    should_continue,
    {
        'loop': 'increment',
        'end': END
    }
)

graph = builder.compile()
result = graph.invoke({'count': 0})
print(result)
print(graph.get_graph().draw_mermaid())
```

### **代码解释**

1\. **State 定义**：\`CounterState\` 只包含一个字段 \`count\`，用于存储当前计数值。

2\. **increment\_node**：每次执行时，从状态中取出 \`count\`，加1后放回状态。同时打印当前计数值。

3\. **should\_continue**：路由函数，判断是否继续循环。如果 \`count \< 5\` 返回 \`'loop'\`，否则返回 \`'end'\`。

4\. **条件边设置**：

- 来源节点是 `'increment'`

- 路由函数是 `should_continue`

- 如果返回 `'loop'`，就回到 `'increment'` 节点（形成循环）

- 如果返回 `'end'`，就走到 `END`

5\. **执行**：初始状态是 \`\{'count': 0\}\`，图会自动循环执行，直到 \`count\` 达到5。

运行结果显示计数从1到5，说明循环成功执行了5次。

## **4\. 状态累积：消息历史的保存**

循环的核心价值在于**状态累积**。最常见的场景是保存对话历史。

在聊天机器人中，每次用户输入和AI回复都需要保存下来，这样AI才能理解上下文。

我们来构建一个简单的对话系统，每次循环都会把消息追加到历史中。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class ChatState(TypedDict):
    messages: list 
    current_user_input: str

def process_message(state: ChatState):
    user_input = state['current_user_input']

    responses = {
        '你好': '你好！我是一个AI助手。',
        '我叫张三': '你好张三！很高兴认识你。',
        '我今天心情很好': '太好了！祝你有美好的一天！'
    }

    assistant_reply = responses.get(user_input, '抱歉，我不太理解。')

    new_message = {'user': user_input, 'assistant': assistant_reply}

    return {
        'messages': state['messages'] + [new_message],
        'current_user_input': ''
    }

def route(state: ChatState):
    if state['current_user_input']:
        return 'process'
    return END

builder = StateGraph(ChatState)
builder.add_node('process', process_message)

builder.add_edge(START, 'process')
builder.add_conditional_edges(
    'process',
    route,
    {
        'process': 'process',
        END: END
    }
)

graph = builder.compile()

state = {'messages': [], 'current_user_input': '你好'}
state = graph.invoke(state)

state['current_user_input'] = '我叫张三'
state = graph.invoke(state)

state['current_user_input'] = '我今天心情很好'
state = graph.invoke(state)

print(state['messages'])
```

### **代码解释**

1\. **State 定义**：\`ChatState\` 包含两个字段：

- `messages`：保存所有对话历史的列表

- `current_user_input`：当前用户输入

2\. **process\_message**：处理消息的核心逻辑：

- 从状态中获取当前用户输入

- 根据预设的规则生成助手回复

- 将新消息追加到 `messages` 列表中（关键！）

- 清空 `current_user_input`

3\. **route**：路由函数判断是否有新输入需要处理。

4\. **执行流程**：

- 第一次调用：输入'你好'，消息列表变成 `[{'user': '你好', ...}]`

- 第二次调用：输入'我叫张三'，消息列表变成两条记录

- 第三次调用：输入'我今天心情很好'，消息列表变成三条记录

运行结果显示消息历史被正确累积，这就是状态持久化的基础。

## 去除波浪线

- 编写json文件

```MATLAB
{
    "typeCheckkingMode": "basic",
    "reportUnknownMemberType": false
}
```

## 状态累计:携带历史记忆示例

```Python
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

_ = load_dotenv()

llm = ChatOpenAI(
    model='glm-5.2',
    temperature=0.7,
    openai_api_base=os.environ.get('OPENAI_BASE_URL'),
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

class ChatState(TypedDict):
    messages: list
    current_user_input: str

def ask_user(state: ChatState):
    # 获取用户输入
    user_input = input('>>>')
    return {'current_user_input': user_input}

def process_message(state: ChatState):
    user_input = state['current_user_input']

    # 构建对话历史，交给模型生成回复
    history = []
    for msg in state['messages']:
        history.append(('user', msg['user']))
        history.append(('assistant', msg['assistant']))
    history.append(('user', user_input))

    assistant_reply = llm.invoke(history).content
    print('AI：', assistant_reply)

    new_message = {'user': user_input, 'assistant': assistant_reply}

    return {
        'messages': state['messages'] + [new_message],
    }

def route(state: ChatState):
    # 由模型判断用户是否表达了再见、结束对话的意思
    decision = llm.invoke([
        ('system', '判断用户的输入是否表达了结束对话、再见、告辞的意思。你只需要回答 yes 或 no，不要输出其他任何内容。'),
        ('user', state['current_user_input'])
    ]).content.strip().lower()

    if decision.startswith('yes'):
        return END
    return 'process'

builder = StateGraph(ChatState)
builder.add_node('ask_user', ask_user)
builder.add_node('process', process_message)

builder.add_edge(START, 'ask_user')
builder.add_conditional_edges(
    'ask_user',
    route,
    {
        'process': 'process',
        END: END
    }
)
# 回答完继续等待下一次用户输入，形成图内循环
builder.add_edge('process', 'ask_user')

graph = builder.compile()

# 整个问答循环在图内部完成，route 判断退出后图执行结束
final_state = graph.invoke({'messages': [], 'current_user_input': ''})
print('对话已结束，共 %d 轮。' % len(final_state['messages']))

```

## **5\. MemorySaver：持久化状态存储**

上面的例子中，我们手动传递状态。但在真实应用中，状态需要：

- 在多次请求之间保持

- 支持多个用户同时使用（会话隔离）

- 重启后不丢失

LangGraph 提供了 `MemorySaver` 来解决这个问题。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    messages: list 
    current_user_input: str 

def process_messages(state: ChatState):
    user_input = state['current_user_input']

    responses = {
        '你好': '你好！我是一个AI助手',
        '我叫张三': '你好张三！很高兴认识你。',
        '我今天心情很好': '太好了！祝你有美好的一天！'
    }
    assistant_reply = responses.get(user_input, '抱歉，我不太理解')

    new_message = {'user': user_input, 'assistant': assistant_reply}

    return {
        'messages': state['messages'] + [new_message],
        'current_user_input': ''
    }

def route(state: ChatState):
    if state['current_user_input']:
        return 'process'
    return END

builder = StateGraph(ChatState)
builder.add_node('process', process_messages)

builder.add_edge(START, 'process')
builder.add_conditional_edges(
    'process',
    route,
    {
        'process': 'process',
        END: END
    }
)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id': 'session_1'}}

graph.invoke({'messages': [], 'current_user_input': '你好'}, config)
graph.invoke({'current_user_input': '我叫张三'}, config)
graph.invoke({'current_user_input': '我今天心情很好'}, config)

final_state = graph.get_state(config)
print(final_state.values['messages'])
```

### **代码解释**

1\. **导入 MemorySaver**：从 \`langgraph\.checkpoint\.memory\` 导入。

2\. **创建 MemorySaver 实例**：\`memory = MemorySaver\(\)\`

3\. **编译时传入 checkpointer**：\`graph = builder\.compile\(checkpointer=memory\)\`

4\. **配置会话ID**：

```Python
config = {'configurable': {'thread_id': 'session_1'}}
```

`thread_id` 是会话的唯一标识，不同用户用不同的 `thread_id`。

5\. **多次调用**：每次调用只传入新的输入，不需要手动传递状态。

6\. **获取状态**：\`graph\.get\_state\(config\)\` 可以获取指定会话的当前状态。

关键区别：之前需要手动传递 `state`，现在只需传入 `config`，MemorySaver 会自动保存和恢复状态。

## **6\. 多会话隔离**

MemorySaver 的一个重要特性是**会话隔离**。不同的 \`thread\_id\` 会维护独立的状态。

下面的例子演示两个用户同时使用同一个图，但各自的对话历史完全独立。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    messages: list
    current_user_input: str

def process_message(state: ChatState):
    user_input = state['current_user_input']
    
    responses = {
        '你好': '你好！我是一个AI助手。',
        '我是Alice': '你好Alice！很高兴认识你。',
        '我是Bob': '你好Bob！很高兴认识你。'
    }
    
    assistant_reply = responses.get(user_input, '抱歉，我不太理解。')

    new_message = {'user': user_input, 'assistant': assistant_reply}
    
    return {
        'messages': state['messages'] + [new_message],
        'current_user_input': ''
    }

def route(state: ChatState):
    if state['current_user_input']:
        return 'process'
    return END

builder = StateGraph(ChatState)
builder.add_node('process', process_message)

builder.add_edge(START, 'process')
builder.add_conditional_edges(
    'process',
    route,
    {
        'process': 'process',
        END: END
    }
)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config_a = {'configurable': {'thread_id': 'user_a'}}
config_b = {'configurable': {'thread_id': 'user_b'}}

graph.invoke({'messages': [], 'current_user_input': '你好'}, config_a)
graph.invoke({'current_user_input': '我是Alice'}, config_a)

graph.invoke({'messages': [], 'current_user_input': 'Hi'}, config_b)
graph.invoke({'current_user_input': '我是Bob'}, config_b)

state_a = graph.get_state(config_a)
state_b = graph.get_state(config_b)

print('用户A的对话历史:')
print(state_a.values['messages'])
print()
print('用户B的对话历史:')
print(state_b.values['messages'])
```

### **代码解释**

1\. **创建两个配置**：

- `config_a` 使用 `thread_id: 'user_a'`

- `config_b` 使用 `thread_id: 'user_b'`

2\. **分别调用**：

- 用户A发送了'你好'和'我是Alice'

- 用户B发送了'Hi'和'我是Bob'

3\. **获取各自状态**：

- 用户A的消息历史只有自己的两条消息

- 用户B的消息历史只有自己的两条消息

运行结果清楚地展示了会话隔离：两个用户的对话历史完全独立，互不干扰。

这就是多用户聊天系统的核心原理。

## **8\. 综合案例：带工具调用的循环智能体**

现在我们把循环、记忆和工具调用结合起来，构建一个完整的智能体。

流程目标：

- 用户询问关于天气或时间的问题

- 智能体判断是否需要调用工具

- 如果需要，调用工具获取信息

- 如果不需要，直接回答

- 循环直到任务完成

- 使用MemorySaver保存对话历史

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

class AgentState(TypedDict):
    messages: list
    next_tool: str

def decide_tool(state: AgentState):
    last_message = state['messages'][-1]
    user_input = last_message.content
    
    print(f'用户问了：{user_input}')
    
    if '几点' in user_input or '时间' in user_input:
        print('需要调用工具：get_time')
        return {'next_tool': 'get_time'}
    elif '天气' in user_input:
        print('需要调用工具：get_weather')
        return {'next_tool': 'get_weather'}
    else:
        print('不需要调用工具，直接回答')
        return {'next_tool': None}

def call_tool(state: AgentState):
    tool_name = state['next_tool']
    
    if tool_name == 'get_time':
        result = '2024-01-15 14:30:00'
    elif tool_name == 'get_weather':
        result = '晴天，25度'
    else:
        result = '未知工具'
    
    print(f'调用工具 {tool_name}，结果：{result}')
    
    return {
        'messages': state['messages'] + [ToolMessage(content=result, tool_call_id='1')],
        'next_tool': None
    }

def summarize(state: AgentState):
    last_message = state['messages'][-1]
    
    if isinstance(last_message, ToolMessage):
        tool_result = last_message.content
        reply = f'总结回答：{tool_result}'
    else:
        reply = '抱歉，我不太理解你的问题。'
    
    print(reply)
    
    return {
        'messages': state['messages'] + [AIMessage(content=reply)]
    }

def route_after_decide(state: AgentState):
    if state['next_tool']:
        return 'call_tool'
    return 'summarize'

def route_after_summarize(state: AgentState):
    last_message = state['messages'][-1]
    if isinstance(last_message, HumanMessage):
        return 'decide_tool'
    return END

builder = StateGraph(AgentState)
builder.add_node('decide_tool', decide_tool)
builder.add_node('call_tool', call_tool)
builder.add_node('summarize', summarize)

builder.add_edge(START, 'decide_tool')
builder.add_conditional_edges(
    'decide_tool',
    route_after_decide,
    {
        'call_tool': 'call_tool',
        'summarize': 'summarize'
    }
)
builder.add_edge('call_tool', 'summarize')
builder.add_conditional_edges(
    'summarize',
    route_after_summarize,
    {
        'decide_tool': 'decide_tool',
        END: END
    }
)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id': 'tool_agent_session'}}

graph.invoke({'messages': [HumanMessage(content='现在几点了')], 'next_tool': None}, config)

current_state = graph.get_state(config)
new_messages = current_state.values['messages'] + [HumanMessage(content='今天天气怎么样')]
graph.invoke({'messages': new_messages, 'next_tool': None}, config)

final_state = graph.get_state(config)
print()
print('完整对话历史:')
for msg in final_state.values['messages']:
    if isinstance(msg, HumanMessage):
        print(f'Human: {msg.content}')
    elif isinstance(msg, AIMessage):
        print(f'AI: {msg.content}')
```

### **代码解释**



这个案例展示了一个完整的智能体流程：



1\. **AgentState**：使用 TypedDict 定义状态，包含：

- `messages`：消息列表

- `next_tool`：下一个要调用的工具

2\. **decide\_tool**：判断是否需要调用工具

- 如果用户问时间，返回 `{'next_tool': 'get_time'}`

- 如果用户问天气，返回 `{'next_tool': 'get_weather'}`

- 如果不需要工具，返回 `{'next_tool': None}`

3\. **call\_tool**：执行工具调用

- 根据 `next_tool` 调用相应工具

- 将结果追加到 `messages` 列表

4\. **summarize**：总结回答

- 如果最后是工具消息，提取结果并生成回答

- 将回答追加到 `messages` 列表

5\. **route\_after\_decide**：decide\_tool 之后的路由

- 如果 `next_tool` 有值，进入 `call_tool`

- 如果 `next_tool` 为 None，进入 `summarize`

6\. **route\_after\_summarize**：summarize 之后的路由

- 如果最后是用户消息，回到 `decide_tool`

- 否则结束

流程图：

```Plain Text
START -> decide_tool -> [条件边] -> call_tool -> summarize
                                    ^              |
                                    |              |
                                    +--------------+
```

这个流程可以处理多轮对话，每轮都可以调用工具并保存历史。

# ai coding

1. 登录trae，提供\.env文件，创建并打开一个项目文件夹

2. 在trae中输入

```MATLAB
/spec 我要开发一个基于知识库的RAG系统，可以使用我自定义的大语言模型，以及向量模型，支持解析常见文档格式

/spec 使用python的flask作为web框架，使用sqlite作为数据库，使用ChromaDB作为向量数据库，测试使用的两个大模型信息在.env文件中，模型名自己去读取模型列表，使用原生html前端，需要支持流式输出，pip安装使用清华源，使用虚拟环境

你是一个产品经理兼项目管理，使用多智能体组建teams，完成项目开发与验收，开搞
```



