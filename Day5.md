# Day5

# 人机协作

## **1\. 什么是人机协作**

人机协作，也叫 Human\-in\-the\-loop，意思是：**智能体不是从头到尾完全自动执行，而是在关键步骤暂停下来，让人参与判断、确认或补充信息。**

普通自动流程可能是这样：

```Plain Text
输入 -> 智能体处理 -> 自动执行 -> 输出结果
```

人机协作流程更像这样：

```Plain Text
输入 -> 智能体处理 -> 暂停等待人工确认 -> 继续执行 -> 输出结果
```

也就是说，人不是在流程外面旁观，而是成为流程中的一个关键环节。

在人机协作中，AI 适合做：

- 整理信息

- 生成草稿

- 给出建议

- 自动处理低风险步骤

人更适合做：

- 最终确认

- 风险判断

- 补充缺失信息

- 决定是否继续执行

这就是人机协作的核心思想：**让 AI 提高效率，让人把控关键决策。**

## **2\. 为什么智能体需要人工介入**

很多智能体任务不能完全交给模型自动完成。原因主要有三类。

### **2\.1 有些操作风险较高**

例如：

- 发送正式邮件

- 删除数据

- 提交订单

- 执行付款

- 修改线上配置

这些操作一旦执行，可能造成真实影响，所以最好先让人确认。

### **2\.2 有些信息 AI 不知道**

例如用户说：

```Plain Text
帮我给客户写一封邮件。
```

但 AI 可能不知道客户姓名、邮件目的、语气要求、是否要附带报价。

这时流程应该暂停，让用户补充信息。

### **2\.3 有些判断需要人负责**

例如 AI 生成了一个方案，但最终是否采用，需要人做业务判断。

所以，人机协作不是因为 AI 不够强，而是因为很多场景天然需要人工负责。

## **3\. LangGraph 中的人机协作机制**

在 LangGraph 中，人机协作通常依赖两个核心动作：

|动作|含义|作用|
|---|---|---|
|`interrupt(...)`|暂停流程|把需要人工处理的信息抛出来|
|`Command(resume=...)`|恢复流程|把人工输入传回图中，继续执行|

可以这样理解：

- `interrupt` 像是智能体举手说：‘这里需要人来决定一下’

- `Command(resume=...)` 像是人回复：‘我决定好了，你继续吧’

不过要注意：暂停和恢复必须依赖检查点（checkpoint）。

因为图暂停后，需要记住自己停在了哪里。等人给出答案后，才能从正确的位置继续执行。

所以，人机协作一般会和 `MemorySaver` 一起使用。

## **4\. 第一个例子：人工确认后再继续**

先从最简单的例子开始：智能体准备执行一个操作，但在执行前需要人工确认。

流程如下：

```Plain Text
START -> request_approval -> perform_action -> END
```

其中 `request_approval` 会暂停，等待用户输入 `yes` 或 `no`。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    task: str
    approved: bool
    result: str

def request_approval(state: ApprovalState):
    answer = interrupt({
        'question': '是否允许执行这个任务？',
        'task': state['task'],
        'options': ['yes', 'no']
    })
    return {'approved': answer == 'yes'}

def perform_action(state: ApprovalState):
    if state['approved']:
        return {'result': f'任务已执行：{state["task"]}'}
    return {'result': f'任务已取消：{state["task"]}'}

builder = StateGraph(ApprovalState)
builder.add_node('request_approval', request_approval)
builder.add_node('perform_action', perform_action)

builder.add_edge(START, 'request_approval')
builder.add_edge('request_approval', 'perform_action')
builder.add_edge('perform_action', END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id': 'approval-demo'}}

first_result = graph.invoke(
    {'task': '发送项目周报邮件', 'approved': False, 'result': ''},
    config=config
)

print(first_result)
```

### **代码解释**

这个例子第一次使用了 `interrupt`，所以需要重点理解。

**\#\#\#\# \`ApprovalState\`**

状态里有三个字段：

- `task`：准备执行的任务

- `approved`：人工是否批准

- `result`：最终执行结果

也就是说，整张图围绕一个任务展开，先确认，再执行或取消。

**\#\#\#\# \`request\_approval\` 节点**

这个节点的核心代码是：

```Python
answer = interrupt({...})
```

当代码执行到这里时，图不会继续往下走，而是暂停下来。

`interrupt` 里面传入的是要展示给人的信息。这里包括：

- 问题：是否允许执行这个任务

- 任务内容：`state['task']`

- 可选项：`yes` 或 `no`

#### **为什么\`interrupt\` 后面还有 return**

这行代码：

```Python
return {'approved': answer == 'yes'}
```

不是第一次暂停时立刻执行完成的。

第一次运行到 `interrupt` 时，图会停住。等我们用 `Command(resume='yes')` 恢复时，`interrupt` 才会把 `'yes'` 作为返回值交给 `answer`，然后继续执行后面的 `return`。

**\#\#\#\# \`perform\_action\` 节点**

这个节点根据 `approved` 决定结果：

- 如果 `approved=True`，说明人工批准，任务执行

- 如果 `approved=False`，说明人工拒绝，任务取消

#### **为什么需要\`MemorySaver\`**

因为图会暂停。暂停之后，系统必须记住：

- 当前执行到哪个节点

- 当前状态是什么

- 后面应该从哪里继续

这些信息都需要检查点保存。`MemorySaver` 就是一个内存版检查点保存器。

#### **第一次运行会得到什么**

第一次 `invoke` 不会直接得到最终结果，而是会得到一个包含中断信息的结果。

这表示图已经暂停，正在等待人工输入。

## **5\. 恢复流程：用\`Command\(resume=\.\.\.\)\` 传回人工决定**

上一步图已经暂停。现在我们模拟人工选择 `yes`，让流程继续执行。

恢复时要使用同一个 `config`，尤其是同一个 `thread_id`。

因为 LangGraph 需要根据 `thread_id` 找回刚才暂停的那一次运行。

```MATLAB
second_result = graph.invoke(
    Command(resume='yes'),
    config=config
)

print(second_result)
```

### **代码解释**

这一段代码负责恢复刚才暂停的流程。

**\#\#\#\# \`Command\(resume='yes'\)\`**

它的意思是：把人工输入 `'yes'` 传回给刚才的 `interrupt`。

也就是说，前面这行代码：

```Python
answer = interrupt(...)
```

会在恢复后得到：

```Python
answer = 'yes'
```

然后节点继续执行：

```Python
return {'approved': answer == 'yes'}
```

于是 `approved` 会变成 `True`。

#### **为什么还要传\`config=config\`**

因为恢复流程必须知道要恢复哪一次暂停。

`thread_id='approval-demo'` 就像这次流程的编号。

如果换了另一个 `thread_id`，LangGraph 就找不到刚才暂停的位置。

#### **恢复后的执行路径**

恢复后流程会继续：

```Plain Text
request_approval -> perform_action -> END
```

最终会得到：

```Plain Text
任务已执行：发送项目周报邮件
```

这就是最基本的人机协作流程。

## 4与5的结合

- 手动确认

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    task: str
    approved: bool
    result: str

def request_approval(state: ApprovalState):
    answer = interrupt({
        'question': '是否允许执行这个任务？',
        'task': state['task'],
        'options': ['yes', 'no']
    })
    return {'approved': answer == 'yes'}

def perform_action(state: ApprovalState):
    if state['approved']:
        return {'result': f'任务已执行：{state["task"]}'}
    return {'result': f'任务已取消：{state["task"]}'}

builder = StateGraph(ApprovalState)
builder.add_node('request_approval', request_approval)
builder.add_node('perform_action', perform_action)

builder.add_edge(START, 'request_approval')
builder.add_edge('request_approval', 'perform_action')
builder.add_edge('perform_action', END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id': 'approval-demo'}}

first_result = graph.invoke(
    {'task': '发送项目周报邮件', 'approved': False, 'result': ''},
    config=config
)

answer = input(f'''{first_result['task']}，需要确认，输入yes or no:''')

second_result = graph.invoke(
    Command(resume=answer),
    config=config
)

print(second_result)
```

## **6\. 第二个例子：人工补充缺失信息**

人机协作不只是审批，也可以用来补充信息。

例如用户想让智能体写邮件，但没有提供收件人。

流程可以这样设计：

```Plain Text
START -> check_info -> draft_email -> END
```

如果缺少收件人，`check_info` 就暂停，让用户补充。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class EmailState(TypedDict):
    topic: str
    recipient: str
    email_draft: str

def check_info(state: EmailState):
    if state['recipient']:
        return {}

    recipient = interrupt({
        'question': '请补充邮件收件人',
        'current_topic': state['topic']
    })
    return {'recipient': recipient}

def draft_email(state: EmailState):
    draft = f'''收件人：{state["recipient"]}
主题：{state["topic"]}

你好，
这里是一封关于“{state["topic"]}”的邮件草稿。
请根据具体情况补充正文细节。
'''
    return {'email_draft': draft}

builder = StateGraph(EmailState)
builder.add_node('check_info', check_info)
builder.add_node('draft_email', draft_email)

builder.add_edge(START, 'check_info')
builder.add_edge('check_info', 'draft_email')
builder.add_edge('draft_email', END)

email_graph = builder.compile(checkpointer=MemorySaver())
email_config = {'configurable': {'thread_id': 'email-demo'}}

first_result = email_graph.invoke(
    {'topic': '下周项目会议安排', 'recipient': '', 'email_draft': ''},
    config=email_config
)

print(first_result)
```

### **代码解释**

这个例子展示了人机协作的另一种常见用途：补充缺失信息。

**\#\#\#\# \`EmailState\`**

状态包含三个字段：

- `topic`：邮件主题

- `recipient`：邮件收件人

- `email_draft`：生成的邮件草稿

**\#\#\#\# \`check\_info\` 节点**

这个节点负责检查信息是否完整。

如果 `recipient` 已经有值，就返回 `{}`，表示不需要修改状态。

如果 `recipient` 为空，就执行：

```Python
recipient = interrupt({...})
```

这会暂停流程，并向用户请求收件人信息。

#### **为什么这里不直接报错**

在真实智能体中，缺少信息不一定是错误。

更好的做法是：停下来问人。

这就是人机协作比普通异常处理更适合智能体流程的地方。

**\#\#\#\# \`draft\_email\` 节点**

这个节点依赖完整信息生成邮件草稿。

它会读取：

- `state['recipient']`

- `state['topic']`

然后写入 `email_draft`。

#### **第一次运行为什么会暂停**

因为初始状态中：

```Python
'recipient': ''
```

所以流程会停在 `check_info`，等待人补充收件人。

## **7\. 恢复并补充收件人**

现在模拟人工输入收件人，例如：`张经理`。

恢复后，流程会继续进入 `draft_email`，并生成邮件草稿。

```MATLAB
final_result = email_graph.invoke(
    Command(resume='张经理'),
    config=email_config
)

print(final_result['email_draft'])
```

### **代码解释**

这一段代码把人工补充的信息传回给图。

**\#\#\#\# \`Command\(resume='张经理'\)\`**

它会让刚才暂停的 `interrupt` 返回 `'张经理'`。

于是这行代码：

```Python
recipient = interrupt(...)
```

恢复后就相当于：

```Python
recipient = '张经理'
```

然后节点返回：

```Python
return {'recipient': recipient}
```

状态中的收件人就被补上了。

#### **后续为什么能生成邮件草稿**

恢复后，图会继续执行 `draft_email`。

这时状态已经包含：

- `topic='下周项目会议安排'`

- `recipient='张经理'`

所以就可以生成完整的邮件草稿。

#### **这个例子的重点**

人机协作不是只能做‘同意/不同意’。

它也可以用来让人补充缺失信息，然后让图继续完成后续任务。

## **8\. 第三个例子：人工修改 AI 草稿**

在很多真实场景中，AI 不一定直接执行最终结果，而是先生成一个草稿，让人修改或确认。

例如：

- AI 生成邮件草稿，人修改后再发送

- AI 生成计划，人确认后再执行

- AI 生成 SQL，人审核后再运行

下面我们用一个简单的任务计划示例。

```Python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class PlanState(TypedDict):
    goal: str
    draft_plan: str
    final_plan: str

def create_draft_plan(state: PlanState):
    draft = f'''目标：{state["goal"]}
步骤 1：明确当前需求
步骤 2：拆分任务并安排执行顺序
步骤 3：检查结果并总结经验
'''
    return {'draft_plan': draft}

def review_plan(state: PlanState):
    revised_plan = interrupt({
        'question': '请审核并修改计划草稿',
        'draft_plan': state['draft_plan']
    })
    return {'final_plan': revised_plan}

builder = StateGraph(PlanState)
builder.add_node('create_draft_plan', create_draft_plan)
builder.add_node('review_plan', review_plan)

builder.add_edge(START, 'create_draft_plan')
builder.add_edge('create_draft_plan', 'review_plan')
builder.add_edge('review_plan', END)

plan_graph = builder.compile(checkpointer=MemorySaver())
plan_config = {'configurable': {'thread_id': 'plan-demo'}}

review_result = plan_graph.invoke(
    {'goal': '准备一次 LangGraph 分享', 'draft_plan': '', 'final_plan': ''},
    config=plan_config
)

print(review_result)
```

### **代码解释**

这个例子展示的是‘AI 先生成草稿，人再修改确认’的模式。

**\#\#\#\# \`PlanState\`**

状态中有三个字段：

- `goal`：目标

- `draft_plan`：AI 或程序生成的草稿计划

- `final_plan`：人工确认后的最终计划

**\#\#\#\# \`create\_draft\_plan\` 节点**

这个节点根据目标生成一个初步计划。

这里为了教学简单，计划是用模板字符串生成的。

在真实项目中，这一步也可以换成大模型生成。

**\#\#\#\# \`review\_plan\` 节点**

这个节点调用 `interrupt`，把草稿交给人审核。

传给人的信息包括：

- 问题：请审核并修改计划草稿

- 当前草稿：`state['draft_plan']`

人可以直接接受，也可以修改后再返回。

#### **为什么返回\`final\_plan\`**

AI 生成的是草稿，不一定等于最终结果。

人工修改后的内容才写入 `final_plan`。

这能清楚地区分：

- AI 草稿是什么

- 人工最终确认的版本是什么

这在真实工作流中非常重要，因为最终责任通常属于人工确认后的结果。

## **9\. 恢复并提交人工修改后的计划**

现在模拟人工审核后，提交一个修改版计划。

```Python
human_revised_plan = '''目标：准备一次 LangGraph 分享
步骤 1：先介绍图结构、状态和边的概念
步骤 2：演示一个最小可运行的 LangGraph 示例
步骤 3：重点讲解人机协作中的 interrupt 和 resume
步骤 4：最后总结适合人机协作的真实业务场景
'''

final_plan_result = plan_graph.invoke(
    Command(resume=human_revised_plan),
    config=plan_config
)

print(final_plan_result['final_plan'])
```

### **代码解释**

这段代码模拟人工修改草稿后，把最终版本交回给图。

**\#\#\#\# \`human\_revised\_plan\`**

这是人工修改后的计划。

它和原始草稿相比，更具体，也更贴近分享主题。

**\#\#\#\# \`Command\(resume=human\_revised\_plan\)\`**

这表示把人工修改后的计划传回给 `interrupt`。

恢复后，`review_plan` 节点会继续执行：

```Python
return {'final_plan': revised_plan}
```

于是最终状态中就有了人工确认后的 `final_plan`。

#### **这个模式有什么用**

这类流程非常适合高风险或高要求任务。

例如：

- AI 生成合同条款，人审核后确认

- AI 生成客户邮件，人修改后发送

- AI 生成执行计划，人确认后落地

核心思想是：AI 负责初稿，人负责把关。

## **10\. 人机协作流程的关键点**

设计人机协作流程时，要特别注意下面几点。

### **10\.1 暂停点要放在关键位置**

不是每一步都需要人确认。

一般只在这些地方暂停：

- 信息缺失

- 风险较高

- 需要业务判断

- 执行前需要最终确认

### **10\.2 给人的信息要清楚**

`interrupt` 传出的内容应该让人一眼看懂：

- 当前要做什么

- 为什么要暂停

- 希望人提供什么

- 可选项有哪些

### **10\.3 恢复时必须使用同一个线程**

恢复流程时，必须使用同一个 `thread_id`。

否则图找不到之前暂停的位置。

### **10\.4 不要滥用人工确认**

如果每一步都让人确认，智能体就会变得很繁琐。

合理做法是：低风险步骤自动执行，高风险步骤人工把关。



