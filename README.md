# points

1. 使用RMSNorm归一化，不做均值中心化，节省开销同时效果不错
2. RoPE & YaRN，RoPE二维一组位置编码，YaRN上下文扩展
3. GQA，4个Q共享一个K和V，节约显存和算力
4. FFN

# studynote

1. SiLU 激活函数，引入非线性，段数增多，可以逼近拟合复杂的曲线

# model_structure

## 1. 自注意力机制（Self-Attention / GQA）
自注意力机制是 Transformer 架构的核心创新。
若无注意力机制，模型对每个词的理解仅为提取出的固定维度向量，无法区分同一词汇在不同语境下的语义差异。

### Transformer 的上下文理解逻辑
自注意力机制为序列中的每个词分配三个角色：$Q$（Query，需求向量）、$K$（Key，特征向量）、$V$（Value，内容向量），通过“找朋友”的方式实现上下文理解：
*   计算当前词的表示时，使用该词的 $Q$ 向量，与序列中所有词的 $K$ 向量进行点积运算，点积结果用于衡量相似度或匹配度。
*   匹配度高的词会获得更高的权重。
*   根据匹配度权重，将所有词的 $V$ 向量按比例加权求和，融合到当前词的表示中。
*   **最终效果**：经过自注意力层，每个词的向量会融合上下文信息，实现真正的语境感知。

---

## 2. 旋转位置编码（RoPE）
纯自注意力机制对序列中词的位置不敏感，无法区分词的先后顺序。
例如“张三打李四”与“李四打张三”，在纯注意力机制下的计算结果几乎一致，但语义完全相反。

### Transformer 的位置信息注入方案
该实现采用 `apply_rotary_pos_emb`（RoPE，旋转位置编码），利用三角函数（Cos 与 Sin）在注意力计算前，为每个位置的向量在多维空间中施加特定角度的旋转：
*   序列靠前的位置旋转角度较小，序列靠后的位置旋转角度较大。
*   两个词计算注意力匹配度时，其向量旋转角度的差值会反映两者的相对位置关系。

该机制为模型提供了对序列顺序的数学化理解方式。

---

## 3. 前馈神经网络（FFN）与 SiLU
若仅依赖自注意力机制，每个词的表示会过度融合周围信息，缺乏独立的特征变换能力。

### Transformer 的特征深度变换逻辑
自注意力计算完成后，每个词的特征向量会输入独立的 FFN（前馈神经网络）模块：
1.  首先将向量从隐藏层维度投影到更高维度（例如从 512 维提升至更高维度）。
2.  通过 **SiLU 激活函数** 引入复杂的非线性变换。
3.  最后将向量降维回原始隐藏层维度。

该步骤的作用是让模型在特征空间中进行深度的逻辑推演与特征提取。

---

## 完整训练流程闭环

将各模块组合，大模型的完整运行流程如下：

1. **词嵌入（Embedding）**：将无意义的 Token ID 转换为固定维度的初始特征向量。
2. **位置编码（RoPE）**：为向量注入位置信息，区分序列中的先后顺序。
3. **自注意力计算（Attention）**：每个词结合上下文信息更新自身特征表示。
4. **前馈网络变换（FFN）**：通过非线性激活函数进行深度特征提取与逻辑推演。
   *(步骤3与4构成 Transformer Block，会重复执行 `num_hidden_layers` 次)*
5. **最终输出**：序列最后一个 Token 的特征向量经过多层变换后，融合了全序列的上下文信息。
6. **词表投影（lm_head）**：将最终特征向量投影至词表维度，生成每个词的概率分数（Logits）。
7. **损失计算（Cross Entropy Loss）**：将模型输出与正确答案对比，计算损失值。
8. **参数更新（Backward & Step）**：误差信号沿计算图反向传播，更新所有网络参数（Q/K/V矩阵、FFN参数、词嵌入表等），准备下一轮迭代。

以上为 CookMind 模型运行时的完整底层逻辑。

# training

## 大模型是怎么训练的？

既然 token id 只是一个毫无关联的字典编号（比如“我”是 11，“吃”是 25，这并不意味着“吃”=“我”+ 14），那怎么用它来算数学公式呢？

要解开这个疑惑，你只需要记住一句话：

大模型并不是在预测一个“数值大小”，而是在做一道“有几万个选项的单项选择题”。

predicting the next token，本质上和一个能识别猫、狗、猪的图片分类器是一模一样的。只不过分类器只有3个类别，而大模型有 vocab_size（比如 6400）个类别。

### 1. 破除数字幻觉：词嵌入（Embedding）
在该项目的 model.py 中，第一步是：
`hidden_states = self.dropout(self.embed_tokens(input_ids))`

这里的 token ID（比如 11、25）**根本不参与任何加减乘除运算**，它完完全全只是一个**“数组下标（索引）”**！
- 该模型在内部维护了一个巨大的表格 `nn.Embedding(6400, 512)`。
- 当输入 token 编号 `11` 时，该模型不去对 `11` 做乘法，而是直接去表格的第 11 行，把一整条长度为 512 的浮点数向量（比如 `[0.12, -0.65, 0.88, ...]`）抽出来。
- 从这一刻起，毫无数学意义的 ID 就被丢弃了，参与后续层层 Transformer 复杂矩阵乘法计算的，是这些富含特征信息的 512 维向量。

### 2. 模型的输出形式：打分（Logits）
经过多层计算后，到了该模型的最后一层 `self.lm_head`：
`logits = self.lm_head(hidden_states)`

这一步该模型输出了什么呢？该模型不会直接输出一个单一数字“11”。它是把 512 维的向量投影回词表大小（6400维），输出的是**该模型对下一个位置可能出现的 6400 个词汇，分别打出的 6400 个分数**！

比如输出的 `logits` 可能是：
- 第 0 个词（pad）的得分：`-5.2`
- 第 11 个词（我）的得分：`-1.1`
- 第 25 个词（吃）的得分：`8.6`  👈 该模型目前最倾向这个词
- ……一直到第 6399 个词。

### 3. loss 是怎么算出来的？（交叉熵的魔法）
这是最困惑的地方：交叉熵损失 `F.cross_entropy` 是怎么消除 ID 数值差异的？见 model.py。

假设当前位置的真实标签（正确答案）应该是一个编号为 `25` 的 token。
交叉熵在计算时，会在内部把这个毫无意义的标量 `25`，变成一个有数学意义的 **“独热向量 (One-Hot Vector)”** 分布。
也就是构造一个长度 6400 的概率分布：除了第 25 个位置是 `1`，其他全为 `0`。
`True_Prob = [0, 0, ..., 1 (第25位), 0, ...]`

然后把该模型刚才打的分数（Logits）通过 Softmax 转化成总和为 1 的概率分布：
`Pred_Prob = [0.001, 0.002, ..., 0.85 (第25位), ...]`

**交叉熵损失的计算公式是：**
$$ Loss = -\log(\text{预测分布中正确答案对应的概率}) $$

看！如果正确答案是第 25 类的 token，公式**直接忽略了另外 6399 个错误概率的计算，也不在乎“25”这个数字本身有多大**，它唯一关心的数学量是：**“该模型给编号 25 的类别打出的概率是多少？”**
- 如果该模型预测第 25 类的概率是 $0.99$，$Loss = -\log(0.99) \approx 0$，几乎没有损失。
- 如果该模型很瞎，预测第 25 类的概率是 $0.01$，$Loss = -\log(0.01) \approx 4.6$，损失巨大！

这就是为什么 token 值大小没有数学意义，但能用来算 Loss：**因为真实 token 只是用来当指示牌，告诉交叉熵函数：“喂！去提取该模型最后一张大表里的第几个概率值来算对数！”**

### 4. 怎么反向传播？怎么更新参数？
到了这一步，手里有一个标量 $Loss$（例如 `4.6`）。

该代码框架在执行 `scaler.scale(loss).backward()` 时，PyTorch 会利用微积分里的**链式法则（Chain Rule）**沿着计算图反向追溯。

举个具象的例子：
1. $Loss$ 发现自己很大，原因是 `Softmax(正确位置25)` 的概率太低了（只有 0.01）。
2. 于是它对上一层的 `logits[25]` 求偏导，传递一个信号：“把第 25 项的原始打分提高！把其他 6399 项的分数压低！”
3. `lm_head` 收到信号后，调整里面的矩阵权重（Weight），说：“为了下次让第 25 项分数变高，我需要把输入我的这段 512 维向量在特定维度上的权重调大一点。”
4. 这个梯度信号一路往回传，穿过 FFN，穿过 Attention（甚至调整 Q、K、V 寻找之前 token 的专注度权重），最后传到第一层的 `Embedding`。
5. `Embedding` 收到信号后更新表格：“原来当我们看到前文是某个词的时候，后续走下来的预测概率不够……我要略微修改前面那些词所对应的浮点数向量，让特征更匹配。”

这就是 `optimizer.step()` 时发生的事情。每一次参数更新，都是全网几百万/几亿个参数在响应这个信号，纷纷微微调整自身数值（旋钮），只为了下一次走同一个计算流程时，该模型输出给那个目标 token 编号的概率能从 $0.01$ 变成 $0.02$。不断循环，直到给出极大的概率。

---

**总结**：
1. **输入阶段：** token ID 等同于考号，不运算，只查表提取高维向量。
2. **输出阶段：** 该模型也不输出标量，它输出词表所有词的候选概率。
3. **Loss 计算：** 真实标签的作用就是个“裁判指针”，指着众多概率中的一个说：“计算这个对错！”
4. **反向传播：** 根据错多少，要求前面的所有神经元调整数值，迎合这个正确词的特征。

## 预训练数据处理

1. **读取数据**
初始化阶段通过 `lm_dataset.py` 完整读取 json 或 jsonl 格式文件。
要求每条样本至少包含一个 `text` 字段，取样时会访问 `lm_dataset.py` 中的 `sample['text']`。

2. **限制最大长度**
`max_length` 参数保存在 `lm_dataset.py` 中。
该参数定义了最终输入模型的序列固定长度，固定长度的优势包括：
   - 降低 DataLoader 拼接 batch 的复杂度
   - 保证模型输入张量形状稳定
   - 简化训练实现逻辑

对应的代价为：短文本会被填充 pad 字符，长文本会被截断。

3. **文本转 token**
单条样本的处理与核心编码动作均在 `lm_dataset.py` 中完成。
该步骤执行以下操作：
   - 将 `sample['text']` 转换为字符串格式
   - 通过 tokenizer 执行编码
   - 设置 `add_special_tokens=False`，禁止 tokenizer 自动添加特殊符号
   - 设置 `truncation=True`，对超长文本执行截断
   - 设置 `max_length=self.max_length-2`，为后续手动拼接 bos 和 eos 预留2个token位置

此处预留2个位置为关键设计，后续 `lm_dataset.py` 会手动拼接 bos 和 eos 标识，若不提前预留位置会导致最终序列长度溢出。

4. **手动补 bos 和 eos**
相关实现在 `lm_dataset.py` 中。
标识含义：
   - bos：序列开始标识
   - eos：序列结束标识

选择手动拼接而非依赖 tokenizer 自动添加，是为了保证行为可控，避免不同 tokenizer 默认特殊符号处理逻辑不一致带来的问题。

5. **右侧 padding**
相关实现在 `lm_dataset.py` 中。
若文本长度不足 `max_length`，会在序列右侧填充 `pad_token_id`，直至序列长度等于 `max_length`。

示例：
原始token序列：`[ bos, A, B, C, eos ]`
若 `max_length=8`，最终序列为：`[ bos, A, B, C, eos, pad, pad, pad ]`

该实现采用右 padding 方案，而非左 padding。

6. **转成张量**
相关实现在 `lm_dataset.py` 中。
采用 long 类型的核心原因：
   - embedding 查表操作要求索引为整数类型
   - cross_entropy 损失函数要求标签为整数类别id

7. **生成 labels**
相关实现在 `lm_dataset.py` 中。
采用 `labels = input_ids.clone()` 的处理方式，为因果语言模型（causal LM）的标准实现。

自回归模型的核心训练目标为：给定前文token，预测下一个token。

示例：输入序列为 `bos A B C eos`
模型内部会自动完成错位匹配：
   - 用 bos 预测 A
   - 用 bos A 预测 B
   - 用 bos A B 预测 C
   - 用 bos A B C 预测 eos

因此数据集无需手动对 labels 执行左右移位操作，该逻辑已在 `model.py` 的模型前向传播中实现。

8. **pad 位置标签改为 -100 的原因**
相关实现在 `lm_dataset.py` 中，为预训练数据处理的核心步骤之一。

若不修改pad位置标签，pad区域会参与损失计算，导致模型被迫学习“预测pad字符”，该过程无训练价值，还会污染梯度。

将pad位置标签设为-100后，交叉熵损失函数会自动忽略这些位置，该逻辑由 `model.py` 中的 `ignore_index=-100` 参数实现。

完整逻辑链路：
   - 数据集将pad对应的标签设置为-100
   - 模型计算损失时忽略标签为-100的位置
   - 仅真实文本token会产生监督信号

这也是当前预训练流程即使不传入 attention_mask 仍可正常训练的核心原因：pad位置本身已不参与损失计算。

---

## 预训练样本示例
假设原始文本为“今天天气很好”，经tokenizer编码后得到token序列：`[11, 25, 78]`

预设参数：
   - `bos_token_id = 1`
   - `eos_token_id = 2`
   - `pad_token_id = 0`
   - `max_length = 8`

处理结果：
`input_ids` 最终序列：`[1, 11, 25, 78, 2, 0, 0, 0]`
`labels` 初始序列：`[1, 11, 25, 78, 2, 0, 0, 0]`
pad位置替换为-100后：`[1, 11, 25, 78, 2, -100, -100, -100]`

模型内部完成错位匹配后，仅前序真实token对后续token的预测会参与损失计算，后三个pad相关位置会被自动忽略。

---

## SFT 数据处理
SFT类定义在 `lm_dataset.py` 中，与预训练的核心差异为：
预训练流程通常让所有真实token参与监督；
SFT流程通常仅让assistant的回答部分参与监督，user输入部分不参与。

因此SFT的数据处理逻辑更为复杂。

1. **读取对话样本**
初始化逻辑在 `lm_dataset.py` 中，数据预期格式为conversations对话结构，而非纯文本text。

2. **预先构造角色边界标记**
在 `lm_dataset.py` 中，提前将两段特殊片段编码为token序列：
   - `bos_id`：assistant回答开始标识
   - `eos_id`：assistant回答结束标识

该步骤的核心目的，是在完整prompt的token序列中准确定位assistant回答对应的区间。

3. **对话预处理**
相关实现在 `lm_dataset.py` 中。
`pre_processing_chat` 函数的核心逻辑：若当前对话第一条消息非system角色，会按预设概率自动补充一条system prompt。

该操作为轻量数据增强，核心作用包括：
   - 提升模型对system指令格式的熟悉度
   - 增加system设定的多样性
   - 提升chat模板的一致性

4. **生成 chat prompt**
相关实现在 `lm_dataset.py` 中。
通过调用 `tokenizer.apply_chat_template`，将结构化对话messages渲染为模型训练所需的文本格式。

示例：原始数据结构
```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```
经chat template处理后，会转换为带角色标记的完整文本字符串，再通过tokenizer编码为token序列。

5. **后处理 think 标签**
相关实现在 `lm_dataset.py` 中。
该步骤会按预设概率删除空的think标签块，避免模型学习到无意义的空思维模板。

6. **编码、截断、补 pad**
相关实现在 `lm_dataset.py` 中，处理逻辑与预训练一致：
   - 完整prompt编码为input_ids
   - 超长文本截断至max_length
   - 长度不足则执行右侧pad填充

7. **仅为assistant部分生成 labels**
核心实现在 `lm_dataset.py` 中，为SFT与预训练最本质的区别。

处理逻辑：先将完整labels初始化为全-100，即默认所有位置均不参与损失计算；再通过滑动窗口扫描input_ids，定位每一段assistant回答的起止位置；仅将assistant回答区间内的token复制到labels中，其余位置保持-100不变。

该处理方式实现的训练效果：
   - user提问部分仅提供上下文，不计入损失
   - assistant回答部分为唯一的监督目标

该方案为指令微调的通用标准实现。

### SFT 该设计的核心原因
SFT的核心目标不是让模型背诵完整对话模板，而是让模型学会“基于用户输入输出高质量的assistant回答”。

若user与system的token也参与损失计算，模型会将大量训练能力浪费在“复述输入格式”上，而非聚焦于回答生成本身。

### pre_processing_chat 与 post_processing_chat 细节
两个函数均定义在 `lm_dataset.py` 中，不属于张量构造的核心逻辑，但会直接影响训练数据分布。

`pre_processing_chat` 逻辑：
   - 若对话首条消息非system角色
   - 按 `add_system_ratio` 预设概率补充system提示
   - system内容从预设列表中随机选取

`post_processing_chat` 逻辑：
   - 若prompt中出现空的think模板
   - 按预设概率删除该模板
   - 保留少量样本不做删除，维持数据多样性

两个步骤本质均为轻量数据增强操作。

### 该文件核心设计思想
可总结为三句话：
1. 预训练数据：尽可能让所有真实文本token参与next-token预测学习
2. SFT数据：仅让assistant的回答部分参与监督
3. pad位置永远不应该产生损失

### 核心理解要点
1. `lm_dataset.py` 中设置 `max_length-2` 的原因：为手动添加的bos和eos预留位置
2. `lm_dataset.py` 中将pad位置改为-100的原因：让损失函数忽略无效填充位置
3. `model.py` 中对logits和labels错位一位的原因：因果语言模型的核心逻辑是用前文预测下一个token
4. `lm_dataset.py` 中SFT仅为assistant部分打标签的原因：监督目标为回答生成，而非完整对话模板

---

## pretrain
该pretrain流程为标准的因果语言模型训练链路，核心逻辑如下：
1. 训练脚本完成模型、分词器、数据集、优化器与混合精度环境的构造，入口为 `train_pretrain.py`
2. 数据集将原始文本转换为固定长度的input_ids与labels，逻辑定义在 `lm_dataset.py`
3. 模型将input_ids转换为每个位置对“下一个token”的预测分数logits，逻辑定义在 `model.py`
4. 训练循环通过labels与logits计算交叉熵损失，执行反向传播更新参数，学习率采用余弦衰减策略，相关逻辑在 `trainer_utils.py`

核心逻辑可概括为：模型基于前文token，学习预测后续token。

### 一、数据到训练样本的转换流程
预训练样本来自 `lm_dataset.py`，单条样本的处理顺序如下：
1. 读取 `sample["text"]` 字段
2. 通过tokenizer编码为token id
3. 禁用tokenizer自动添加特殊符号，手动拼接bos与eos标识
4. 超长文本截断至 `max_length - 2`，为bos与eos预留位置
5. 短文本执行右侧pad填充至固定长度
6. labels初始化为input_ids的克隆
7. 所有pad位置的labels设置为-100，相关实现在 `lm_dataset.py`

该设计的核心意义：
   - input_ids为模型的实际输入
   - labels为监督信号
   - -100标识该位置不参与损失计算
   - 模型仅在真实文本token上执行学习，不会在填充pad上学习

示例：单条文本编码后序列为 `bos 我 爱 你 eos pad pad`
对应结果：
   - input_ids: `bos, 我, 爱, 你, eos, pad, pad`
   - labels: `bos, 我, 爱, 你, eos, -100, -100`

注：序列错位操作在模型损失计算阶段完成，而非数据集阶段。

### 二、模型的核心学习目标
核心逻辑定义在 `model.py` 中的 `CookMindForCausalLM.forward` 方法。
模型内部前向传播流程：
1. input_ids经过词嵌入层转换为向量表示
2. 输入多层Transformer block，主干逻辑在 `model.py` 中
3. 每层依次执行注意力计算与前馈网络，均采用残差结构
4. 最终经过lm_head层，将隐藏状态投影至词表维度，得到logits

logits的形状为：`batch_size × seq_len × vocab_size`
含义为：序列中每个位置，模型都会输出对应“下一个token概率分布”的原始分数。

### 三、损失错位计算的核心原因
核心逻辑在 `model.py` 中实现，处理方式如下：
   - `shifted_logits`：logits去掉最后一个位置
   - `shifted_labels`：labels去掉第一个位置

因果语言模型的核心目标不是“自预测”，而是基于前文预测下文：
   - 第1个位置的输出，预测第2个token
   - 第2个位置的输出，预测第3个token
   - 第t个位置的输出，预测第t+1个token

因此尽管数据集中labels与input_ids外观一致，在进入交叉熵计算前，模型内部会自动完成时序对齐。

沿用前文示例，原始序列为 `bos, 我, 爱, 你, eos`
训练时实际的监督匹配关系：
   - 用bos位置的输出预测“我”
   - 用“我”位置的输出预测“爱”
   - 用“爱”位置的输出预测“你”
   - 用“你”位置的输出预测eos

末尾pad位置因labels为-100，会被交叉熵函数自动忽略。

### 四、不传入attention_mask的合理性
该pretrain实现中不传入attention_mask为合理设计，核心原因如下：
训练数据采用“右侧padding”格式，即 `真实token 真实token 真实token pad pad pad`
该场景下：
1. 因果掩码本身已保证模型无法看到未来token
2. pad全部位于序列末尾，前文真实token不会访问到后续pad
3. 末尾pad位置虽参与前向传播，但对应labels为-100，不参与损失计算

因此该pretrain流程中不传入attention_mask可正常运行，该逻辑仅适用于右padding场景。若后续改为左padding、packed samples或复杂推理缓存逻辑，需显式传入mask。

### 五、训练循环单步执行流程
主循环定义在 `train_pretrain.py` 中，单step执行顺序如下：
1. 从DataLoader中获取一个batch数据
2. 将input_ids与labels迁移至对应设备
3. 基于全局step计算当前学习率，学习率函数定义在 `trainer_utils.py`
4. 进入autocast混合精度上下文，执行前向传播
5. 计算得到损失值
6. 基于 `accumulation_steps` 对损失做缩放，执行梯度累积
7. 执行 `scaler.scale(loss).backward()` 完成反向传播
8. 累积步数达标后，执行参数更新：
   - 对梯度执行unscale操作
   - 执行梯度裁剪
   - 执行 `optimizer.step()` 更新参数
   - 执行 `scaler.update()` 更新缩放因子
   - 执行 `optimizer.zero_grad()` 清空梯度

梯度累积的核心作用：在显存不足以支撑大batch时，通过累积多个小batch的梯度，等效模拟大batch训练效果。
示例：`batch_size=32`，`accumulation_steps=8`，单次参数更新等效于使用256条样本。

### 六、动态学习率设计
该实现未采用固定学习率，而是每个step动态更新，策略为：
   - 训练前期学习率较高
   - 训练后期学习率逐步下降
   - 最低学习率不低于初始值的10%左右

该策略为简化版余弦退火，优势为训练前期收敛速度快，训练后期参数更新更平稳，避免收敛阶段出现大幅抖动。

### 七、断点续训与DDP适配
相关逻辑主要分布在 `train_pretrain.py` 与 `trainer_utils.py` 中，核心实现如下：
1. 若通过torchrun启动，自动初始化DDP分布式训练环境
2. 每个epoch为DistributedSampler设置epoch值，保证多卡shuffle结果一致
3. 若存在resume checkpoint，自动恢复以下状态：
   - 模型参数
   - 优化器状态
   - 梯度缩放器scaler状态
   - 训练epoch与step进度
4. 通过SkipBatchSampler跳过已训练完成的batch，避免重复训练

该套逻辑的核心目标为：训练中断后，可从断点位置继续训练，避免重复计算。

---

## SFT 的本质
该代码实现的SFT，核心逻辑不是让模型学习完整文本的所有token，而是仅让模型学习assistant的回答部分。

与pretrain的核心区别：
1. pretrain流程中，除pad外几乎所有token均参与监督
2. SFT流程中，system与user部分仅作为条件输入，不参与损失计算
3. 仅assistant的输出内容为模型需要拟合的目标

SFT可概括为：给模型输入整理完成的多轮对话上下文，仅让模型对assistant的回复部分负责。
相关逻辑主要在 `lm_dataset.py`、`train_full_sft.py` 与 `model.py` 中实现。

### 一、SFT单条样本的处理流程
入口定义在 `lm_dataset.py` 中，单条样本处理步骤如下：
1. 读取样本中的conversations字段
2. 执行 `pre_processing_chat` 预处理
3. 通过tokenizer的chat template将多轮消息拼接为完整prompt字符串
4. 执行 `post_processing_chat` 后处理
5. 将完整prompt编码为input_ids
6. 超长文本截断至max_length
7. 执行右侧pad填充
8. 单独生成labels，仅为assistant部分设置有效标签

该流程与pretrain的核心差异不在input_ids，而在labels的构造逻辑：
input_ids仍为完整对话的全部内容，labels仅对assistant回答部分设置有效监督，其余位置均为-100。

### 二、pre_processing_chat 核心逻辑
相关实现在 `lm_dataset.py` 中，核心操作为：
若对话首条消息非system角色，按预设概率（默认0.2，即20%）补充一条system提示词。

该操作为轻量数据增强，核心作用包括：
1. 让模型接触更多带system指令的对话格式
2. 强化模型对assistant角色设定的适配能力
3. 避免训练样本的开头风格过于单一

该操作不会对所有样本生效，因此训练数据分布中会同时存在无system的原始对话与补充了system的对话，让模型学习到更通用的聊天格式。

### 三、create_chat_prompt 核心逻辑
相关实现在 `lm_dataset.py` 中，核心作用不是分词，而是将结构化对话转换为统一的模板字符串。

原始数据通常为多轮结构化消息，格式如下：
1. system
2. user
3. assistant
4. user
5. assistant

模型无法直接处理字典列表格式，因此通过 `tokenizer.apply_chat_template` 将其展开为完整字符串。

该阶段执行的操作包括：
1. 为每一轮对话添加角色模板标记
2. 明确区分user与assistant角色
3. 将多轮对话拼接为模型可识别的文本格式

若首条system消息包含functions字段，会同步将tools信息传入模板，因此该SFT数据格式同时支持普通聊天与工具调用能力的训练。

### 四、post_processing_chat 核心逻辑
相关实现在 `lm_dataset.py` 中，核心处理对象为空的think标签块（即无内容的思维链占位片段）。

处理策略为：
1. 若prompt中存在空think块，按预设概率删除该块
2. 保留少量样本不做删除，维持数据多样性

该设计的核心目标为：避免模型被大量空think模板污染，同时不完全丢失该格式的适配能力，属于训练数据分布的精细化控制。

### 五、input_ids 的生成逻辑
相关实现在 `lm_dataset.py` 中，处理流程如下：
1. 执行 `tokenizer(prompt).input_ids` 完成编码
2. 超长文本截断至max_length
3. 右侧pad填充至固定长度

该流程与pretrain类似，核心差异为：
1. 编码前先执行chat template处理，而非直接对纯文本编码
2. 未手动拼接bos与eos，依赖chat template与tokenizer的聊天格式定义

即SFT的序列边界不再是纯文本的起止，而是多轮对话模板中的角色边界。

### 六、SFT核心：labels 的生成逻辑
核心实现在 `lm_dataset.py` 中，`generate_labels` 函数的核心逻辑可概括为：
完整输入全部传入模型，但仅assistant回答区间的内容会被复制到labels中，其余位置全部保持-100。

初始化逻辑：
1. labels先初始化为全-100
2. 即默认所有位置均不参与损失计算

区间定位逻辑：
1. 在input_ids中扫描assistant段落的起点标记
2. 定位该段assistant内容的终点标记
3. 将该区间对应的input_ids复制到labels中
4. 继续向后扫描下一轮assistant内容

通过两个标记完成定位：
1. `bos_id`：assistant段落的开头模板，定义在 `lm_dataset.py`
2. `eos_id`：assistant段落的结尾模板，定义在 `lm_dataset.py`

仅assistant回答正文与结尾eos标记会被设置为有效标签，其余位置（system提示、user提问、assistant起始模板、pad）均保持-100，不参与损失计算。

### 七、assistant开头标记不参与监督的设计原因
该细节实现在 `lm_dataset.py` 中，核心代码为 `start = i + len(self.bos_id)`，即从assistant起始模板之后才开始设置有效labels。

该设计的核心目标为：模型无需拟合assistant角色标记本身，仅需学习assistant正文的生成，让训练目标更聚焦，上下文仅负责提供条件信息，回答内容负责提供监督信号，比全模板监督的设计更纯净。

### 八、eos标记纳入监督的设计原因
相关实现在 `lm_dataset.py` 中，会将assistant内容直至eos结束标记的全部内容写入labels。

该设计的核心目标为：让模型不仅学会生成回答内容，还学会判断回答的结束时机。
若不训练结束边界，模型推理时易出现以下问题：
1. 回复无法正常终止，持续生成无效内容
2. 错误生成多轮对话结构
3. 结束位置不稳定

因此将eos纳入监督，本质是训练模型的“回答终止能力”。

### 九、损失的最终计算逻辑
尽管SFT数据集已将labels处理为仅保留assistant区间，但next token的时序对齐仍在模型内部完成，位置在 `model.py` 中。

计算逻辑与pretrain完全一致：
1. logits去掉最后一个时间步
2. labels去掉第一个时间步
3. 计算交叉熵损失
4. 设置 `ignore_index=-100`

该逻辑实现的效果：
1. 每个位置仅负责预测下一个token
2. 非assistant目标的位置因labels为-100，会被自动跳过
3. 最终损失仅来源于assistant回复区间

即SFT未修改模型的损失公式，仅通过labels掩码改变了“参与损失计算的位置”，为该实现的核心设计点。

### 十、SFT与pretrain的本质差异
仅看训练主循环，SFT与pretrain的结构几乎完全一致，`train_full_sft.py` 与pretrain脚本均遵循以下流程：
1. 构建数据集
2. 初始化DataLoader
3. 前向传播
4. 计算损失
5. 反向传播
6. 梯度裁剪
7. 优化器参数更新
8. 保存checkpoint

两者的核心差异仅为两层：
1. 数据来源不同：pretrain使用纯文本数据，SFT使用多轮对话数据
2. labels构造不同：pretrain除pad外全量监督，SFT仅监督assistant回复

因此SFT可概括为：模型结构与训练框架不变，仅改变监督信号的选择方式。

### 十一、该SFT设计的合理性
聊天场景下，模型的核心任务不是“复述用户输入”，而是“基于用户输入生成对应回答”。

若将user内容也纳入损失计算，会带来两个核心问题：
1. 模型会被鼓励生成用户已输入的内容
2. 监督目标会混入大量“上下文复现”的无效信号，而非聚焦“回答生成”

仅训练assistant区间的设计，本质是明确告知模型：前文对话负责理解，后续assistant回答负责生成，与真实使用场景完全匹配。

---

## SFT 核心记忆要点
可概括为三句话：
1. 完整对话全部输入模型
2. 仅assistant回复部分参与损失计算
3. 损失仍为标准的next token预测，仅通过-100屏蔽非回答区域

---

## LoRA 的核心思想
LoRA的核心逻辑为：不直接更新大模型的原始权重 $W$，而是在原始权重旁额外新增一个“低秩增量” $\Delta W$，训练阶段仅学习该增量部分。

数学表达式如下：
$$
y = Wx + \Delta W x
$$

LoRA将 $\Delta W$ 拆解为两个更小的矩阵：
$$
\Delta W = BA
$$

其中：
- $A \in \mathbb{R}^{r \times d_{in}}$
- $B \in \mathbb{R}^{d_{out} \times r}$
- $r$ 为秩（rank），通常远小于原始权重的维度

最终前向传播表达式为：
$$
y = Wx + BAx
$$

该设计下，训练阶段无需修改原始大矩阵 $W$，仅需训练A、B两个小矩阵，可显著降低参数量、显存占用与优化器状态开销。

该代码中的LoRA实现在 `model_lora.py` 中。

### 一、LoRA模块的核心结构
LoRA模块定义在 `model_lora.py` 中，核心结构如下：
1. `A = nn.Linear(in_features, rank, bias=False)`
2. `B = nn.Linear(rank, out_features, bias=False)`
3. 前向传播返回 `B(A(x))`

即先将输入从高维压缩至低维，再从低维投影回输出维度，对应低秩更新 $\Delta W = BA$ 的数学逻辑。

两个关键初始化细节：
1. 矩阵A采用高斯初始化，相关实现在 `model_lora.py`
2. 矩阵B采用全0初始化，相关实现在 `model_lora.py`

该初始化的核心效果：
   - 训练初始阶段 `B(A(x)) = 0`
   - 注入LoRA的初始时刻，模型输出与原始模型完全一致
   - 训练启动后，LoRA分支才会逐步学习到有效偏移量

该方案为LoRA的经典初始化设计，可避免初始加载时破坏底座模型的行为。

### 二、LoRA模块注入原模型的逻辑
注入逻辑定义在 `model_lora.py` 中，`apply_lora(model, rank=8)` 函数执行以下三步操作：
1. 遍历模型中的所有子模块
2. 筛选出符合条件的 `nn.Linear` 层
3. 为符合条件的线性层动态挂载 `lora` 子模块，并重写原层的 `forward` 方法

重写后的前向计算逻辑：
$$
\text{original}(x) + \text{lora}(x)
$$

对应代码实现：
   - 原始层输出：`layer1(x)`
   - LoRA分支输出：`layer2(x)`
   - 最终返回两者之和

相关实现在 `model_lora.py` 中，即原模型的目标线性层，从“单路输出”变为“主干输出+可训练增量输出”的并联结构。

### 三、LoRA的注入层筛选规则
相关实现在 `model_lora.py` 中，筛选条件为：
1. 模块类型为 `nn.Linear`
2. `module.weight.shape[0] == module.weight.shape[1]`

即仅对“方阵线性层”注入LoRA。

该规则下，以下层更易被选中：
   - 注意力模块中输入输出维度一致的投影层
   - FFN模块中权重为方阵的线性层

并非所有线性层都会被注入LoRA，该实现与标准LoRA实现存在差异。
标准LoRA更常见的方案为显式指定目标层，例如 `q_proj`、`k_proj`、`v_proj`、`o_proj` 等注意力相关层。

该实现的优势为逻辑简单，缺点为层筛选的精细度不足，基于矩阵形状筛选而非模块语义筛选。

### 四、LoRA训练的显存与算力优势
核心逻辑在 `train_lora.py` 中实现，训练脚本执行了参数冻结操作：
1. 名称包含 `lora` 的参数，设置 `requires_grad = True`
2. 其余所有参数，设置 `requires_grad = False`

该设计带来的核心优势：
   - 底座模型仅参与前向传播与反向传播的梯度传递
   - 优化器仅更新LoRA参数
   - AdamW优化器仅维护LoRA参数的优化状态

优化器定义在 `train_lora.py` 中，仅将 `lora_params` 传入AdamW优化器。

因此从训练角度，LoRA微调的本质为：
1. 基座模型提供稳定的通用能力
2. LoRA小分支负责学习任务相关的偏移量
3. 最终仅更新少量附加参数

这也是LoRA可实现低成本微调的核心原因。

### 五、LoRA训练的任务类型
从训练脚本可知，LoRA训练采用的数据集仍为 `SFTDataset`，相关实现在 `train_lora.py` 中。

这说明该LoRA训练未采用独立的损失函数，也未采用预训练的全token监督方案，本质仍为：
   - 聊天式SFT任务
   - 仅监督assistant回复部分
   - 仅将“可训练参数”从全模型缩小为LoRA参数

因此该LoRA训练可概括为：LoRA版SFT = SFT的训练目标 + LoRA的参数高效更新方式。

即：
1. 数据逻辑与SFT完全一致
2. labels构造逻辑与SFT完全一致，仅训练assistant部分
3. 模型损失仍为标准的next-token交叉熵
4. 唯一改变的是参数更新策略

### 六、LoRA训练单步执行流程
相关实现在 `train_lora.py` 中，单step核心流程如下：
1. 从 `SFTDataset` 中获取一批 `input_ids` 与 `labels`
2. 输入模型执行前向传播
3. 获取 `res.loss` 损失值
4. 执行反向传播计算梯度
5. 仅对 `lora_params` 执行梯度裁剪
6. 优化器仅更新 `lora_params`

关键细节：尽管仅LoRA参数可训练，前向传播阶段整个模型仍会正常运行。
即：
   - 原始大模型负责提供主干能力
   - LoRA分支在每个注入层提供小幅修正
   - 损失通过完整计算图回传
   - 最终仅LoRA权重会被修改

因此LoRA并非“仅运行部分网络”，而是“全网络前向，局部参数更新”。

### 七、LoRA权重的保存逻辑
保存逻辑定义在 `model_lora.py` 与 `train_lora.py` 中。

`save_lora` 函数执行以下操作：
1. 遍历模型的所有模块
2. 筛选出包含 `lora` 属性的层
3. 仅提取这些层中 `lora.state_dict()` 的内容
4. 保存为独立的权重文件

该设计的核心优势：
1. 权重文件体积更小
2. 可复用同一个底座模型
3. 不同任务仅需切换对应的LoRA权重文件

该方案为LoRA的通用使用方式：一份底座模型，搭配多份任务适配器。

训练脚本中同时调用了 `lm_checkpoint(...)`，相关实现在 `train_lora.py` 中，因此训练过程中会同时保存两类文件：
1. LoRA专用权重文件
2. 用于断点续训的checkpoint

断点续训的checkpoint中会包含当前模型的完整 `state_dict`，因模型已挂载 `lora` 子模块，因此该checkpoint会包含LoRA参数，用于恢复训练状态。

### 八、LoRA权重的加载逻辑
加载逻辑定义在 `model_lora.py` 中，执行以下操作：
1. 读取保存的权重字典
2. 去除可能存在的 `module.` 前缀
3. 遍历模型中所有带 `lora` 属性的模块
4. 基于模块名提取对应的LoRA子权重
5. 将权重加载至 `module.lora` 中

该设计的核心逻辑：LoRA权重文件本身不包含完整模型结构，仅按名称保存每个注入点的LoRA参数。

因此加载LoRA权重必须遵循以下顺序：
1. 加载与训练时一致的底座模型
2. 执行 `apply_lora` 注入LoRA模块
3. 执行 `load_lora` 加载LoRA权重

若模型中无对应的 `lora` 子模块，会导致加载失败。

### 九、该实现与标准LoRA的差异
该实现为可落地的简化版LoRA，与工业界标准实现相比，缺少以下核心设计：

1. 缺少 `alpha / rank` 缩放项
标准LoRA的前向传播表达式为：
$$
y = Wx + \frac{\alpha}{r} BAx
$$
该缩放项可更稳定地控制增量分支的输出尺度，当前实现直接采用 `y = Wx + BAx`，逻辑更简单，但缺少强度控制旋钮。

2. 缺少dropout层
标准LoRA通常会在LoRA分支前添加dropout层做正则化，当前实现未添加，逻辑更纯粹，但正则化能力较弱。

3. 缺少精细的目标模块筛选
当前实现基于“方阵线性层”筛选注入目标，而非显式指定注意力投影层，使用便捷但可控性较弱。

4. 缺少权重merge/unmerge逻辑
标准LoRA通常支持推理前将 $BA$ 直接合并至原始权重中，降低推理时的额外开销，当前实现未做merge，采用运行时并联分支的方案。

### 十一、LoRA流程核心记忆要点
可概括为四步：
1. 加载已完成SFT的底座模型
2. 为指定线性层外挂小型低秩分支
3. 冻结底座模型，仅训练LoRA分支
4. 最终仅保存LoRA轻量参数

代码层面可概括为一句话：LoRA训练本质是“采用SFT的数据监督方式，仅微调每个目标线性层旁的低秩补丁”。

### 十二、该实现的核心理解要点
最核心的逻辑并非新增了LoRA类，而是以下设计：
原始模型权重为知识底座，LoRA参数为任务偏移量。

因此LoRA训练的核心目标，不是重新教模型语言能力，而是在尽可能少改动底座模型的前提下，让模型向特定任务方向偏移。

这也是LoRA适用于以下场景的核心原因：
1. 垂直领域微调
2. 多任务场景切换
3. 低成本算法实验
4. 一份底座搭配多任务适配器

---

## DPO 的核心逻辑
DPO的核心目标，不是教模型“什么是正确答案”，而是教模型“在两个候选回答中，应更偏好chosen回复，而非rejected回复”。

与SFT的核心差异：
1. **SFT为单答案监督**：给定上下文，直接拟合assistant的标准回答
2. **DPO为相对偏好监督**：给定同一上下文，模型需让chosen回复的概率高于rejected回复

该实现的主流程在 `train_dpo.py` 中，数据构造逻辑在 `lm_dataset.py` 中。

核心逻辑可概括为：同一问题下，策略模型需比参考模型更偏向用户偏好的回答。

### 一、DPO数据集格式
DPO数据集定义在 `lm_dataset.py` 中，每条样本包含两组对话：
1. **chosen**：质量更优的回答路径
2. **rejected**：质量更差的回答路径

两个字段均为完整的多轮消息列表，而非单句字符串，即chosen与rejected均包含完整上下文与assistant回答，仅最终偏好的回复存在差异。

`lm_dataset.py` 中 `DPODataset` 的处理步骤如下：
1. 读取chosen与rejected两组对话消息
2. 分别执行 `apply_chat_template`，拼接为完整对话字符串
3. 分别执行 `post_processing_chat` 后处理
4. 分别通过tokenizer编码，执行截断与pad填充至固定长度
5. 构造input与target序列
6. 额外构造loss mask，仅让assistant回复部分参与偏好比较

因此DPO与SFT一致，不会对完整输入的所有位置计算损失，仅关注assistant回答部分。

### 二、DPODataset多字段返回的设计原因
`lm_dataset.py` 中 `DPODataset` 返回以下字段：
1. `x_chosen`
2. `y_chosen`
3. `mask_chosen`
4. `x_rejected`
5. `y_rejected`
6. `mask_rejected`

字段含义：
1. `x`：输入序列
2. `y`：下一个token的目标序列
3. `mask`：标识哪些位置属于assistant回复、应参与偏好比较

序列构造采用标准的next-token格式：
   - `x_chosen = chosen_input_ids` 去掉最后一个token
   - `y_chosen = chosen_input_ids` 去掉第一个token

rejected序列采用相同的处理方式。

该步骤与普通语言模型训练一致，仍为“当前位置预测下一个token”，差异在于DPO不会直接用交叉熵优化y，而是将这些token的对数概率汇总后，执行偏好比较。

### 三、generate_loss_mask 的核心作用
该函数为DPO实现的核心细节之一，相关实现在 `lm_dataset.py` 中，逻辑与SFT的 `generate_labels` 高度相似：
1. 通过assistant开头标记定位回答起始位置
2. 通过eos标记定位回答结束位置
3. 仅将assistant回复区间对应的位置设为1
4. 其余位置全部设为0

该设计的核心含义：
   - system提示不参与偏好比较
   - user提问不参与偏好比较
   - 仅assistant的回答内容，决定chosen回复与rejected回复的优劣

该设计符合偏好学习的核心逻辑：偏好学习的核心是比较回答质量，而非比较上下文内容。

因此该实现的DPO，本质是在相同上下文下，比较chosen回答token序列与rejected回答token序列，哪一个更应被模型偏好。

### 四、双模型设计的核心原因
`train_dpo.py` 中初始化了两个模型：
1. **model**：策略模型，为训练更新的目标模型
2. **ref_model**：参考模型，仅用于打分，不参与训练

两个模型初始权重均来自同一个 `from_weight`。

两个模型的分工：
1. 策略模型负责学习“更偏好chosen回复”
2. 参考模型提供稳定的基准线，避免策略模型过度偏移

该设计为DPO的核心思想。
若无参考模型，策略模型只会一味拉高chosen回复的概率，易出现过度偏移，甚至破坏原SFT模型学到的通用能力。参考模型的核心作用是提供“原始偏好基线”。

`train_dpo.py` 中参考模型被设置为eval模式，且执行 `requires_grad_(False)`，即完全冻结，仅负责前向计算。

### 五、chosen与rejected拼接batch的设计原因
`train_dpo.py` 中，训练阶段会将以下字段沿batch维度拼接：
   - `x_chosen` 与 `x_rejected`
   - `y_chosen` 与 `y_rejected`
   - `mask_chosen` 与 `mask_rejected`

即batch的前半部分为chosen样本，后半部分为rejected样本。

该设计的核心优势：
1. 一次前向传播即可同时获取chosen与rejected的logits
2. 计算效率更高
3. 后续拆分逻辑更简洁

因此该DPO训练并非“先跑chosen再跑rejected”，而是将两者合并为一个batch一次性完成前向计算。

### 六、logits_to_log_probs 函数的核心作用
该函数定义在 `train_dpo.py` 中，核心作用为：将模型输出的logits，转换为“目标token的对数概率”。

执行步骤：
1. 对词表维度执行 `log_softmax`，得到每个位置每个词的对数概率
2. 通过真实标签y执行gather操作，提取每个位置目标token对应的对数概率

输出结果的形状为：`batch_size × seq_len`，含义为序列中每个位置，模型对“正确下一个token”的对数概率。

该步骤为DPO的核心前置操作：DPO不直接使用完整词表分布，仅关注真实回答路径上每一步token的概率，即衡量模型对该条回答路径的整体认可程度。

### 七、dpo_loss 的设计逻辑
核心函数定义在 `train_dpo.py` 中。

核心思路：先通过mask过滤掉非参与位置，仅保留assistant回答部分，对每条序列的对数概率做平均：
$$
\log p(\text{response}) = \frac{1}{|M|} \sum_t M_t \log p(y_t \mid x, y_{<t})
$$

其中 $M_t$ 为loss mask，即一条回答最终会被压缩为一个标量，代表模型对该条回答的整体认可程度。

基于该标量，得到四个核心值：
1. **chosen_policy_log_probs**：策略模型对chosen回复的总分
2. **reject_policy_log_probs**：策略模型对rejected回复的总分
3. **chosen_ref_log_probs**：参考模型对chosen回复的总分
4. **reject_ref_log_probs**：参考模型对rejected回复的总分

计算两个对数概率差值：
$$
\pi_{\text{logratio}} = \log p_{\pi}(chosen) - \log p_{\pi}(rejected)
$$

$$
ref_{\text{logratio}} = \log p_{ref}(chosen) - \log p_{ref}(rejected)
$$

计算最终差值：
$$
z = \pi_{\text{logratio}} - ref_{\text{logratio}}
$$

最终损失函数：
$$
L_{DPO} = -\log \sigma(\beta z)
$$

该损失函数的核心含义：
若策略模型相对于参考模型，更明显地偏好chosen回复而非rejected回复，则z值更大，损失值更小；若策略模型未体现出该偏好，损失值会更大。

### 八、beta 参数的核心作用
beta为DPO的核心温度参数，在脚本中通过 `train_dpo.py` 的命令行参数传入。

beta参数控制的是：策略模型偏离参考模型偏好的力度。

直观理解：
1. **beta值越小**：参数更新越保守，模型越不容易偏离参考模型
2. **beta值越大**：参数更新越激进，模型对chosen与rejected的区分度越强

beta值并非越大越好，过大的beta值会导致模型过度强化偏好，更易遗忘原有的通用能力。
该实现默认设置为0.1，属于行业通用的偏稳妥取值范围。

### 九、DPO与SFT训练目标的核心差异
SFT的核心目标为：给定上下文，最大化标准答案token序列的概率。
数学表达式：
$$
\max \log p(y_{chosen} \mid x)
$$

DPO的核心目标并非单独最大化chosen回复的概率，而是最大化chosen回复相对于rejected回复的优势，且该优势是相对于参考模型定义的。

即DPO的核心不是“chosen回复的概率大不大”，而是“策略模型是否比参考模型更偏爱chosen回复胜过rejected回复”。

因此DPO属于相对偏好学习，而非绝对监督学习。