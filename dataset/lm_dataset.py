"""
预训练数据集。

这个文件负责把原始文本样本转换成模型能直接吃进去的张量：
1. 读取 json/jsonl 数据。
2. 用 tokenizer 把文本切成 token id。
3. 在序列首尾手动加上 bos/eos。
4. 右侧补 pad，让每条样本长度一致。
5. 生成 labels，供自回归语言模型计算 next-token loss。
"""

from torch.utils.data import Dataset
import torch
import os
import random
import json
from datasets import load_dataset

# 关闭 tokenizer 的并行提示，避免 DataLoader 多进程时频繁打印警告信息。
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是cookmind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是cookmind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are cookmind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are cookmind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    """
    预训练阶段使用的数据集。

    参数说明：
    - data_path: 训练数据路径，通常是 jsonl 文件。
    - tokenizer: 分词器，负责把字符串映射成 token id。
    - max_length: 每条样本最终统一到的最大长度。

    这里采用的是“固定长度 + 右侧 padding”的做法。
    这样 DataLoader 拼 batch 时最简单，模型前向时张量形状也固定。
    """

    def __init__(self,data_path,tokenizer,max_length=512):
        super().__init__()
        # 保存分词器，后面每次取样本时都要调用它完成文本编码。
        self.tokenizer=tokenizer

        # 统一的序列长度上限。
        # 如果原始文本太长，会被截断；如果太短，会在右侧补 pad。
        self.max_length=max_length

        # 通过 Hugging Face datasets 读取 json/jsonl 数据。
        # split='train' 表示把整个文件当作训练集来读。
        self.samples=load_dataset('json',data_files=data_path,split='train')

    def __len__(self):
        # Dataset 必须实现 __len__，这样 DataLoader 才知道一共有多少条样本。
        return len(self.samples)
    
    def __getitem__(self, index):
        # 先取出第 index 条原始样本。
        sample=self.samples[index]

        # 取出文本字段并转成字符串。
        # 有些数据可能不是纯 str，这里统一包一层 str() 更稳妥。

        # add_special_tokens=False 的意思是：
        # 不让 tokenizer 自动添加特殊 token。
        # 因为下面我们会手动添加 bos/eos，避免和 tokenizer 的默认行为冲突。

        # max_length=self.max_length-2 是因为我们要给 bos 和 eos 预留两个位置。
        # 如果这里不减 2，后面再手动加特殊 token 就可能超过最大长度。
        tokens=self.tokenizer(str(sample['text']),add_special_tokens=False,max_length=self.max_length-2,truncation=True).input_ids

        # 手动在开头加 bos，在结尾加 eos。
        # bos: beginning of sequence，表示序列开始。
        # eos: end of sequence，表示序列结束。
        tokens=[self.tokenizer.bos_token_id]+tokens+[self.tokenizer.eos_token_id]

        # 右侧补 pad 到固定长度。
        # pad_token_id 不代表真实文本，只是为了把不同长度的样本补成同样长。
        input_ids=tokens+[self.tokenizer.pad_token_id]*(self.max_length-len(tokens))

        # 转成 LongTensor，因为 embedding 查表和交叉熵标签都要求整型 token id。
        input_ids=torch.tensor(input_ids,dtype=torch.long)

        # 对自回归语言模型来说，labels 一般直接拷贝 input_ids。
        # 后面模型内部会自动做一个时间步的错位：
        # 用前面的 token 去预测后面的 token。
        labels=input_ids.clone()

        # pad 位置不应该参与 loss 计算。
        # 在 PyTorch 的 cross_entropy 里，ignore_index=-100 表示“跳过这些位置”。
        # 所以凡是 pad 的地方，我们都把标签改成 -100。
        labels[input_ids==self.tokenizer.pad_token_id]=-100

        # 返回一条训练样本：
        # - input_ids: 输入给模型的 token 序列
        # - labels:    用来计算 next-token loss 的监督信号
        return input_ids,labels
    
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        #滑动窗口寻找每轮对话的起止位置，给 assistant 说的话生成标签，user 说的话保持 -100。
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
