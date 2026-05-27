"""
步骤4: 转换为 HuggingFace Dataset 格式
========================================
将评分后的数据转换为 HuggingFace Dataset，供模型训练使用。
支持两种模型格式：

1. DialoGPT (dgpt):
   - 使用特殊 token 标记说话者（<|sp1|> 用户, <|sp2|> Bot）
   - 将 persona、对话历史、评分和回复拼接为一个 token 序列
   - 输出格式: <|startoftext|> persona <|sp1|> 用户输入 <|sp2|> Bot回复 ...
     Score: 分数 <|sp2|> Bot: 回复 <|endoftext|>

2. Llama (llama):
   - 将 persona 转为第二人称（"I am" -> "you are"）
   - 使用标准的 chat template 格式（system/user/assistant 消息列表）
   - 在最后一条 user 消息中附加评分信息（"Score: 分数"）

输入: {split}_scores.json（步骤3的输出）
输出: dgpt_train_final/ 或 llama_train_final/ 目录（HuggingFace Dataset 磁盘格式）
"""

import json
import sys
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer


def concatenate_and_tokenize(example, tokenizer, vocab):
    """
    将 persona、对话历史、评分和回复拼接为 DialoGPT 格式的单个 token 序列。

    DialoGPT 格式:
    <|startoftext|> persona_text <|sp1|> User: msg <|sp2|> Bot: reply ... Score: 0.8 <|sp2|> Bot: final_reply <|endoftext|>
    """
    # 起始 token
    example['input_ids'] = [vocab['<|startoftext|>']]
    example['input_ids'].extend(tokenizer(example['persona'])['input_ids'])

    # 交错插入对话历史，用特殊 token 区分说话者
    for i in range(len(example['history'])):
        if i % 2 == 0:
            # 偶数索引 = 用户发言，用 <|sp1|> 标记
            example['input_ids'].append(vocab['<|sp1|>'])
        else:
            # 奇数索引 = Bot 发言，用 <|sp2|> 标记
            example['input_ids'].append(vocab['<|sp2|>'])
        example['input_ids'].extend(tokenizer(example['history'][i])['input_ids'])

    # 附加评分和回复
    example['input_ids'].extend(tokenizer("Score: " + str(example['score']))['input_ids'])
    example['input_ids'].append(vocab['<|sp2|>'])
    example['input_ids'].extend(tokenizer("Bot: " + example['response'])['input_ids'])
    example['input_ids'].append(vocab['<|endoftext|>'])

    return example


def dpgt_dataset(data):
    """
    构建 DialoGPT 格式的训练数据集。

    数据增强逻辑:
    - 原始回复的评分固定为 1.0（最高分）
    - 对于每个原始回复，将其对应的所有 augmented 变体加入训练集
    - 变体的评分来自步骤3的 BERTScore 结果（< 1.0）
    - 这样模型可以同时学习高质量回复（高分）和低质量回复（低分）的区别
    """
    personas, histories, responses, scores = [], [], [], []

    for chat in tqdm(data):
        # 构建 persona 文本（拼接所有人设句子）
        persona = "Your persona:"
        for p in chat['persona']:
            persona += " " + p

        # 每个回复都对应相同的 persona
        personas.extend([persona] * len(chat['responses']))

        # 将 query 和 response 交错合并为完整对话历史
        full_dialog = [
            x for xs in zip(chat['queries'], chat['responses'])
            for x in xs
        ]

        # 为每个 response 构建其对应的对话历史（包含之前的所有轮次）
        for resp in chat['responses']:
            idx = full_dialog.index(resp)
            hist = []
            temp = full_dialog[:idx]  # 当前回复之前的所有内容
            for i in range(len(temp)):
                if i % 2 == 0:
                    hist.append("User: " + temp[i])
                else:
                    hist.append("Bot: " + temp[i])
            histories.append(hist)

        # 原始回复
        responses.extend(chat['responses'])
        scores.extend([1.0] * len(chat['responses']))  # 原始回复质量最高，分数为1

        # 如果有 augmented 数据（mask 变体），一并加入
        if len(chat['aug_data']) == 0:
            continue

        for aug in chat['aug_data']:
            for masked in aug['masked']:
                # 复用当前 conversation 第一个 response 的 persona 和 history
                # 同一 conversation 的所有 response 共享同一 persona
                base_idx = len(personas) - len(chat['responses'])
                personas.append(personas[base_idx])
                histories.append(histories[base_idx])
                responses.append(masked['sent'])
                scores.append(masked['score'])

        # 确保所有列表长度一致
        assert len(personas) == len(responses) == len(histories) == len(scores), \
            "Check code"

    # 创建 HuggingFace Dataset
    ds = Dataset.from_dict({
        'persona': personas,
        'history': histories,
        'response': responses,
        'score': scores
    })

    # 初始化 DialoGPT tokenizer 并添加自定义特殊 token
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    vocab = tokenizer.get_vocab()
    special_tokens = {
        'bos_token': '<|startoftext|>',
        'additional_special_tokens': ['<|sp1|>', '<|sp2|>']  # sp1=用户, sp2=Bot
    }
    _ = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()

    # 对数据集进行 tokenize 和拼接
    kwargs = {
        "tokenizer": tokenizer,
        "vocab": vocab
    }
    processed_train_data = ds.map(
        concatenate_and_tokenize,
        remove_columns=ds.column_names,  # 移除原始文本列，只保留 input_ids
        num_proc=4,                       # 4进程并行处理
        fn_kwargs=kwargs
    )

    # 保存到磁盘
    processed_train_data.save_to_disk("dgpt_train_final")


def llama_dataset(data):
    """
    构建 Llama 3.1 格式的训练数据集。

    数据增强逻辑:
    - 与 DialoGPT 版本相同：原始回复分数=1.0，变体分数=BERTScore
    - 主要的区别在于格式转换：
      1. persona 转为第二人称（"I" -> "you"），作为 system message
      2. 对话历史拆分为 user/assistant 消息
      3. 在最后一个 user 消息中附加评分
    """
    personas, histories, responses, scores = [], [], [], []

    for chat in tqdm(data):
        # 将 persona 转为第二人称并拼接
        # 例如 "I'm a teacher" -> "you are a teacher"
        persona = ""
        for p in chat['persona']:
            p = p.replace("i'm", "you are")
            p = p.replace("i'll", "you'll")
            p = p.replace("i am ", "you are ")
            p = p.replace("i was", "you were")
            p = p.replace("i've", "you have")
            p = p.replace("my", "your")
            p = p.replace("i ", "you ")
            p = p.replace(" me ", " you ")
            persona += p + " "

        personas.extend([persona] * len(chat['responses']))

        full_dialog = [
            x for xs in zip(chat['queries'], chat['responses'])
            for x in xs
        ]

        for resp in chat['responses']:
            idx = full_dialog.index(resp)
            hist = []
            temp = full_dialog[:idx]
            for i in range(len(temp)):
                if i % 2 == 0:
                    hist.append(temp[i])    # 用户输入（不加 "User: " 前缀）
                else:
                    hist.append(temp[i])    # Bot 回复（不加 "Bot: " 前缀）
            histories.append(hist)

        responses.extend(chat['responses'])
        scores.extend([1.0] * len(chat['responses']))

        for aug in chat['aug_data']:
            for masked in aug['masked']:
                base_idx = len(personas) - len(chat['responses'])
                personas.append(personas[base_idx])
                histories.append(histories[base_idx])
                responses.append(masked['sent'])
                scores.append(masked['score'])

        assert len(personas) == len(responses) == len(histories) == len(scores), \
            "Check code"

    # 构建 Llama 3.1 的 chat 格式消息列表
    outputs = []
    for idx in tqdm(range(len(responses))):
        messages = []

        # System message: 第二人称的 persona
        messages.append({
            "role": "system",
            "content": personas[idx]
        })

        # 对话历史: 交替 user / assistant
        for i in range(len(histories[idx])):
            if i == (len(histories[idx]) - 1):
                # 最后一条 user 消息附带评分信息
                messages.append({
                    'role': 'user',
                    'content': histories[idx][i] + " Score: " + str(scores[idx])
                })
            elif i % 2 == 0:
                messages.append({
                    'role': 'user',
                    'content': histories[idx][i]
                })
            else:
                messages.append({
                    'role': 'assistant',
                    'content': histories[idx][i]
                })

        # Assistant 的最终回复
        messages.append({
            'role': 'assistant',
            'content': responses[idx]
        })

        outputs.append(messages)

    # 使用 Llama 3.1 的 chat template 进行 tokenize
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    all_inputs = []
    for dialog in tqdm(messages):
        # apply_chat_template 会自动添加 <|begin_of_text|>、<|start_header_id|> 等特殊 token
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        all_inputs.append(dialog_tokens)

    processed_data = Dataset.from_dict({
        'input_ids': all_inputs
    })

    # 保存到磁盘
    processed_data.save_to_disk("llama_train_final")


def main(file_path, model_name):
    """主函数：加载评分数据 -> 选择模型格式 -> 生成 HuggingFace Dataset"""

    # 加载步骤3生成的评分数据
    with open(f'{file_path}', 'r') as f:
        data = json.loads(f.read())

    if model_name == "dgpt":
        dpgt_dataset(data)
    elif model_name == "llama":
        llama_dataset(data)
    else:
        print("Invalid model name!")


if __name__ == "__main__":
    # 命令行参数:
    #   argv[1]: 步骤3的评分结果文件路径
    #   argv[2]: 目标模型格式 ("dgpt" 或 "llama")
    main(sys.argv[1], sys.argv[2])
