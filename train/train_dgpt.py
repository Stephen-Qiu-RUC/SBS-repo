from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_from_disk
from transformers import AutoTokenizer
import os
import wandb
import sys
import argparse

block_size = 512

def group_texts(examples):
    """
    将分词后的样本拼接并分块为固定长度，用于因果语言模型的自回归训练。
    每个块向后偏移 1 个 token 作为 labels。
    """
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size + 1

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(exp_name, dataset_path, n_epochs, output_path):

    # 离线模式记录训练指标，避免联网需求
    wandb.init(project=exp_name, mode="offline")

    model_id = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()
    # 添加对话分隔特殊 token：<|sp1|> 标记说话人切换，<|sp2|> 标记对话轮次边界
    special_tokens = {
        'bos_token': '<|startoftext|>',
        'additional_special_tokens': ['<|sp1|>', '<|sp2|>']
    }

    _ = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()

    train_data = load_from_disk(dataset_path)

    train_dataset = train_data.map(group_texts, batched=True)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    # 因为添加了特殊 token，需要扩展 embedding 矩阵
    model.resize_token_embeddings(len(vocab))

    # Seq2Seq collator 会自动处理 padding 和 labels 的 -100 掩码
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=output_path,
        logging_strategy="epoch",
        learning_rate=6.25e-5,
        weight_decay=0.01,
        num_train_epochs=n_epochs,
        skip_memory_metrics=True,
        per_device_train_batch_size=16,
        save_strategy='no',  # 不保存中间 checkpoint，仅训练结束时保存最终模型
        group_by_length=True,  # 将相近长度的样本分到同一 batch，减少 padding 浪费
        report_to="wandb",
        ddp_find_unused_parameters=False  # 多卡训练时跳过未使用参数检测，加速启动
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--exp_name", default="", type=str)
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--n_epochs", default=None, type=int)
    parser.add_argument("--output_path", default="", type=str)

    args = parser.parse_args()
    exp_name = args.exp_name
    dataset_path = args.dataset_path
    n_epochs = args.n_epochs
    output_path = args.output_path

    main(exp_name, dataset_path, n_epochs, output_path)