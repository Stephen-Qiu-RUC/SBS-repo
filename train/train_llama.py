import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig
from dataclasses import asdict
from llama_recipes.configs import lora_config as LORA_CONFIG
import wandb
from datasets.utils.logging import disable_progress_bar
import argparse

block_size = 2048

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

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    disable_progress_bar()

    train_data = load_from_disk(dataset_path)
    train_dataset = train_data.map(group_texts, batched=True)

    # 使用 float16 加载模型以降低显存占用，8B 模型全精度需要 ~32GB 显存
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    # LoRA 低秩适配：r=8 控制低秩矩阵维度，alpha=32 缩放适配器权重
    lora_config = LORA_CONFIG()
    lora_config.r = 8
    lora_config.lora_alpha = 32
    peft_config = LoraConfig(**asdict(lora_config))
    # 将原始模型包装为 PEFT 模型，仅训练 LoRA 适配器参数
    model = get_peft_model(model, peft_config)

    #model.print_trainable_parameters()

    # Seq2Seq collator 会自动处理 padding 和 labels 的 -100 掩码
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=output_path,
        learning_rate=3e-4,
        weight_decay=0.0,
        num_train_epochs=n_epochs,
        fp16=True,  # 混合精度训练，加速计算并节省显存
        skip_memory_metrics=True,
        per_device_train_batch_size=1,  # 8B 模型 + LoRA 在单卡上通常只能 batch_size=1
        group_by_length=True,  # 将相近长度的样本分到同一 batch，减少 padding 浪费
        save_strategy='epoch',  # 每 epoch 保存一次 checkpoint
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    # 仅训练 LoRA 权重，保存时会自动合并或单独保存 adapter
    #trainer.save_model(output_path)

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