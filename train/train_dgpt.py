import torch
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from datasets import load_from_disk
from transformers import AutoTokenizer
import os
import wandb
import sys
import argparse

block_size = 512


def patch_gpt2_attention_sdpa():
    """用 PyTorch SDPA 替换 GPT2 原生 eager attention，融合 kernel 避免物化注意力矩阵"""
    @functools.wraps(GPT2Attention._attn)
    def _attn_sdpa(self, query, key, value, attention_mask=None, head_mask=None):
        return (
            torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            ),
            None,
        )
    GPT2Attention._attn = _attn_sdpa

def group_texts(examples):
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
    
    wandb.init(project=exp_name, mode="offline")

    # --- SDPA monkey-patch（必须在模型加载之前） ---
    patch_gpt2_attention_sdpa()

    model_id = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()
    #print(len(vocab))
    special_tokens = {
        'bos_token': '<|startoftext|>',
        'additional_special_tokens': ['<|sp1|>', '<|sp2|>']
    }
    
    _ = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    
    train_data = load_from_disk(dataset_path)
    
    train_dataset = train_data.map(group_texts, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(vocab))

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_path,
        logging_strategy="steps",
        logging_steps=100,
        log_level="info",                         # 输出 INFO 级别日志（含 loss/lr）
        learning_rate=6.25e-5,
        weight_decay=0.01,
        num_train_epochs=n_epochs,
        skip_memory_metrics=True,
        per_device_train_batch_size=8,
        save_strategy='epoch',
        report_to="wandb",
        ddp_find_unused_parameters=False,
        # --- 性能优化 ---
        bf16=True,                            # bfloat16 混合精度（减少显存，加速计算）
        optim="adamw_torch_fused",            # CUDA 融合 AdamW kernel
        dataloader_pin_memory=True,           # 锁页内存加速 CPU→GPU 传输
        # group_by_length 已移除: 所有 block 统一 512 token，按长度分组无意义
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