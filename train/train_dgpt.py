import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import itertools
import functools
import math
import os
import sys

block_size = 512


def patch_gpt2_attention_sdpa():
    """
    Monkey-patch GPT2Attention._attn to use PyTorch SDPA instead of the
    default "eager" implementation.  Eager materializes the full
    [batch*heads, seq, seq] attention matrix and launches separate CUDA
    kernels — severely memory-bandwidth-bound on consumer GPUs.  SDPA
    fuses these into one kernel and avoids materializing the matrix.
    """
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
    """
    Concatenate tokenized samples and chunk into fixed-length blocks for
    autoregressive training.  Uses itertools.chain to avoid the O(n^2)
    behaviour of sum(list, []).
    """
    concatenated = {
        k: list(itertools.chain.from_iterable(examples[k]))
        for k in examples.keys()
    }
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result


def main(exp_name, dataset_path, n_epochs, output_path):
    wandb.init(project=exp_name, mode="offline")

    # --- SDPA monkey-patch (must happen before model init) ---
    patch_gpt2_attention_sdpa()

    model_id = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    special_tokens = {
        'bos_token': '<|startoftext|>',
        'additional_special_tokens': ['<|sp1|>', '<|sp2|>']
    }
    _ = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()

    # Load + chunk + materialise
    train_data = load_from_disk(dataset_path)
    tmp_path = "/tmp/dgpt_train_grouped"
    train_data.map(group_texts, batched=True).save_to_disk(tmp_path)
    train_dataset = load_from_disk(tmp_path)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(vocab))
    model = model.cuda()

    ckpt_path = os.path.join(output_path, "checkpoint_latest.pt")
    start_epoch = 0

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1

    # JIT-compile for operator fusion  (after potential state-dict load so the
    # inductor sees the correct weights from the start).
    model = torch.compile(model, mode="default")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=6.25e-5, weight_decay=0.01, fused=True)
    total_steps = n_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    if start_epoch > 0:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    model.train()
    steps_per_epoch = len(dataloader)

    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", unit="step",
                    file=sys.stdout, dynamic_ncols=True)
        for batch in pbar:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch)

            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = out.loss.item()
            epoch_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / steps_per_epoch
        tqdm.write(f"Epoch {epoch + 1}/{n_epochs}  avg_loss={avg_loss:.4f}  "
                   f"lr={scheduler.get_last_lr()[0]:.2e}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})

        # Atomic checkpoint: write to temp file first, then rename, so a crash
        # mid-write never leaves a corrupted checkpoint.
        os.makedirs(output_path, exist_ok=True)
        ckpt_tmp = os.path.join(output_path, "checkpoint_latest.tmp")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
        }, ckpt_tmp)
        os.replace(ckpt_tmp, ckpt_path)

        torch.cuda.empty_cache()

    # Final model export
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="", type=str)
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--n_epochs", default=None, type=int)
    parser.add_argument("--output_path", default="", type=str)
    args = parser.parse_args()

    main(args.exp_name, args.dataset_path, args.n_epochs, args.output_path)
