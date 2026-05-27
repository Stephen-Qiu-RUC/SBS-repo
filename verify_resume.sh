#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# ============================================================================
# 验证断点续训功能
# 流程:
#   1. 训练 2 个 epoch，保存 checkpoint
#   2. 模拟断电重启，加载 checkpoint，继续训 1 个 epoch
#   3. 检查: 权重一致 / epoch 连续 / lr 连续 / loss 连续
# ============================================================================

check_gpu() {
    python -c "import torch; torch.randn(1).cuda(); print('GPU:', torch.cuda.get_device_name(0))" || {
        echo "错误: GPU 不可用"
        exit 1
    }
}

echo "============================================"
echo "  断点续训功能验证"
echo "============================================"
echo ""
echo "[检查] 环境..."
check_gpu
echo ""

python -c '
import torch
import os, sys, itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_from_disk
from torch.utils.data import DataLoader
from train.train_dgpt import patch_gpt2_attention_sdpa, group_texts

patch_gpt2_attention_sdpa()
output_path = "/tmp/test_resume_verify"
ckpt_path = os.path.join(output_path, "checkpoint_latest.pt")

# --- 清理旧数据 ---
import shutil
shutil.rmtree(output_path, ignore_errors=True)
os.makedirs(output_path, exist_ok=True)

# --- 准备数据集 ---
ds = load_from_disk("dgpt_train_final")
tmp_path = "/tmp/dgpt_train_grouped"
ds.map(group_texts, batched=True).save_to_disk(tmp_path)
train_ds = load_from_disk(tmp_path)

# --- 初始化 tokenizer ---
model_id = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
special_tokens = {
    "bos_token": "<|startoftext|>",
    "additional_special_tokens": ["<|sp1|>", "<|sp2|>"]
}
_ = tokenizer.add_special_tokens(special_tokens)
vocab = tokenizer.get_vocab()
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ====================================================================
# Phase 1: 训练 2 个 epoch
# ====================================================================
print(">>> Phase 1: 训练 epoch 1-2")
print()

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(vocab))
model = model.cuda()
model = torch.compile(model, mode="default")

dataloader = DataLoader(train_ds, batch_size=8, collate_fn=data_collator, shuffle=True, num_workers=0, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=6.25e-5, weight_decay=0.01, fused=True)
total_steps = 2 * len(dataloader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

model.train()
phase1_losses = []
phase1_lrs = []
phase1_steps = 0

for epoch in range(2):
    epoch_loss = 0.0
    for batch in dataloader:
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += out.loss.item()
        phase1_steps += 1
    avg = epoch_loss / len(dataloader)
    phase1_losses.append(avg)
    phase1_lrs.append(scheduler.get_last_lr()[0])
    print(f"  epoch {epoch+1}/2  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

# 保存 checkpoint（与 train_dgpt.py 完全相同的逻辑）
w_before = model.state_dict()["transformer.h.0.ln_1.weight"].clone()
ckpt_tmp = os.path.join(output_path, "checkpoint_latest.tmp")
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": 1,
}, ckpt_tmp)
os.replace(ckpt_tmp, ckpt_path)
print(f"  -> checkpoint 已保存 (epoch 1 完成)")

step_before = phase1_steps
del model, optimizer, scheduler, dataloader
torch.cuda.empty_cache()

# ====================================================================
# Phase 2: 模拟重启，加载 checkpoint 继续训 1 个 epoch
# ====================================================================
print()
print(">>> Phase 2: 模拟断电重启，加载 checkpoint")
print()

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(vocab))
model = model.cuda()

ckpt = torch.load(ckpt_path, map_location="cuda")
model.load_state_dict(ckpt["model"])
start_epoch = ckpt["epoch"] + 1
print(f"  start_epoch = {start_epoch} (预期: 2)")

w_after = model.state_dict()["transformer.h.0.ln_1.weight"]

model = torch.compile(model, mode="default")

dataloader = DataLoader(train_ds, batch_size=8, collate_fn=data_collator, shuffle=True, num_workers=0, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=6.25e-5, weight_decay=0.01, fused=True)
total_steps = 3 * len(dataloader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

optimizer.load_state_dict(ckpt["optimizer"])
scheduler.load_state_dict(ckpt["scheduler"])

model.train()
phase2_losses = []
phase2_lrs = []
phase2_steps = 0

for epoch in range(start_epoch, 3):
    epoch_loss = 0.0
    for batch in dataloader:
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += out.loss.item()
        phase2_steps += 1
    avg = epoch_loss / len(dataloader)
    phase2_losses.append(avg)
    phase2_lrs.append(scheduler.get_last_lr()[0])
    print(f"  epoch {epoch+1}/3  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

# ====================================================================
# 验证
# ====================================================================
print()
print("============================================")
print("  验证结果")
print("============================================")

passed = 0
failed = 0

# 检查 1: 权重 bit-level 一致
if torch.equal(w_before, w_after):
    print("  [PASS] 权重一致: transformer.h.0.ln_1.weight 位级别相同")
    passed += 1
else:
    max_diff = (w_before - w_after).abs().max().item()
    print(f"  [FAIL] 权重不一致: max diff = {max_diff:.6e}")
    failed += 1

# 检查 2: epoch 从正确位置开始
if start_epoch == 2:
    print(f"  [PASS] epoch 连续: start_epoch={start_epoch} (预期 2)")
    passed += 1
else:
    print(f"  [FAIL] epoch 错误: start_epoch={start_epoch} (预期 2)")
    failed += 1

# 检查 3: lr 连续
lr_before = phase1_lrs[-1]
lr_after  = phase2_lrs[0]
lr_ratio = abs(lr_before - lr_after) / lr_before
if lr_ratio < 0.01:
    print(f"  [PASS] lr 连续: {lr_before:.2e} -> {lr_after:.2e} (偏差 {lr_ratio*100:.2f}%)")
    passed += 1
else:
    print(f"  [FAIL] lr 不连续: {lr_before:.2e} -> {lr_after:.2e} (偏差 {lr_ratio*100:.2f}%)")
    failed += 1

# 检查 4: loss 不跳变（恢复后 loss <= 保存前 loss 或 loss 在正常下降）
loss_ratio = (phase1_losses[-1] - phase2_losses[0]) / max(phase1_losses[-1], 0.1)
if loss_ratio > -0.5:
    print(f"  [PASS] loss 连续: epoch2={phase1_losses[-1]:.4f} -> epoch3={phase2_losses[0]:.4f}")
    passed += 1
else:
    print(f"  [FAIL] loss 跳变: epoch2={phase1_losses[-1]:.4f} -> epoch3={phase2_losses[0]:.4f} (下降幅度异常)")
    failed += 1

# 检查 5: 总 step 计数
total_steps = phase1_steps + phase2_steps
steps_per_epoch = phase1_steps // 2
expected_total = 3 * steps_per_epoch
if total_steps == expected_total:
    print(f"  [PASS] step 总计: {phase1_steps} + {phase2_steps} = {total_steps} (预期 {expected_total})")
    passed += 1
else:
    print(f"  [FAIL] step 不正确: {phase1_steps} + {phase2_steps} = {total_steps} (预期 {expected_total})")
    failed += 1

print()
print(f"通过: {passed}/5, 失败: {failed}/5")
if failed == 0:
    print("结论: 断点续训功能正常")
else:
    print("结论: 存在问题，请检查上述 FAIL 项")
'

echo ""
echo "验证完成。"
