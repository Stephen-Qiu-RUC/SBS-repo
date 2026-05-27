"""最小 benchmark：测试 DialoGPT-small 在 RTX 5060 上的原始训练速度"""
import torch
import time
from transformers import AutoModelForCausalLM

device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-small",
    torch_dtype=torch.bfloat16,
).to(device)
model.train()

# 模拟训练: batch=16, seq=512, 与训练代码一致
B, S = 16, 512
x = torch.randint(0, 50000, (B, S), device=device)
opt = torch.optim.AdamW(model.parameters(), lr=6.25e-5)

# warmup
for _ in range(3):
    loss = model(x, labels=x).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
torch.cuda.synchronize()

# benchmark
N = 20
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    loss = model(x, labels=x).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
torch.cuda.synchronize()
t1 = time.time()

avg = (t1 - t0) / N
print(f"BF16 每步耗时: {avg:.2f}s")
print(f"预估 15 epoch: {avg * 38445 / 3600:.0f} 小时")

# FP32 对比
model_fp32 = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-small"
).to(device)
model_fp32.train()
opt32 = torch.optim.AdamW(model_fp32.parameters(), lr=6.25e-5)

for _ in range(3):
    loss = model_fp32(x, labels=x).loss
    loss.backward()
    opt32.step()
    opt32.zero_grad()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N):
    loss = model_fp32(x, labels=x).loss
    loss.backward()
    opt32.step()
    opt32.zero_grad()
torch.cuda.synchronize()
t1 = time.time()

avg_fp32 = (t1 - t0) / N
print(f"FP32 每步耗时: {avg_fp32:.2f}s")
