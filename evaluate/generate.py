"""
推理与生成脚本
================
加载训练好的 DialoGPT/Llama 模型，在测试集上生成回复。
支持不同 score 值（论文 Table 4）和去掉 score（论文 Table 3 消融实验）。

用法:
    # 默认: 加载 dgpt 模型, score=1.0, 在 personachat test 集上生成
    python evaluate/generate.py \
        --model_path models/personachat_dgpt \
        --test_data test_scores.json \
        --output outputs/gen_dgpt_score1.0.json

    # 多个 score 值（论文 Table 4 风格）
    python evaluate/generate.py \
        --model_path models/personachat_dgpt \
        --test_data test_scores.json \
        --scores 1.0 0.95 0.9 0.85 0.8 0.75 \
        --output outputs/gen

    # 去掉 score in prompt（论文 Table 3 消融）
    python evaluate/generate.py \
        --model_path models/personachat_dgpt \
        --test_data test_scores.json \
        --no_score \
        --output outputs/gen_no_score.json
"""

import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_dgpt_prompt(persona_sentences, queries, responses, turn_idx,
                       score=None, tokenizer=None, vocab=None):
    """
    构建 DialoGPT 格式的输入序列（到 "Bot: " 为止，模型从此处开始生成）。

    训练时的格式:
    <|startoftext|>Your persona: ... <|sp1|>User: q1 <|sp2|>Bot: r1
    ... <|sp1|>User: q_n Score: S <|sp2|>Bot: r_n <|endoftext|>

    推理时去掉最后的 response 和 <|endoftext|>:
    ... <|sp1|>User: q_n Score: S <|sp2|>Bot:
    """
    persona_text = "Your persona: " + " ".join(persona_sentences)

    tokens = [vocab['<|startoftext|>']]
    tokens.extend(tokenizer(persona_text)['input_ids'])

    # 对话历史: 交错插入 user/bot 轮次
    for i in range(turn_idx):
        tokens.append(vocab['<|sp1|>'])
        tokens.extend(tokenizer("User: " + queries[i])['input_ids'])
        tokens.append(vocab['<|sp2|>'])
        tokens.extend(tokenizer("Bot: " + responses[i])['input_ids'])

    # 当前轮次的用户输入
    tokens.append(vocab['<|sp1|>'])
    tokens.extend(tokenizer("User: " + queries[turn_idx])['input_ids'])

    # Score（可选）
    if score is not None:
        tokens.extend(tokenizer(f"Score: {score}")['input_ids'])

    # Bot 回复开始标记
    tokens.append(vocab['<|sp2|>'])
    tokens.extend(tokenizer("Bot:")['input_ids'])

    return tokens


def build_llama_prompt(persona_sentences, queries, responses, turn_idx,
                        score=None, tokenizer=None):
    """
    构建 Llama 3.1 chat 格式的输入。

    训练时的格式:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    <persona in 2nd person>
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    q1<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    r1<|eot_id|>...<|start_header_id|>user<|end_header_id|>
    q_n Score: S<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    r_n<|eot_id|>
    """
    # Persona 转第二人称
    persona = ""
    for p in persona_sentences:
        p = p.replace("i'm", "you are")
        p = p.replace("i'll", "you'll")
        p = p.replace("i am ", "you are ")
        p = p.replace("i was", "you were")
        p = p.replace("i've", "you have")
        p = p.replace("my", "your")
        p = p.replace("i ", "you ")
        p = p.replace(" me ", " you ")
        persona += p + " "

    messages = [{"role": "system", "content": persona.strip()}]

    for i in range(turn_idx):
        messages.append({"role": "user", "content": queries[i]})
        messages.append({"role": "assistant", "content": responses[i]})

    # 当前轮次的用户输入 + score
    user_content = queries[turn_idx]
    if score is not None:
        user_content += f" Score: {score}"
    messages.append({"role": "user", "content": user_content})

    # apply_chat_template 会生成完整序列（含最后的 assistant 头）
    # 我们需要截掉最后的 assistant 回复部分
    # 实际做法: 先用 chat_template 生成到 assistant 开头，再让模型生成
    full_tokens = tokenizer.apply_chat_template(messages, tokenize=True)
    # 追加 assistant 头标记
    assistant_header = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": ""}],
        tokenize=True
    )
    # 取去掉末尾空回复后的 assistant header 部分
    # 简单做法: 在 messages 后手动加 assistant header tokens
    return full_tokens, assistant_header


def compute_ppl(model, input_ids, target_ids, device):
    """使用 HuggingFace 原生 loss 计算 PPL，将 prompt 部分 mask 掉只计算 target 的困惑度"""
    full_ids = input_ids + target_ids
    input_tensor = torch.tensor([full_ids]).to(device)

    labels = input_tensor.clone()
    labels[0, :len(input_ids)] = -100

    with torch.no_grad():
        outputs = model(input_tensor, labels=labels)
        loss = outputs.loss

    if loss is None or torch.isnan(loss):
        return float('nan')

    return torch.exp(loss).item()


def generate_response(model, tokenizer, prompt_tokens, device,
                      max_new_tokens=64, do_sample=False, temperature=0.7,
                      top_p=0.9, eos_token_id=None):
    """给定 prompt token 序列，生成回复文本"""
    input_tensor = torch.tensor([prompt_tokens]).to(device)
    attention_mask = torch.ones_like(input_tensor)

    gen_kwargs = {
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": eos_token_id or tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": 5,
            "early_stopping": True,
        })

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    # 解码生成部分
    generated_ids = outputs[0][len(prompt_tokens):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def main(model_path, test_data_path, output_path, scores, no_score,
         model_type="dgpt", device="cuda", do_sample=False):
    """
    主流程:
    1. 加载模型和测试数据
    2. 对每组合（dialogue, turn, score）生成回复
    3. 计算 PPL
    4. 保存结果
    """
    # ---- 加载模型 ----
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    if model_type == "dgpt":
        vocab = tokenizer.get_vocab()

    # ---- 加载测试数据 ----
    print(f"Loading test data from {test_data_path} ...")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    print(f"Test data: {len(test_data)} dialogues")

    # ---- 确定要测试的 score 列表 ----
    if no_score:
        score_list = [None]  # None 表示不加 Score: 前缀
    elif scores:
        score_list = [float(s) for s in scores]
    else:
        score_list = [1.0]

    # ---- 生成 ----
    eos_id = tokenizer.eos_token_id
    if model_type == "dgpt":
        # DialoGPT 的 eos 是 <|endoftext|>
        eos_id = vocab.get('<|endoftext|>', tokenizer.eos_token_id)

    for score in score_list:
        score_label = f"score_{score}" if score is not None else "no_score"
        out_file = f"{output_path}_{score_label}.json" if len(score_list) > 1 else output_path

        print(f"\n{'='*60}")
        print(f"Generating with score = {score}")
        print(f"{'='*60}")

        results = []
        total_ppl = []
        skipped = 0

        for chat_idx, chat in enumerate(tqdm(test_data)):
            persona = chat['persona']
            queries = chat['queries']
            responses = chat['responses']

            for turn_idx in range(len(responses)):
                ref_response = responses[turn_idx]

                if model_type == "dgpt":
                    prompt_tokens = build_dgpt_prompt(
                        persona, queries, responses, turn_idx,
                        score=score, tokenizer=tokenizer, vocab=vocab
                    )
                    ref_tokens = tokenizer(" " + ref_response)['input_ids'] + [vocab['<|endoftext|>']]
                elif model_type == "llama":
                    prompt_tokens, assistant_header = build_llama_prompt(
                        persona, queries, responses, turn_idx,
                        score=score, tokenizer=tokenizer
                    )
                    prompt_tokens = prompt_tokens + assistant_header
                    ref_tokens = tokenizer(ref_response)['input_ids'] + [tokenizer.eos_token_id]

                # 生成回复
                try:
                    generated = generate_response(
                        model, tokenizer, prompt_tokens, device,
                        max_new_tokens=64, do_sample=do_sample,
                        eos_token_id=eos_id
                    )
                except Exception as e:
                    print(f"  [WARN] dialogue {chat_idx} turn {turn_idx}: {e}")
                    skipped += 1
                    generated = ""

                # 计算 PPL
                try:
                    ppl = compute_ppl(model, prompt_tokens, ref_tokens, device)
                    total_ppl.append(ppl)
                except Exception:
                    ppl = float('nan')

                results.append({
                    "dialogue_idx": chat_idx,
                    "turn_idx": turn_idx,
                    "persona": persona,
                    "context": {
                        "queries": queries[:turn_idx + 1],
                        "responses": responses[:turn_idx],
                    },
                    "reference": ref_response,
                    "generated": generated,
                    "score": score,
                    "ppl": round(ppl, 2) if not (ppl != ppl) else None,
                })

        # ---- 保存 ----
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        avg_ppl = (sum(x for x in total_ppl if x == x) / max(len([x for x in total_ppl if x == x]), 1))

        output = {
            "model_path": model_path,
            "model_type": model_type,
            "test_data": test_data_path,
            "score": score,
            "no_score": no_score,
            "do_sample": do_sample,
            "num_generations": len(results),
            "skipped": skipped,
            "avg_ppl": round(avg_ppl, 2),
            "generations": results,
        }

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(results)} generations to {out_file}")
        print(f"Average PPL: {avg_ppl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Before You Speak — 推理生成")

    parser.add_argument("--model_path", required=True, help="训练好的模型路径")
    parser.add_argument("--test_data", required=True, help="测试数据 JSON (如 test_scores.json)")
    parser.add_argument("--output", default="outputs/generations.json", help="输出文件路径")
    parser.add_argument("--scores", nargs="*", default=None,
                        help="Score 值列表, 如: 1.0 0.95 0.9 0.85 0.8 0.75")
    parser.add_argument("--no_score", action="store_true",
                        help="去掉 prompt 中的 Score 字段（消融实验）")
    parser.add_argument("--model_type", default="dgpt", choices=["dgpt", "llama"],
                        help="模型类型")
    parser.add_argument("--do_sample", action="store_true",
                        help="使用采样而不是 beam search")
    parser.add_argument("--device", default="cuda", help="设备: cuda / cpu")

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_path=args.output,
        scores=args.scores,
        no_score=args.no_score,
        model_type=args.model_type,
        device=args.device,
        do_sample=args.do_sample,
    )
