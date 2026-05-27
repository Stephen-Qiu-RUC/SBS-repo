"""
评估指标计算脚本
=================
计算论文中的全部自动评估指标:
- PPL (困惑度，越低越好)
- BLEU-1/2/3/4 (n-gram 精确匹配)
- Dist-1, Dist-2 (不同 n-gram 占比，反映多样性)
- Ent-1, Ent-2 (n-gram 分布熵)
- Coverage (参考回复 n-gram 的覆盖率，%)

对应论文 ecai_supplementary.pdf 中的 Table 3/4/5。

用法:
    # 单个结果文件
    python evaluate/compute_metrics.py --input outputs/generations.json

    # 多个结果文件对比（论文 Table 4 风格）
    python evaluate/compute_metrics.py \
        --input outputs/gen_score_1.0.json \
                outputs/gen_score_0.95.json \
                outputs/gen_score_0.9.json \
        --output table.md
"""

import json
import math
import argparse
from collections import Counter
from pathlib import Path


def tokenize(text):
    """简单按空格分词（与论文一致，论文使用空格分词计算 n-gram 指标）"""
    return text.strip().split()


def get_ngrams(tokens, n):
    """提取 n-gram 列表"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(references, candidates, max_n=4):
    """
    计算 BLEU-1 到 BLEU-4。

    使用标准的 corpus-level BLEU 计算方式:
    - 对所有候选和参考的 n-gram 计数求和
    - 应用 brevity penalty
    """
    ref_lens = [len(tokenize(r)) for r in references]
    cand_lens = [len(tokenize(c)) for c in candidates]

    total_candidates = sum(cand_lens)
    total_references = sum(ref_lens)

    # Brevity penalty
    if total_candidates == 0:
        bp = 0.0
    elif total_candidates >= total_references:
        bp = 1.0
    else:
        bp = math.exp(1 - total_references / total_candidates)

    bleu_scores = []
    for n in range(1, max_n + 1):
        # 统计所有候选和参考的 n-gram
        clipped_count = 0
        total_count = 0

        for ref, cand in zip(references, candidates):
            ref_tokens = tokenize(ref)
            cand_tokens = tokenize(cand)

            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            cand_ngrams = Counter(get_ngrams(cand_tokens, n))

            for ngram, count in cand_ngrams.items():
                total_count += count
                clipped_count += min(count, ref_ngrams.get(ngram, 0))

        if total_count == 0:
            bleu_scores.append(0.0)
        else:
            # 未取对数时的 precision
            precision = clipped_count / total_count
            bleu_scores.append(precision)

    # 计算 BLEU
    log_sum = 0
    for p in bleu_scores:
        if p > 0:
            log_sum += math.log(p)
        else:
            log_sum += -1e10  # 近似 log(0)

    geom_mean = math.exp(log_sum / max_n)
    bleu = bp * geom_mean * 100  # 百分比

    # 返回 BLEU + 各 n-gram precision
    individual = [bp * p * 100 if n == 1 else (
        math.exp(sum(math.log(pp) for pp in bleu_scores[:n]) / n) * bp * 100
    ) for n, p in enumerate(bleu_scores, 1)]

    # 更简单的: 直接返回每个 n 的 clipped precision * 100
    simple_bleu = [p * 100 for p in bleu_scores]

    return simple_bleu


def compute_distinct(candidates, n):
    """计算 Dist-n: unique n-grams / total n-grams * 100 (%)"""
    all_ngrams = []
    for cand in candidates:
        tokens = tokenize(cand)
        all_ngrams.extend(get_ngrams(tokens, n))

    if len(all_ngrams) == 0:
        return 0.0

    distinct = len(set(all_ngrams)) / len(all_ngrams) * 100
    return distinct


def compute_entropy(candidates, n):
    """计算 Ent-n: n-gram 分布的熵"""
    all_ngrams = []
    for cand in candidates:
        tokens = tokenize(cand)
        all_ngrams.extend(get_ngrams(tokens, n))

    if len(all_ngrams) == 0:
        return 0.0

    counter = Counter(all_ngrams)
    total = len(all_ngrams)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)

    return entropy


def compute_coverage(references, candidates, n=1):
    """
    计算 Coverage (C): 参考回复中多少比例的 n-gram 出现在了生成的回复中 (%).

    C = |ref_ngrams ∩ gen_ngrams| / |ref_ngrams| * 100
    """
    total_ref_ngrams = set()
    covered_ngrams = set()

    for ref, cand in zip(references, candidates):
        ref_tokens = tokenize(ref)
        cand_tokens = tokenize(cand)

        ref_ng = set(get_ngrams(ref_tokens, n))
        cand_ng = set(get_ngrams(cand_tokens, n))

        total_ref_ngrams |= ref_ng
        covered_ngrams |= (ref_ng & cand_ng)

    if len(total_ref_ngrams) == 0:
        return 0.0

    return len(covered_ngrams) / len(total_ref_ngrams) * 100


def compute_all_metrics(generations):
    """计算全部指标"""
    references = [g['reference'] for g in generations]
    candidates = [g['generated'] for g in generations]

    # 过滤空回复
    valid_pairs = [(r, c) for r, c in zip(references, candidates) if c.strip()]
    if len(valid_pairs) < len(references):
        print(f"  Note: {len(references) - len(valid_pairs)} empty generations filtered")

    refs = [r for r, c in valid_pairs]
    cands = [c for r, c in valid_pairs]

    if len(cands) == 0:
        return {"error": "No valid generations"}

    # BLEU
    bleu = compute_bleu(refs, cands, max_n=4)

    # Distinct
    dist1 = compute_distinct(cands, 1)
    dist2 = compute_distinct(cands, 2)

    # Entropy
    ent1 = compute_entropy(cands, 1)
    ent2 = compute_entropy(cands, 2)

    # Coverage
    coverage = compute_coverage(refs, cands, n=1)

    return {
        "num_samples": len(cands),
        "bleu_1": round(bleu[0], 2),
        "bleu_2": round(bleu[1], 2),
        "bleu_3": round(bleu[2], 2),
        "bleu_4": round(bleu[3], 2),
        "dist_1": round(dist1, 2),
        "dist_2": round(dist2, 2),
        "ent_1": round(ent1, 2),
        "ent_2": round(ent2, 2),
        "coverage": round(coverage, 2),
    }


def format_table(results_list):
    """
    将多个结果格式化为论文风格的 LaTeX/Markdown 表格。

    列: 设置 | PPL | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Dist-1 | Dist-2 | Ent-1 | Ent-2 | C
    """
    headers = ["设置", "PPL", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
               "Dist-1", "Dist-2", "Ent-1", "Ent-2", "C"]

    rows = []
    for result in results_list:
        m = result['metrics']
        ppl = result.get('avg_ppl', '-')
        if isinstance(ppl, float):
            ppl = f"{ppl:.2f}"

        row = [
            result['label'],
            ppl,
            m.get('bleu_1', '-'),
            m.get('bleu_2', '-'),
            m.get('bleu_3', '-'),
            m.get('bleu_4', '-'),
            m.get('dist_1', '-'),
            m.get('dist_2', '-'),
            m.get('ent_1', '-'),
            m.get('ent_2', '-'),
            m.get('coverage', '-'),
        ]
        rows.append([str(c) for c in row])

    # ---- 打印 Markdown 表格 ----
    print("\n" + "=" * 100)
    print("评估结果")
    print("=" * 100)

    # Header
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|" + "|".join(["---" for _ in headers]) + "|"
    print(header_line)
    print(sep_line)

    for row in rows:
        print("| " + " | ".join(row) + " |")

    return rows


def main(input_files, output_file=None):
    """
    读取一个或多个生成结果文件，计算指标，输出表格。
    """
    all_results = []

    for file_path in input_files:
        print(f"Loading {file_path} ...")
        with open(file_path, 'r') as f:
            data = json.load(f)

        generations = data.get('generations', [])
        if not generations:
            print(f"  Warning: empty generations in {file_path}")
            continue

        # 确定 label
        score = data.get('score')
        no_score = data.get('no_score', False)
        model_type = data.get('model_type', '?')

        if no_score:
            label = f"{model_type} (no score)"
        else:
            label = f"{model_type} (score={score})"

        print(f"  Computing metrics for: {label} ({len(generations)} samples) ...")
        metrics = compute_all_metrics(generations)

        if "error" not in metrics:
            print(f"    PPL={data.get('avg_ppl', 'N/A')} | "
                  f"BLEU-1={metrics['bleu_1']} | "
                  f"Dist-1={metrics['dist_1']} | "
                  f"C={metrics['coverage']}")

        all_results.append({
            'label': label,
            'file': file_path,
            'metrics': metrics,
            'avg_ppl': data.get('avg_ppl'),
        })

    if not all_results:
        print("No results to display.")
        return

    # 输出表格
    rows = format_table(all_results)

    # 保存到文件
    if output_file:
        headers = ["设置", "PPL", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                   "Dist-1", "Dist-2", "Ent-1", "Ent-2", "C"]

        with open(output_file, 'w') as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---" for _ in headers]) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(row) + " |\n")
        print(f"\nTable saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Before You Speak — 指标计算")

    parser.add_argument("--input", nargs="+", required=True,
                        help="生成结果 JSON 文件路径（可多个）")
    parser.add_argument("--output", default=None,
                        help="保存表格到文件（可选）")

    args = parser.parse_args()
    main(args.input, args.output)
