"""
步骤3: BERTScore 质量评分
==========================
使用 BERTScore 对步骤2生成的 persona 句子变体进行语义质量评分。
BERTScore 基于 BERT 的上下文嵌入计算候选文本与参考文本的语义相似度，
得分越高表示变体与原句的语义越接近。

用途:
- 为训练数据中的每个 augmented 回复分配一个质量分数
- 训练时，模型需要学会预测这个分数（"开口前先打分"）

输入: {split}_masked.json（步骤2的输出）
输出: {split}_scores.json，每个 masked 句子附带 BERTScore F1 分数
"""

from pathlib import Path
import json
from tqdm import tqdm
import sys
from bert_score import BERTScorer


def make_ref_cand_pairs(data):
    """
    构建参考文本（reference）和候选文本（candidate）的对齐列表。

    每个对话包含多个人设句子及其变体，需要将所有变体展平，
    同时将对应的原始句子复制多份以对齐。

    返回:
        all_refs: list[list[str]] - 每个对话的参考文本列表
        all_cands: list[list[str]] - 每个对话的候选文本列表
    """
    all_refs, all_cands = [], []

    for chat in data:
        refs, cands = [], []
        for aug in chat['aug_data']:
            # 每个原始句子对应多个 masked 变体
            # 将原始句子复制 len(masked) 份以实现一一对应
            refs.extend([aug['original']] * len(aug['masked']))
            cands.extend(aug['masked'])

        assert len(refs) == len(cands), "Check code!"
        all_refs.append(refs)
        all_cands.append(cands)

    assert len(all_refs) == len(all_cands), "Check code!"
    return all_refs, all_cands


def main(file_path):
    """主函数：加载数据 -> 计算 BERTScore -> 合并分数"""

    split = Path(file_path).parts[-1].split('_')[0]

    # 1. 加载步骤2生成的 mask 填空数据
    with open(f'{file_path}', 'r') as f:
        data = json.loads(f.read())

    # 2. 构建参考-候选文本对
    refs, cands = make_ref_cand_pairs(data)

    # 3. 初始化 BERTScorer
    # - model_type: 使用 deberta-xlarge-mnli，在 NLI 任务上微调过的 DeBERTa
    # - rescale_with_baseline=True: 将原始 BERTScore 重新缩放，使其更接近人类判断
    # - lang: 英语
    scorer = BERTScorer(
        lang="en",
        rescale_with_baseline=True,
        model_type="microsoft/deberta-xlarge-mnli"
    )

    # 4. 逐对话计算 BERTScore F1 分数
    #    BERTScore 返回 (Precision, Recall, F1) 三元组
    #    我们使用 F1 作为综合质量指标
    fscores = []
    for i in tqdm(range(len(refs))):
        P, R, F1 = scorer.score(cands[i], refs[i])
        fscores.append(F1.tolist())  # 将 tensor 转换为 Python list

    # 5. 将分数合并回数据结构
    #    每个 masked 句子新增 'score' 字段
    for i in range(len(data)):
        scores = fscores[i]
        j = 0
        for aug in data[i]['aug_data']:
            masked_with_scores = []
            for masked_sent in aug['masked']:
                masked_with_scores.append({
                    'sent': masked_sent,
                    'score': round(scores[j], 2)  # 保留2位小数
                })
                j += 1
            aug['masked'] = masked_with_scores

    # 6. 保存带评分的完整数据
    with open(f'{split}_scores.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=3)


if __name__ == "__main__":
    # 命令行参数: 步骤2的 mask 填空结果文件路径
    main(sys.argv[1])
