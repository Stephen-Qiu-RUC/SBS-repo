"""
步骤2: 名词 Mask 与 BART 填空
==============================
对 persona 中的名词进行 mask（替换为 <mask>），
然后使用 BART-large 模型对 mask 位置进行文本填空，
生成原句的"语义相近但表达不同"的变体（augmentation）。

核心思路（来自论文 "Score Before You Speak"）:
- 对 persona 句子中的名词进行遮挡
- 用 BART 生成多个填空结果
- 如果最优填空结果与原句完全相同，则使用次优结果
- 生成的变体将用于训练，让模型学会区分高质量的 paraphrase 和低质量的替换

输入: {split}_pos_tagged.json（步骤1的输出）
输出: {split}_masked.json，包含每个 persona 句子的原句和 mask 填空变体
"""

from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer
import random
import json
from tqdm import tqdm
import sys
import string


def main(file_path):
    """主函数：加载数据 -> 创建 mask 句子 -> BART 填空 -> 合并结果"""

    # 1. 加载步骤1生成的词性标注数据
    with open(f'{file_path}', 'r') as f:
        data = json.loads(f.read())

    random.seed(42)
    empty_masks = []

    # 2. 遍历所有对话，为每个包含名词的 persona 句子创建 mask 版本
    #    对每个名词单独 mask，即一个句子中如果有3个名词，就生成3个 mask 句子
    for chat in tqdm(data):
        for sent in chat['persona_postags']:
            # 筛选出所有名词（XPOS 以 "NN" 开头，如 NN、NNS、NNP、NNPS）
            nouns = [word[0] for word in sent if word[1].startswith("NN")]

            # 如果句子中没有名词，跳过
            if len(nouns) == 0:
                continue

            # 对每个名词单独进行 mask
            for noun in nouns:
                # 将当前名词替换为 <mask>，其余词保持不变
                words = [
                    "<mask>" if word[0] == noun else word[0]
                    for word in sent
                ]

                # 如果最后一个词是 <mask>，追加句号
                # 这样做是为了避免 BART 把标点填入 mask 位置
                if words[-1] == '<mask>':
                    words.append(".")

                # 将 mask 后的词列表拼接为字符串
                empty_masks.append(' '.join(words))

    # 3. 使用 BART-large 对 mask 句子进行填空
    #    BART 是一个 encoder-decoder 预训练模型，适合做文本填空（denoising）任务
    device = "cuda"
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large",
        forced_bos_token_id=0  # 强制使用 BOS token 作为起始
    )
    tok = BartTokenizer.from_pretrained("facebook/bart-large")
    model = model.to(device)

    filled_masks = []
    batch_size = 500  # 批量大小，平衡速度和显存占用

    # 分批处理，避免一次性加载过多数据到 GPU
    for idx in tqdm(range(0, len(empty_masks), batch_size)):
        if idx + batch_size >= len(empty_masks):
            end_idx = len(empty_masks)
        else:
            end_idx = idx + batch_size

        # 对当前批次进行 tokenize 并移到 GPU
        batch = tok(
            empty_masks[idx:idx + batch_size],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # 生成填空结果
        # num_return_sequences=2: 每个 mask 句子生成2个候选结果
        # max_new_tokens: 限制生成的最大 token 数，基于输入长度 + 5 留有余量
        generated_ids = model.generate(
            batch["input_ids"],
            num_return_sequences=2,
            max_new_tokens=5 + batch['input_ids'].size(1)
        )

        # 解码生成的 token ID 为文本
        sequences = tok.batch_decode(generated_ids, skip_special_tokens=True)
        filled_masks.extend(sequences)

    # 4. 将填空结果与原始数据合并
    i = 0  # 追踪当前在 filled_masks 中的位置
    for chat in tqdm(data):
        all_masked = []
        for sent in chat['persona_postags']:
            nouns = [word[0] for word in sent if word[1].startswith("NN")]

            if len(nouns) == 0:
                continue

            # 每个名词有2个候选填空结果
            end_idx = len(nouns) * 2
            regenerated = filled_masks[i:i + end_idx]

            # 找到该 persona 句子对应的原始文本
            original = chat['persona'][chat['persona_postags'].index(sent)]
            sent_masked = []

            # 对每个名词的2个候选结果进行处理
            for idx_inner in range(0, len(regenerated), 2):
                # 去除标点并归一化候选1（top-1 completion）
                corrupted = (
                    regenerated[idx_inner]
                    .translate(str.maketrans('', '', string.punctuation))
                    .strip()
                    .lower()
                )
                corrupted = ' '.join(corrupted.split())

                # 如果最优结果与原句相同或为空，则使用候选2（次优结果）
                # 这确保了我们获得的是"变化过"的文本，而非原样输出
                if corrupted == original or corrupted == '':
                    sent_masked.append(regenerated[idx_inner + 1])
                else:
                    sent_masked.append(regenerated[idx_inner])

            # 记录原始句子及其 mask 填空变体
            all_masked.append({
                'original': original,
                'masked': sent_masked
            })

            i = i + end_idx

        # 删除不再需要的词性标注信息，减小文件大小
        del chat['persona_postags']
        chat['aug_data'] = all_masked

    # 5. 保存结果
    split = Path(file_path).parts[-1].split('_')[0]
    with open(f'{split}_masked.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 命令行参数: 步骤1的词性标注结果文件路径
    main(sys.argv[1])
