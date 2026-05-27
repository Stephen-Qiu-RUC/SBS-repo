"""
步骤1: 词性标注 (POS Tagging)
==============================
使用 Stanza 对对话数据中的 persona（人物设定）进行词性标注，
提取每个单词及其对应的 XPOS（细粒度词性标签），
为后续步骤2中识别名词并做 mask 做准备。

输入: 原始对话数据文件（如 train_both_original.txt）
输出: {split}_pos_tagged.json，包含每个 persona 句子的词性标注结果
"""

import stanza
import json
from tqdm import tqdm
import sys
from pathlib import Path


def create_data(data_file):
    """
    解析原始对话数据文件，提取 persona、query、response、candidate 四类信息。

    原始数据格式说明：
    - 以 "your persona: " 开头的行表示人物设定（persona），连续的 persona 行属于同一个对话
    - 非 persona 行包含 4 个 tab 分隔的字段：
      [0] query（用户输入）, [1] response（正确回复）,
      [2] (未使用), [3] candidates（候选回复，以 | 分隔）
    - 每个 persona 段落对应一个对话，包含多轮 query/response

    返回:
        persona: list[list[str]] - 每个对话的 persona 句子列表
        query: list[list[str]] - 每个对话的用户输入列表
        response: list[list[str]] - 每个对话的回复列表
        cand: list[list[list[str]]] - 每个对话的候选回复列表
    """
    with open(data_file, "r", encoding="utf8") as f:
        persona = []
        query = []
        response = []
        cand = []
        is_persona = False        # 标记当前是否正在读取 persona 行
        tmp_persona = []          # 临时存储当前对话的 persona
        tmp_query = []            # 临时存储当前对话的 query
        tmp_response = []         # 临时存储当前对话的 response
        tmp_cand = []             # 临时存储当前对话的 candidates
        first = True              # 标记是否为第一个对话段落
        cnt = 0                   # 行计数器
        sum_u = 0                 # 总 utterance 数量统计

        for line in f:
            cnt += 1
            line = line.strip()

            if "your persona: " in line:
                # 遇到新的 persona 段落时，保存上一个对话的数据
                if not is_persona and not first:
                    query.append(tmp_query)
                    response.append(tmp_response)
                    cand.append(tmp_cand)
                    sum_u += len(tmp_query)
                    tmp_query = []
                    tmp_response = []
                    tmp_cand = []

                first = False
                is_persona = True
                # 提取 "your persona: " 后面的内容
                line = line.split(": ", maxsplit=1)[1]
                tmp_persona.append(line)
            else:
                # persona 段落结束，保存并重置
                if is_persona:
                    persona.append(tmp_persona)
                    is_persona = False
                    tmp_persona = []

                # 跳过行开头的序号（如 "1 user said: " 中的 "1 "）
                line = line[line.find(" ") + 1:]
                tmp_query.append(line.split("\t")[0])
                tmp_response.append(line.split("\t")[1])
                # 第4个字段是候选回复，以 | 分隔
                tmp_cand.append(line.split("\t")[3].split("|"))

        # 保存最后一个对话段落的数据
        query.append(tmp_query)
        response.append(tmp_response)
        cand.append(tmp_cand)
        sum_u += len(tmp_query)

        # 确保四种数据的对话数量一致
        assert len(query) == len(response) == len(persona) == len(cand)

    print("{} has {} dialog and {} query".format(data_file, len(query), sum_u))
    return persona, query, response, cand


def main(file_path):
    """主函数：解析数据 -> 词性标注 -> 保存结果"""

    # 1. 从原始文件中提取对话数据
    persona, query, response, cand = create_data(file_path)

    # 从文件路径中提取数据集分割名（如 train、valid、test）
    split = Path(file_path).parts[-1].split('_')[0]

    # 2. 将数据组织为字典列表，每个字典对应一个对话
    data = []
    for i in range(len(persona)):
        data.append({
            'persona': persona[i],
            'queries': query[i],
            'responses': response[i],
            # 'distractors': cand[i]  # 暂不需要候选回复
        })

    # 3. 初始化 Stanza 流水线
    # - 语言: 英语
    # - 处理器: tokenize（分词）, mwt（多词扩展）, pos（词性标注）
    # - pos_batch_size: 3000 足够高效且不会在 8GB 显存的 GPU 上 OOM
    nlp = stanza.Pipeline(
        lang='en',
        processors='tokenize,mwt,pos',
        use_gpu=True,
        pos_batch_size=300000
    )

    # 4. 对每个对话的 persona 进行词性标注
    for chat in tqdm(data):
        pos_tags = []
        for response in chat['persona']:
            # 使用 Stanza 处理每个 persona 句子
            res = nlp(response)
            # 提取每个词及其 XPOS 词性标签
            # XPOS 是 Penn Treebank 风格的细粒度词性标签
            tags = [
                (word.text, word.xpos)
                for sent in res.sentences
                for word in sent.words
            ]
            pos_tags.append(tags)

        chat['persona_postags'] = pos_tags
        assert len(chat['persona_postags']) == len(chat['persona']), "Check code!"

    # 5. 保存词性标注结果到 JSON 文件
    with open(f'{split}_pos_tagged.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 命令行参数: 原始数据文件路径
    main(sys.argv[1])
