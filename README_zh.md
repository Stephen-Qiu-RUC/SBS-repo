# Score Before You Speak
<a href="https://arxiv.org/abs/2508.06886"><img src="https://img.shields.io/badge/arXiv-2508.06886-red"></a>
<a href="https://arpita2512.github.io/score_before_you_speak/"><img src="https://img.shields.io/badge/Project%20Page-online-green"></a>
<a href="https://huggingface.co/collections/Arpita1/score-before-you-speak-68a4a9f2b2598c476d35b723"><img src="https://img.shields.io/badge/%F0%9F%A4%97Hugging%20Face-Models-blue"></a>
<br>

ECAI 2025 论文 **Score Before You Speak: Improving Persona Consistency in Dialogue Generation using Response Quality Scores** 的代码仓库。

![](assets/overview.png)

## 环境安装

实验在两种不同的环境下进行（详见补充材料）。主要依赖为 `torch, transformers, datasets, tqdm, stanza, bert-score, accelerate, wandb`。精确的环境配置请参阅 *persona.yml*（DialoGPT）和 *llama.yml*（Llama 3.1）。下载 Llama 模型需要同意社区许可协议，并设置 `HF_TOKEN` 环境变量。

```
git clone https://github.com/arpita2512/score_before_you_speak.git
cd score_before_you_speak
conda env create -f <environment-name>.yml
conda activate <environment-name>
pip install stanza
pip install bert-score
```

## 一键运行

项目提供了 `run.sh` 脚本用于一键完成数据预处理和训练：

```bash
# 完整复现: 预处理 + 训练 (PERSONA-CHAT + DialoGPT)
bash run.sh

# 指定数据集和模型
bash run.sh --dataset convai2 --model dgpt --epochs 6
bash run.sh --dataset personachat --model llama --epochs 3

# 只预处理，不训练
bash run.sh --steps preprocess

# 只训练 (需要有预处理好的数据)
bash run.sh --steps train --model dgpt --epochs 15

# 自定义输出路径
bash run.sh --output models/my_experiment

# 查看所有选项
bash run.sh --help
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集: `personachat` / `convai2` | `personachat` |
| `--model` | 模型: `dgpt` / `llama` | `dgpt` |
| `--epochs` | 训练轮数 | dgpt+personachat=15, dgpt+convai2=6, llama+personachat=3, llama+convai2=2 |
| `--output` | 模型保存路径 | `models/<dataset>_<model>` |
| `--steps` | 执行步骤: `all` / `preprocess` / `train` | `all` |

脚本会自动跳过已生成的中间文件，如果某一步被中断，重新运行会从断点继续。

## 数据

PERSONA-CHAT 和 ConvAI2 数据集可通过 [ParlAI](https://github.com/facebookresearch/ParlAI) 获取。我们使用两个数据集所有划分的 `<split>_self_original.txt` 文件。

## 数据预处理

### 词性标注

```
python preprocess/postag.py <path_to_txt_file> # 将词性标注后的文件保存为 json
```

**词性标注后的数据格式**

```
{
    "persona": [
      ...
    ],
    "queries": [
      ...
    ],
    "responses": [
      ...
    ],
    "response_postags": [
      [
        [
          word,
          pos-tag
        ],
        ...
      ]
      ...
    ]
}
```

### 掩码填充

```
python preprocess/masking.py <path_to_pos_tagged_json> # 将掩码填充后的数据保存为 json
```

- `bart-large` 的批次大小默认设置为 500，可能需要根据 GPU 显存进行调整。

**掩码填充后的数据格式**

```
{
    "persona": [
      ...
    ],
    "queries": [
      ...
    ],
    "responses": [
      ...
    ],
    "aug_data": [
      {
        "original": ...,
        "masked": [
          ..., # 掩码 1 的补全结果
          ... # 掩码 n 的补全结果
        ]
      },
    ]
} # 掩码填充后移除词性标注以减小数据体积
```

### 评分

```
python preprocess/scoring.py <path_to_masked_json> # 将评分后的数据保存为 json
```

**评分后的数据格式**

```
{
    "persona": [
      ...
    ],
    "queries": [
      ...
    ],
    "responses": [
      ...
    ],
    "aug_data": [
      {
        "original": ...,
        "masked": [
          {
            "sent": ... ,
            "score": ...,
          },
          {
            "sent": ... ,
            "score": ...,
          },
          ...
        ]
      },
    ]
}
```

### 转换为 HuggingFace 数据集

```
python preprocess/convert_to_hf_dataset.py <path_to_json_with_scores> <model_name> # 将数据保存为已分词的 HF 数据集
```

- 模型名称（第二个参数）可选 `dgpt` 或 `llama`

## 训练

注意：Wandb 追踪需要注册账号并获取 API 密钥（[参见此处](https://docs.wandb.ai/models/quickstart#install-the-wandb-library-and-log-in)）

### DialoGPT

```
python train/train_dgpt.py --exp_name <wandb项目名称> --dataset_path <HF数据集路径> --n_epochs <训练轮数> --output_path <模型保存路径>
```

- PERSONA-CHAT 设置 n_epochs 为 15，ConvAI2 设置 n_epochs 为 6。
- 批次大小默认设置为 16，可能需要根据 GPU 显存进行调整。

**提示模板**

> <|startoftext|>Your persona: *persona信息*<|sp1|>User: *用户话语 1*<|sp2|>Bot: *机器人话语 1*<|sp1|>User: *用户话语 2 Score: ..*<|sp2|>Bot: *机器人话语 2*<|endoftext|>

### Llama 3.1

```
python train/train_llama.py --exp_name <wandb项目名称> --dataset_path <HF数据集路径> --n_epochs <训练轮数> --output_path <模型保存路径>
```

- PERSONA-CHAT 设置 n_epochs 为 3，ConvAI2 设置 n_epochs 为 2。

**提示模板**（基于 [Llama 3.1 模型卡片](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#prompt-template)）

><|begin_of_text|><|start_header_id|>system<|end_header_id|>
>
>Cutting Knowledge Date: December 2023
>Today Date: 26 Jul 2024
>
>*persona信息*..<|eot_id|><|start_header_id|>user<|end_header_id|>
>
>*用户话语 1*<|eot_id|><|start_header_id|>assistant<|end_header_id|>
>
>*机器人话语 1*<|eot_id|><|start_header_id|>user<|end_header_id|>
>
>*用户话语 2 Score: ..*<|eot_id|><|start_header_id|>assistant<|end_header_id|>
>
>*机器人话语 2*<|eot_id|>

## 评估

训练完成后，使用 `evaluate/` 目录下的脚本在测试集上运行推理和评估。

### 推理生成

```bash
# 默认: score=1.0, beam search 解码
python evaluate/generate.py \
    --model_path models/personachat_dgpt \
    --test_data test_scores.json \
    --output outputs/generations.json

# 论文 Table 4: 不同 score 值对生成质量的影响
python evaluate/generate.py \
    --model_path models/personachat_dgpt \
    --test_data test_scores.json \
    --scores 1.0 0.95 0.9 0.85 0.8 0.75 \
    --output outputs/gen

# 论文 Table 3: 消融实验（去掉 prompt 中的 Score）
python evaluate/generate.py \
    --model_path models/personachat_dgpt \
    --test_data test_scores.json \
    --no_score \
    --output outputs/gen_no_score
```

### 计算指标

```bash
# 单个结果文件
python evaluate/compute_metrics.py --input outputs/generations.json

# 多个 score 值对比（输出论文风格表格）
python evaluate/compute_metrics.py \
    --input outputs/gen_score_1.0.json \
            outputs/gen_score_0.95.json \
            outputs/gen_score_0.9.json \
    --output results_table.md
```

### 评估指标说明

| 指标 | 含义 | 方向 |
|------|------|------|
| PPL | 困惑度，模型对参考回复的预测能力 | 越低越好 |
| BLEU-1/2/3/4 | n-gram 与参考回复的重叠度 | 越高越好 |
| Dist-1, Dist-2 | 不同 n-gram 占比，衡量生成多样性 | 越高越好 |
| Ent-1, Ent-2 | n-gram 分布熵，衡量词汇丰富度 | 越高越好 |
| C (Coverage) | 参考回复 n-gram 的覆盖率 (%) | 越高越好 |

### 一键评估（通过 run.sh）

```bash
bash run.sh --steps evaluate --output models/personachat_dgpt
bash run.sh --steps evaluate --eval_scores "1.0 0.95 0.9"
```

## BibTeX

```
@inproceedings{saggar2025,
  author    = {Saggar, Arpita and Darling, Jonathan C. and Dimitrova, Vania and Sarikaya, Duygu and Hogg, David C.},
  title     = {Score Before You Speak: Improving Persona Consistency in Dialogue Generation using Response Quality Scores},
  booktitle = {Proceedings of the 28th European Conference on Artificial Intelligence},
  year      = {2025},
  url = {https://ebooks.iospress.nl/volumearticle/75972},
}
```
