#!/bin/bash
# ============================================================================
# Score Before You Speak — 一键复现脚本
# ============================================================================
# 用法:
#   bash run.sh [选项]
#
# 选项:
#   --dataset     personachat | convai2          (默认: personachat)
#   --model       dgpt | llama                   (默认: dgpt)
#   --epochs      N                              (默认: 数据集默认值)
#   --output      PATH                           (默认: models/<dataset>_<model>)
#   --steps       all | preprocess | train | evaluate  (默认: all)
#   --eval_scores "1.0 0.95 0.9"                      (评估用的 score 值)
#   --help        显示帮助信息
#
# 示例:
#   bash run.sh                                              # 全流程: personachat + dgpt
#   bash run.sh --dataset convai2 --model dgpt --epochs 6    # convai2 + dgpt
#   bash run.sh --model llama                                # personachat + llama
#   bash run.sh --steps preprocess                           # 只预处理
#   bash run.sh --steps evaluate                             # 只评估（需已有模型和测试数据）
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

# ---------- 默认参数 ----------
DATASET="personachat"
MODEL="dgpt"
EPOCHS=""
OUTPUT=""
STEPS="all"
EVAL_SCORES="1.0 0.95 0.9 0.85 0.8 0.75"

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --eval_scores) EVAL_SCORES="$2"; shift 2 ;;
        --help)
            head -40 "$0" | grep -E '^#' | sed 's/^# //;s/^#//'
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---------- 步骤验证 ----------
if [[ "$STEPS" != "all" && "$STEPS" != "preprocess" && \
      "$STEPS" != "train" && "$STEPS" != "evaluate" ]]; then
    echo "错误: --steps 必须为 all / preprocess / train / evaluate"
    exit 1
fi
if [[ "$DATASET" != "personachat" && "$DATASET" != "convai2" ]]; then
    echo "错误: --dataset 必须为 personachat 或 convai2"
    exit 1
fi

# ---------- 模型验证 ----------
if [[ "$MODEL" != "dgpt" && "$MODEL" != "llama" ]]; then
    echo "错误: --model 必须为 dgpt 或 llama"
    exit 1
fi

# ---------- 默认 epochs ----------
if [[ -z "$EPOCHS" ]]; then
    if [[ "$DATASET" == "personachat" ]]; then
        [[ "$MODEL" == "dgpt" ]] && EPOCHS=15 || EPOCHS=3
    else
        [[ "$MODEL" == "dgpt" ]] && EPOCHS=6 || EPOCHS=2
    fi
fi

# ---------- 默认输出路径 ----------
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="models/${DATASET}_${MODEL}"
fi

# ---------- 数据路径 ----------
DATA_DIR="data/${DATASET}"
PREPROCESS_DIR="preprocess"
TRAIN_DIR="train"

# 三个切分
SPLITS=("train" "valid" "test")


# ---------- 打印配置 ----------
echo "============================================"
echo "  Score Before You Speak — 复现执行"
echo "============================================"
echo "数据集:        $DATASET"
echo "模型:          $MODEL"
echo "轮数:          $EPOCHS"
echo "输出路径:      $OUTPUT"
echo "执行步骤:      $STEPS"
echo "数据目录:      $DATA_DIR"
echo "============================================"
echo ""

# ---------- 检查环境 ----------
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "错误: 未找到 $1, 请确认 conda 环境已激活"
        exit 1
    fi
}

check_gpu() {
    python -c "import torch; torch.randn(1).cuda(); print('GPU 可用:', torch.cuda.get_device_name(0))" || {
        echo "错误: GPU 不可用, 请检查 PyTorch/CUDA 安装"
        exit 1
    }
}

echo "[检查] 环境..."
check_cmd python
check_gpu
echo ""

# ---------- 步骤 1: 词性标注 ----------
step1() {
    echo ">>> [1/4] 词性标注 (POS Tagging)..."
    for split in "${SPLITS[@]}"; do
        local input="${DATA_DIR}/${split}_self_original.txt"
        local output="${split}_pos_tagged.json"

        if [[ -f "$output" ]]; then
            echo "  [跳过] $output 已存在"
            continue
        fi

        if [[ ! -f "$input" ]]; then
            echo "  [警告] $input 不存在，跳过"
            continue
        fi

        echo "  处理 $input -> $output"
        python "${PREPROCESS_DIR}/step1-postag.py" "$input"
    done
    echo ""
}

# ---------- 步骤 2: 名词 Mask + BART 填空 ----------
step2() {
    echo ">>> [2/4] 名词掩码 + BART 填空 (Masking)..."
    for split in "${SPLITS[@]}"; do
        local input="${split}_pos_tagged.json"
        local output="${split}_masked.json"

        if [[ -f "$output" ]]; then
            echo "  [跳过] $output 已存在"
            continue
        fi

        if [[ ! -f "$input" ]]; then
            echo "  [警告] $input 不存在，跳过"
            continue
        fi

        echo "  处理 $input -> $output"
        python "${PREPROCESS_DIR}/step2-masking.py" "$input"
    done
    echo ""
}

# ---------- 步骤 3: BERTScore 评分 ----------
step3() {
    echo ">>> [3/4] BERTScore 评分 (Scoring)..."
    for split in "${SPLITS[@]}"; do
        local input="${split}_masked.json"
        local output="${split}_scores.json"

        if [[ -f "$output" ]]; then
            echo "  [跳过] $output 已存在"
            continue
        fi

        if [[ ! -f "$input" ]]; then
            echo "  [警告] $input 不存在，跳过"
            continue
        fi

        echo "  处理 $input -> $output"
        python "${PREPROCESS_DIR}/step3-scoring.py" "$input"
    done
    echo ""
}

# ---------- 步骤 4: 转换为 HF Dataset ----------
step4() {
    echo ">>> [4/4] 转换为 HuggingFace Dataset..."
    local hf_dir="${MODEL}_train_final"

    if [[ -d "$hf_dir" ]]; then
        echo "  [跳过] $hf_dir 已存在"
        echo ""
        return
    fi

    local input="train_scores.json"
    if [[ ! -f "$input" ]]; then
        echo "  [警告] $input 不存在"
        echo ""
        return
    fi

    echo "  处理 $input -> $hf_dir (模型格式: $MODEL)"
    python "${PREPROCESS_DIR}/step4-convert_to_hf_dataset.py" "$input" "$MODEL"
    echo ""
}

# ---------- 训练 ----------
do_train() {
    echo ">>> 训练模型..."
    mkdir -p "$OUTPUT"

    local hf_dir="${MODEL}_train_final"
    if [[ ! -d "$hf_dir" ]]; then
        echo "错误: 未找到 HF 数据集 $hf_dir, 请先运行预处理"
        exit 1
    fi

    echo "  模型: $MODEL | 数据集: $hf_dir | 轮数: $EPOCHS | 输出: $OUTPUT"
    python "${TRAIN_DIR}/train_${MODEL}.py" \
        --exp_name "${DATASET}_${MODEL}" \
        --dataset_path "$hf_dir" \
        --n_epochs "$EPOCHS" \
        --output_path "$OUTPUT"
    echo ""
}

# ---------- 评估 ----------
do_evaluate() {
    echo ">>> 评估模型..."
    local test_data="test_scores.json"

    if [[ ! -f "$test_data" ]]; then
        echo "错误: 未找到测试数据 $test_data，请先运行预处理"
        exit 1
    fi

    if [[ ! -d "$OUTPUT" ]]; then
        echo "错误: 未找到模型 $OUTPUT，请先训练"
        exit 1
    fi

    mkdir -p outputs

    # 生成回复（论文 Table 4: 不同 score 值的影响）
    echo "  生成测试集回复 (scores: $EVAL_SCORES)..."
    python evaluate/generate.py \
        --model_path "$OUTPUT" \
        --test_data "$test_data" \
        --scores $EVAL_SCORES \
        --output "outputs/${DATASET}_${MODEL}" \
        --model_type "$MODEL"

    # 计算指标
    echo "  计算评估指标..."
    local gen_files=""
    for s in $EVAL_SCORES; do
        gen_files="$gen_files outputs/${DATASET}_${MODEL}_score_${s}.json"
    done
    python evaluate/compute_metrics.py \
        --input $gen_files \
        --output "outputs/${DATASET}_${MODEL}_metrics.md"

    echo "  评估完成，结果保存在 outputs/"
}

# ---------- 执行 ----------
if [[ "$STEPS" == "all" || "$STEPS" == "preprocess" ]]; then
    step1
    step2
    step3
    step4
fi

if [[ "$STEPS" == "all" || "$STEPS" == "train" ]]; then
    do_train
fi

if [[ "$STEPS" == "evaluate" ]]; then
    do_evaluate
fi

echo "============================================"
echo "  完成! 模型保存在: $OUTPUT"
echo "============================================"
