# 运行说明

本文档说明当前 `experiment_4x3x2` 实验框架的常用运行方式。

所有命令默认从仓库的实验目录执行：

```bash
cd /home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2
```

## 1. 实验组合

当前框架是 `3 x 4 x 3 x 2` 设计，共 72 组：

```text
3 datasets x 4 input types x 3 LM usages x 2 output formats
```

可选项如下：

| 维度 | 可选值 |
| --- | --- |
| Dataset | `WESAD`, `HHAR`, `DREAMT` |
| Input | `raw_data`, `feature_description`, `encoded_time_series`, `extra_knowledge` |
| LM | `direct`, `few_shot`, `multi_agent` |
| Output | `label_only`, `label_explanation` |

二分类标签定义：

| Dataset | Label 0 | Label 1 |
| --- | --- | --- |
| `WESAD` | no stress | stress |
| `HHAR` | walking downstairs | walking upstairs |
| `DREAMT` | wake | sleep |

## 2. Python 环境

普通实验入口使用项目内 `.venv`：

```bash
source .venv/bin/activate
python main.py -h
```

Slurm 的 72 组 debug 脚本会自动切换两个环境：

```text
.venv_vllm_cu121  # 启动 vLLM 服务
.venv             # 执行 main.py 实验代码
```

如果需要重新安装普通运行依赖：

```bash
.venv/bin/pip install -r requirements.txt
```

## 3. 数据位置

默认数据路径定义在 `Dataset/registry.py`。

| Dataset | 默认位置 | 说明 |
| --- | --- | --- |
| WESAD | `..` | 期望存在 `../S2/S2.pkl` 这类 subject 目录 |
| HHAR | `Dataset/HHAR` | 可用 `--data-dir` 覆盖 |
| DREAMT | `Dataset/DREAMT` | 可用 `--data-dir` 覆盖 |

本地原始数据、`Processed/`、`Results/`、大模型和历史输出不会上传到 GitHub。

## 4. 单次实验

最小命令格式：

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM direct \
  -output label_only
```

常用 debug 参数：

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input raw_data \
  -LM direct \
  -output label_only \
  --subjects S2 \
  --balanced-per-label 1 \
  --log-every 1
```

使用外部 OpenAI-compatible LLM 服务：

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM direct \
  -output label_only \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key lm-studio \
  -llm qwen2.5-7b-instruct
```

结果默认写入 `Results/`：

```text
Results/<run_name>_<timestamp>.csv
Results/<run_name>_<timestamp>_metrics.json
Results/<run_name>_<timestamp>_config.json
```

## 5. 预处理缓存

框架支持两层缓存：

| 缓存 | 内容 | 常用参数 |
| --- | --- | --- |
| Dataset cache | 已切窗的 `SensorSample` | `--use-processed` |
| Input cache | 已转换成 prompt 输入文本的 `LLMSample` | `--use-input-cache` |

生成 dataset cache：

```bash
.venv/bin/python preprocess_datasets.py -dataset WESAD --subjects S2 --overwrite
.venv/bin/python preprocess_datasets.py -dataset HHAR --data-dir "<HHAR_DATA_DIR>" --max-rows 200000 --overwrite
.venv/bin/python preprocess_datasets.py -dataset DREAMT --data-dir "<DREAMT_DATA_DIR>" --subjects S099 --overwrite
```

生成 input cache：

```bash
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input all --subjects S2 --overwrite
```

用 input cache 运行：

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM direct \
  -output label_only \
  --use-input-cache
```

## 6. 固定子集缓存

推荐用固定子集做可复现 debug/pilot/main 实验。先生成固定窗口子集：

```bash
.venv/bin/python prepare_data_subsets.py
```

再生成每种 input 的 LLM 子集缓存：

```bash
.venv/bin/python prepare_subset_inputs.py
```

主要输出路径：

```text
Processed/DataSubsets/<DATASET>/<debug|pilot|main>/<DATASET>_<subset>_windows.pkl
Processed/LLMSubsets/<DATASET>/<debug|pilot|main>/<DATASET>_<INPUT>_<subset>_samples.pkl
```

用固定 debug 子集跑单次实验：

```bash
.venv/bin/python main.py \
  -dataset HHAR \
  -Input raw_data \
  -LM direct \
  -output label_only \
  --use-input-cache \
  --subject-split all \
  --eval-input-cache-file "Processed/LLMSubsets/HHAR/debug/HHAR_raw_data_debug_samples.pkl" \
  --log-every 1
```

## 7. Few-Shot 运行规则

当前 few-shot 默认策略是 `leave_one_subject_out`。

关键语义：

- `test_subjects` 或 eval cache 决定评估样本。
- `leave_one_subject_out` 会为每个评估样本排除当前 subject。
- few-shot 示例从非评估 subject 中按 label 抽样。
- `leave_one_subject_out` 使用 `--few-shot-examples-per-subject-per-label` 控制每个 subject 每个 label 的示例数。
- `--few-shot-n-per-class` 只属于旧的 `class_balanced` 策略，不要和 `leave_one_subject_out` 混用成不同值。
- 不要在 few-shot 中用 `--subjects` 表达训练/测试含义；请明确使用 `--train-subjects`、`--test-subjects`，或使用 `--train-input-cache-file`、`--eval-input-cache-file`。

用固定 cache 跑 few-shot：

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM few_shot \
  -output label_only \
  --use-input-cache \
  --subject-split all \
  --train-input-cache-file "Processed/LLMSubsets/WESAD/pilot/WESAD_feature_description_pilot_samples.pkl" \
  --eval-input-cache-file "Processed/LLMSubsets/WESAD/debug/WESAD_feature_description_debug_samples.pkl" \
  --few-shot-example-selection leave_one_subject_out \
  --few-shot-example-subjects 3 \
  --few-shot-examples-per-subject-per-label 1 \
  --few-shot-example-max-chars 800 \
  --log-every 1
```

## 8. 运行 72 组 Debug

确认固定子集缓存已存在：

```bash
for dataset in WESAD HHAR DREAMT; do
  for level in debug pilot; do
    for input in raw_data feature_description encoded_time_series extra_knowledge; do
      path="Processed/LLMSubsets/${dataset}/${level}/${dataset}_${input}_${level}_samples.pkl"
      [[ -s "$path" ]] || echo "MISSING $path"
    done
  done
done
```

提交 72 组 debug：

```bash
sbatch run_3datasets_72_debug_sbatch.sh
```

默认设置：

| 变量 | 默认值 |
| --- | --- |
| `SUBSET_LEVEL` | `debug` |
| `FEW_SHOT_TRAIN_SUBSET_LEVEL` | `pilot` |
| `CONCURRENCY` | `8` |
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` |
| `SERVED_MODEL_NAME` | `qwen2.5-7b-instruct` |
| `PORT` | `8000` |

覆盖默认值示例：

```bash
sbatch --export=ALL,SUBSET_LEVEL=pilot,CONCURRENCY=4 run_3datasets_72_debug_sbatch.sh
```

查看队列：

```bash
squeue -u 25t9801
```

主 Slurm 日志：

```text
/home/users/grad/2025/25t9801/logs/llm_72_debug_<JOBID>.out
/home/users/grad/2025/25t9801/logs/llm_72_debug_<JOBID>.err
```

每次运行会生成一个详细日志目录：

```text
/home/users/grad/2025/25t9801/logs/llm_72_debug_<YYYYMMDDHHMMSS>/
```

其中最重要的是：

```text
status.csv    # 72 组状态汇总
vllm.log      # vLLM 服务日志
*.log         # 每个组合的单独日志
gpu_usage.csv # GPU 监控记录
```

统计 72 组是否全部成功：

```bash
awk -F, 'NR>1 {total++; if ($5 != 0) failed++} END {printf "total=%d failed=%d\n", total, failed+0}' \
  /home/users/grad/2025/25t9801/logs/llm_72_debug_<YYYYMMDDHHMMSS>/status.csv
```

## 9. 常见问题

### cache metadata mismatch

如果修改了 dataset loader 参数，例如窗口长度、stride、HHAR `max_rows`，旧 cache 会被拒绝。删除并重建相关 cache：

```bash
rm -f Processed/<DATASET>_*_samples.pkl
.venv/bin/python preprocess_inputs.py -dataset <DATASET> -Input all --overwrite
```

### balanced_per_label 报错

`--balanced-per-label N` 要求每个 label 最终都正好有 `N` 个样本。若某个 label 样本不足，降低 `N` 或扩大 subject/cache；若样本超量，检查是否有额外 limit 或标签过滤逻辑改变。

### few-shot subject 报错

few-shot 中 `subjects` 语义不明确。改用：

```text
--train-subjects ...
--test-subjects ...
```

或者固定 cache：

```text
--train-input-cache-file ...
--eval-input-cache-file ...
```

### vLLM 端口占用

72 组脚本启动前会检查 `http://127.0.0.1:8000/v1/models`。如果已有服务占用端口，停止旧服务或换端口：

```bash
sbatch --export=ALL,PORT=8001 run_3datasets_72_debug_sbatch.sh
```

### pytest 不可用

当前集群普通环境可能没有安装 `pytest`。可以先做语法检查：

```bash
.venv/bin/python -m py_compile core/runner.py LM/__init__.py LM/few_shot.py LM/multi_agent.py
```

如果要跑测试，需要在可写环境中安装测试依赖后执行：

```bash
.venv/bin/python -m pytest test_few_shot_sampling.py test_runner_regressions.py
```

## 10. 文件作用速查

主路径文件：

| 文件 | 作用 | 日常是否需要 |
| --- | --- | --- |
| `main.py` | CLI 主入口，最终调用 `core/runner.py` | 常用 |
| `run_experiment.py` | 从 JSON/YAML config 或 grid 批量展开实验，仍然调用同一个 runner | 偶尔用 |
| `requirements.txt` | 普通 `.venv` 依赖列表 | 环境重建时用 |

数据检查和缓存构建：

| 文件 | 作用 | 日常是否需要 |
| --- | --- | --- |
| `count_dataset_samples.py` | 不调用 LLM，只统计各 dataset/subject/label 的窗口数量 | 数据核对时用 |
| `preprocess_datasets.py` | 生成第一层 dataset cache，也就是切窗后的 `SensorSample` | 建 cache 时用 |
| `preprocess_inputs.py` | 生成第二层 input cache，也就是 prompt-ready 的 `LLMSample` | 建 cache 时用 |
| `prepare_data_subsets.py` | 从预处理窗口中抽取固定 `debug/pilot/main` 子集 | 72 组可复现实验前用 |
| `prepare_subset_inputs.py` | 为固定子集生成四种 input 的 LLM cache | 72 组可复现实验前用 |

Slurm / 批处理脚本：

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `run_3datasets_72_debug_sbatch.sh` | 当前推荐的 3 数据集 72 组 debug/pilot/main Slurm 入口 | 推荐使用 |
| `run_wesad_24_full_sbatch.sh` | 只跑 WESAD 的 24 组较大子集/全量脚本 | 专项脚本 |
| `run_hhar_dreamt_preprocess_sbatch.sh` | 在 Slurm 上预处理 HHAR 和 DREAMT cache | 辅助脚本 |
| `run_all_3datasets_4x3x2.ps1` | PowerShell 版 72 组批处理，偏本地/Windows 工作流 | 备用 |
| `run_wesad_4x3x2_small.ps1` | PowerShell 版 WESAD 小规模 24 组 | 备用 |
| `run_wesad_remaining23.sh` | 早期用于补跑 WESAD 剩余组合 | 历史补跑脚本 |
| `run_wesad_remaining5_sbatch.sh` | 早期用于补跑 WESAD 剩余 5 组 | 历史补跑脚本 |

模型服务、benchmark 和 smoke test：

| 文件 | 作用 | 日常是否需要 |
| --- | --- | --- |
| `benchmark_vllm_batch.py` | 压测 OpenAI-compatible vLLM batching/concurrency | 调参时用 |
| `run_vllm_batch_benchmark_sbatch.sh` | Slurm 版 vLLM batching benchmark | 调参时用 |
| `setup_qwen3_vllm_env.sh` | 创建 Qwen3 vLLM 独立环境 | 换 Qwen3 时用 |
| `serve_qwen3_transformers_openai.py` | 用 transformers 起一个小型 OpenAI-compatible Qwen3 服务 | vLLM 不适合时备用 |
| `smoke_qwen3_json.py` | 测试 Qwen3 是否能稳定输出可解析 JSON，并可跑小 pipeline smoke | 换模型时用 |
| `run_qwen3_vllm_smoke_sbatch.sh` | Slurm 版 Qwen3 vLLM smoke test | 换模型时用 |
| `run_qwen3_transformers_smoke_sbatch.sh` | Slurm 版 Qwen3 transformers smoke test | vLLM 失败时备用 |
| `summarize_cost_profile.py` | 汇总 metrics JSON 中的 token、耗时、成本估计，可合并 GPU telemetry | 分析结果时用 |

测试文件：

| 文件 | 作用 |
| --- | --- |
| `test_embedding_alignment_input.py` | 检查 `encoded_time_series`/旧名 `embedding_alignment` 输入格式 |
| `test_feature_description_factory.py` | 检查 feature description factory、dataset 注入和 subject split 等逻辑 |
| `test_few_shot_sampling.py` | 检查 few-shot leave-one-subject-out 抽样 |
| `test_runner_regressions.py` | 检查 runner 近期修复点，包括并发、cache metadata、balanced sampling、few-shot subject 语义 |

简单判断：

- 正式跑实验优先看 `main.py`、`run_3datasets_72_debug_sbatch.sh`、cache 相关四个脚本。
- `run_wesad_remaining*` 属于历史补跑脚本，保留是为了复现以前的运行方式，不建议作为新实验入口。
- Qwen3 和 benchmark 脚本是模型服务调试工具，不影响当前 Qwen2.5 72 组主流程。
