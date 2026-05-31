# experiment_4x3x2 代码运行说明

本文档用于说明 `LLm-pipelin/experiment_4x3x2` 实验框架的运行方式，覆盖单次实验、固定子集实验、few-shot 实验、72 组 Slurm 批量实验、结果查看和常见错误处理。

默认实验目录：

```bash
cd /home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2
```

从 Windows PowerShell 登录服务器时，可先执行：

```powershell
ssh hosei-gpu-master
```

登录后进入实验目录：

```bash
cd ~/projects/LLm-pipelin/experiment_4x3x2
```

---

## 1. 实验框架概览

当前实验设计为：

```text
3 datasets × 4 input types × 3 LM usages × 2 output formats = 72 组实验
```

| 维度 | 可选值 |
| --- | --- |
| Dataset | `WESAD`, `HHAR`, `DREAMT` |
| Input | `raw_data`, `feature_description`, `encoded_time_series`, `extra_knowledge` |
| LM Usage | `direct`, `few_shot`, `multi_agent` |
| Output | `label_only`, `label_explanation` |

二分类标签定义：

| Dataset | Label 0 | Label 1 |
| --- | --- | --- |
| `WESAD` | no stress | stress |
| `HHAR` | walking downstairs | walking upstairs |
| `DREAMT` | wake | sleep |

---

## 2. 运行前检查

进入实验目录后，先确认代码和环境是否可用：

```bash
pwd
ls
source .venv/bin/activate
python main.py -h
```

如果能看到 `main.py` 的参数说明，说明普通实验入口可以正常使用。

如果 `.venv` 不存在，或依赖缺失，可重新安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

普通实验使用：

```text
.venv
```

72 组 Slurm debug 脚本会自动切换两个环境：

```text
.venv_vllm_cu121  # 启动 vLLM 服务
.venv             # 执行 main.py 实验代码
```

---

## 3. 数据目录

默认数据路径定义在：

```text
Dataset/registry.py
```

| Dataset | 默认位置 | 说明 |
| --- | --- | --- |
| WESAD | `..` | 期望存在 `../S2/S2.pkl`、`../S3/S3.pkl` 这类 subject 目录 |
| HHAR | `Dataset/HHAR` | 可通过 `--data-dir` 覆盖 |
| DREAMT | `Dataset/DREAMT` | 可通过 `--data-dir` 覆盖 |

本地原始数据、`Processed/`、`Results/`、大模型文件和历史输出通常不上传到 GitHub。

检查当前目录下是否已有缓存和结果目录：

```bash
ls Processed
ls Results
```

---

## 4. 推荐运行顺序

建议按照下面的顺序运行，避免直接提交 72 组实验后才发现数据或缓存有问题。

### 4.1 检查 main.py 是否可运行

```bash
source .venv/bin/activate
python main.py -h
```

### 4.2 跑一个最小 debug 样本

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

该命令只用于确认 pipeline 能跑通，不用于正式结果分析。

### 4.3 生成 dataset cache

```bash
.venv/bin/python preprocess_datasets.py -dataset WESAD --subjects S2 --overwrite
.venv/bin/python preprocess_datasets.py -dataset HHAR --data-dir "<HHAR_DATA_DIR>" --max-rows 200000 --overwrite
.venv/bin/python preprocess_datasets.py -dataset DREAMT --data-dir "<DREAMT_DATA_DIR>" --subjects S099 --overwrite
```

dataset cache 是切窗后的 `SensorSample`。

### 4.4 生成 input cache

```bash
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input all --subjects S2 --overwrite
```

input cache 是已经转换成 prompt 输入文本的 `LLMSample`。

### 4.5 生成固定子集

```bash
.venv/bin/python prepare_data_subsets.py
.venv/bin/python prepare_subset_inputs.py
```

固定子集用于可复现的 debug、pilot 和 main 实验。

### 4.6 用固定 debug 子集跑单组实验

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

单组实验确认正常后，再提交 72 组 Slurm 任务。

---

## 5. 单次实验运行方式

最小命令格式：

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM direct \
  -output label_only
```

指定 OpenAI-compatible LLM 服务：

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

结果默认写入：

```text
Results/<run_name>_<timestamp>.csv
Results/<run_name>_<timestamp>_metrics.json
Results/<run_name>_<timestamp>_config.json
```

CSV 文件保存逐样本预测结果；`metrics.json` 保存 Accuracy、Macro-F1、Weighted-F1、token、耗时等统计；`config.json` 保存本次运行配置。

---

## 6. 预处理缓存说明

当前框架支持两层缓存。

| 缓存类型 | 内容 | 常用参数 |
| --- | --- | --- |
| Dataset cache | 切窗后的 `SensorSample` | `--use-processed` |
| Input cache | prompt-ready 的 `LLMSample` | `--use-input-cache` |

使用 input cache 运行：

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM direct \
  -output label_only \
  --use-input-cache
```

固定子集输出路径：

```text
Processed/DataSubsets/<DATASET>/<debug|pilot|main>/<DATASET>_<subset>_windows.pkl
Processed/LLMSubsets/<DATASET>/<debug|pilot|main>/<DATASET>_<INPUT>_<subset>_samples.pkl
```

确认固定子集是否存在：

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

如果没有输出 `MISSING ...`，说明 debug 和 pilot 的 input cache 都存在。

---

## 7. 固定子集规则

当前固定子集规则如下：

| Dataset | 子集规则 |
| --- | --- |
| WESAD | 类别平衡抽样，例如 `160:160` |
| HHAR | 9 users，每人每类 50 个样本 |
| DREAMT | 100 subjects，每人每类 5 个样本 |

随机种子固定为：

```text
seed = 42
```

同一套采样流程重复运行时，理论上应得到相同的 subset 和 few-shot examples。若结果不同，需要检查代码中是否存在未固定 seed 的随机过程。

---

## 8. Few-shot 运行规则

当前 few-shot 推荐使用：

```text
leave_one_subject_out
```

语义如下：

- `test_subjects` 或 eval cache 决定评估样本。
- 对每个评估样本，few-shot examples 不使用该样本所属 subject。
- few-shot examples 从其他 subjects 中抽取。
- 推荐规则是随机选择若干非测试 subjects，每个 subject 每个 label 抽取固定数量样本。
- `--few-shot-examples-per-subject-per-label` 控制每个 subject 每个 label 的示例数。
- `--few-shot-n-per-class` 属于旧的 `class_balanced` 策略，不建议和 `leave_one_subject_out` 混用成不同值。

不要在 few-shot 中用 `--subjects` 表示训练/测试划分。应使用：

```text
--train-subjects ...
--test-subjects ...
```

或者直接指定固定 cache：

```text
--train-input-cache-file ...
--eval-input-cache-file ...
```

固定 cache few-shot 示例：

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

如果需要导出或检查 few-shot examples，应确认结果 CSV 中是否包含类似字段：

```text
few_shot_example_subjects
few_shot_example_count
few_shot_example_ids
```

若这些列为空或串台，需要检查 runner 是否从 `build_prompt_with_metadata` 的返回值中读取 metadata，而不是从共享的 few-shot 实例状态中读取。

---

## 9. 运行 72 组 Debug 实验

提交前先确认固定子集缓存已经存在：

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

脚本默认设置：

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

查看任务是否还在运行：

```bash
squeue -j <JOBID>
```

取消任务：

```bash
scancel <JOBID>
```

---

## 10. 日志和结果查看

主 Slurm 日志位置：

```text
/home/users/grad/2025/25t9801/logs/llm_72_debug_<JOBID>.out
/home/users/grad/2025/25t9801/logs/llm_72_debug_<JOBID>.err
```

每次运行还会生成一个详细日志目录：

```text
/home/users/grad/2025/25t9801/logs/llm_72_debug_<YYYYMMDDHHMMSS>/
```

常用文件：

| 文件 | 作用 |
| --- | --- |
| `status.csv` | 72 组运行状态汇总 |
| `vllm.log` | vLLM 服务日志 |
| `*.log` | 每个实验组合的单独日志 |
| `gpu_usage.csv` | GPU 使用记录 |

查看最后几行 Slurm 输出：

```bash
tail -n 50 /home/users/grad/2025/25t9801/logs/llm_72_debug_<JOBID>.out
```

查看错误日志：

```bash
tail -n 80 /home/users/grad/2025/25t9801/logs/llm_72_debug_<JOBID>.err
```

统计 72 组是否全部成功：

```bash
awk -F, 'NR>1 {total++; if ($5 != 0) failed++} END {printf "total=%d failed=%d\n", total, failed+0}' \
  /home/users/grad/2025/25t9801/logs/llm_72_debug_<YYYYMMDDHHMMSS>/status.csv
```

查看失败组合：

```bash
awk -F, 'NR==1 || $5 != 0 {print}' \
  /home/users/grad/2025/25t9801/logs/llm_72_debug_<YYYYMMDDHHMMSS>/status.csv
```

查看 GPU 使用：

```bash
head /home/users/grad/2025/25t9801/logs/llm_72_debug_<YYYYMMDDHHMMSS>/gpu_usage.csv
tail /home/users/grad/2025/25t9801/logs/llm_72_debug_<YYYYMMDDHHMMSS>/gpu_usage.csv
```

---

## 11. 成本和耗时统计

单次实验结束后，可查看对应的 metrics 文件：

```bash
ls Results/*_metrics.json | tail
cat Results/<run_name>_<timestamp>_metrics.json
```

如果要汇总多个 metrics 文件，可使用：

```bash
.venv/bin/python summarize_cost_profile.py --results-dir Results
```

如果需要合并 GPU telemetry，确认 `gpu_usage.csv` 路径后再运行对应统计命令。

重点关注字段：

```text
sample_count
llm_call_count
prompt_tokens
completion_tokens
total_tokens
wall_time_seconds
avg_latency_seconds
invalid_rate
accuracy
macro_f1
weighted_f1
```

---

## 12. 常见问题处理

### 12.1 cache metadata mismatch

原因通常是 dataset loader 参数发生变化，例如窗口长度、stride、HHAR `max_rows` 等。旧 cache 和当前配置不一致时会被拒绝。

处理方式：删除并重建相关 cache。

```bash
rm -f Processed/<DATASET>_*_samples.pkl
.venv/bin/python preprocess_inputs.py -dataset <DATASET> -Input all --overwrite
```

如果使用固定子集，也需要重建：

```bash
.venv/bin/python prepare_data_subsets.py --overwrite
.venv/bin/python prepare_subset_inputs.py --overwrite
```

### 12.2 balanced_per_label 报错

`--balanced-per-label N` 要求每个 label 最终都有 `N` 个样本。

处理方式：

- 某个 label 样本不足：降低 `N`，或扩大 subjects/cache。
- 样本超量但仍报错：检查是否有额外 limit、标签过滤或 subject 过滤逻辑改变。

### 12.3 few-shot subject 报错

few-shot 中不要用 `--subjects` 表达训练/测试划分。

改用：

```text
--train-subjects ...
--test-subjects ...
```

或者固定 cache：

```text
--train-input-cache-file ...
--eval-input-cache-file ...
```

### 12.4 few-shot metadata 串台

并发运行时，如果 few-shot 的 subject 信息写入共享实例状态，可能导致 CSV 中 few-shot example subjects 被其他线程覆盖。

修复方向：

```text
_run_one_sample 调用 build_prompt_with_metadata
直接从返回值读取 few-shot metadata
不要从共享 FewShotUsage 实例的 last_example_subjects / last_example_count 读取
```

并发运行时尤其需要检查该问题。

### 12.5 vLLM 端口占用

72 组脚本启动前会检查：

```text
http://127.0.0.1:8000/v1/models
```

如果已有服务占用端口，可停止旧服务，或换端口：

```bash
sbatch --export=ALL,PORT=8001 run_3datasets_72_debug_sbatch.sh
```

### 12.6 pytest 不可用

集群普通环境可能没有安装 `pytest`。可先做语法检查：

```bash
.venv/bin/python -m py_compile core/runner.py LM/__init__.py LM/few_shot.py LM/multi_agent.py
```

如果需要跑测试，先安装测试依赖：

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest test_few_shot_sampling.py test_runner_regressions.py
```

### 12.7 LLM 输出不是 JSON

如果 `invalid_rate` 较高，需要检查：

- prompt 是否明确要求严格 JSON。
- `label_only` 是否只允许 `{"predicted_state": <label>}`。
- 模型是否输出解释、Markdown 或多余文本。
- `max_tokens` 是否过小。
- temperature 是否过高。

### 12.8 结果明显偏向某一类

如果模型几乎全部预测为一个 label，需要检查：

- 真实标签分布是否平衡。
- prompt 中是否暗示某一类更常见。
- raw_data 是否过长，导致模型只抓住局部异常信号。
- feature description 是否包含对 label 有误导的描述。
- few-shot examples 是否类别均衡。
- train/eval subject 是否混入或泄漏。

---

## 13. 文件作用速查

### 13.1 主入口

| 文件 | 作用 | 日常是否需要 |
| --- | --- | --- |
| `main.py` | CLI 主入口，最终调用 `core/runner.py` | 常用 |
| `run_experiment.py` | 从 JSON/YAML config 或 grid 批量展开实验，仍然调用同一个 runner | 偶尔用 |
| `requirements.txt` | 普通 `.venv` 依赖列表 | 环境重建时用 |

### 13.2 数据检查和缓存构建

| 文件 | 作用 | 日常是否需要 |
| --- | --- | --- |
| `count_dataset_samples.py` | 不调用 LLM，只统计各 dataset/subject/label 的窗口数量 | 数据核对时用 |
| `preprocess_datasets.py` | 生成第一层 dataset cache，即切窗后的 `SensorSample` | 建 cache 时用 |
| `preprocess_inputs.py` | 生成第二层 input cache，即 prompt-ready 的 `LLMSample` | 建 cache 时用 |
| `prepare_data_subsets.py` | 从预处理窗口中抽取固定 `debug/pilot/main` 子集 | 72 组可复现实验前用 |
| `prepare_subset_inputs.py` | 为固定子集生成四种 input 的 LLM cache | 72 组可复现实验前用 |

### 13.3 Slurm / 批处理脚本

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `run_3datasets_72_debug_sbatch.sh` | 当前推荐的 3 数据集 72 组 debug/pilot/main Slurm 入口 | 推荐使用 |
| `run_wesad_24_full_sbatch.sh` | 只跑 WESAD 的 24 组较大子集/全量脚本 | 专项脚本 |
| `run_hhar_dreamt_preprocess_sbatch.sh` | 在 Slurm 上预处理 HHAR 和 DREAMT cache | 辅助脚本 |
| `run_all_3datasets_4x3x2.ps1` | PowerShell 版 72 组批处理，偏本地/Windows 工作流 | 备用 |
| `run_wesad_4x3x2_small.ps1` | PowerShell 版 WESAD 小规模 24 组 | 备用 |
| `run_wesad_remaining23.sh` | 早期用于补跑 WESAD 剩余组合 | 历史补跑脚本 |
| `run_wesad_remaining5_sbatch.sh` | 早期用于补跑 WESAD 剩余 5 组 | 历史补跑脚本 |

### 13.4 模型服务、benchmark 和 smoke test

| 文件 | 作用 | 日常是否需要 |
| --- | --- | --- |
| `benchmark_vllm_batch.py` | 压测 OpenAI-compatible vLLM batching/concurrency | 调参时用 |
| `run_vllm_batch_benchmark_sbatch.sh` | Slurm 版 vLLM batching benchmark | 调参时用 |
| `setup_qwen3_vllm_env.sh` | 创建 Qwen3 vLLM 独立环境 | 换 Qwen3 时用 |
| `serve_qwen3_transformers_openai.py` | 用 transformers 起一个 OpenAI-compatible Qwen3 服务 | vLLM 不适合时备用 |
| `smoke_qwen3_json.py` | 测试 Qwen3 是否能稳定输出可解析 JSON | 换模型时用 |
| `run_qwen3_vllm_smoke_sbatch.sh` | Slurm 版 Qwen3 vLLM smoke test | 换模型时用 |
| `run_qwen3_transformers_smoke_sbatch.sh` | Slurm 版 Qwen3 transformers smoke test | vLLM 失败时备用 |
| `summarize_cost_profile.py` | 汇总 metrics JSON 中的 token、耗时、成本估计 | 分析结果时用 |

### 13.5 测试文件

| 文件 | 作用 |
| --- | --- |
| `test_embedding_alignment_input.py` | 检查 `encoded_time_series`/旧名 `embedding_alignment` 输入格式 |
| `test_feature_description_factory.py` | 检查 feature description factory、dataset 注入和 subject split 等逻辑 |
| `test_few_shot_sampling.py` | 检查 few-shot leave-one-subject-out 抽样 |
| `test_runner_regressions.py` | 检查 runner 近期修复点，包括并发、cache metadata、balanced sampling、few-shot subject 语义 |

---

## 14. 日常使用建议

正式跑实验时，优先关注这些文件：

```text
main.py
run_3datasets_72_debug_sbatch.sh
preprocess_datasets.py
preprocess_inputs.py
prepare_data_subsets.py
prepare_subset_inputs.py
```

历史补跑脚本如 `run_wesad_remaining*` 只用于复现旧实验，不建议作为新实验入口。

Qwen3 和 benchmark 脚本用于模型服务调试，不影响当前 Qwen2.5 72 组主流程。

---

## 15. 最小可执行命令清单

从零开始，推荐按下面顺序执行。

```bash
cd ~/projects/LLm-pipelin/experiment_4x3x2
source .venv/bin/activate
python main.py -h
```

检查单样本：

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

生成固定子集：

```bash
.venv/bin/python prepare_data_subsets.py
.venv/bin/python prepare_subset_inputs.py
```

确认 cache：

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

查看队列：

```bash
squeue -u 25t9801
```

查看运行结果：

```bash
awk -F, 'NR>1 {total++; if ($5 != 0) failed++} END {printf "total=%d failed=%d\n", total, failed+0}' \
  /home/users/grad/2025/25t9801/logs/llm_72_debug_<YYYYMMDDHHMMSS>/status.csv
```
