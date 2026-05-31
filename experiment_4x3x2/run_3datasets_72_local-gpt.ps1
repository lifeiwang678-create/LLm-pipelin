<#
Run 72 MAIN/full-subset combinations locally on Windows PowerShell with OpenAI GPT API.

Combinations:
  3 datasets x 4 inputs x 3 LM usages x 2 outputs = 72
  LM usages: direct, few_shot, multi_agent

This script treats "full data" as SUBSET_LEVEL="main", i.e. the prepared
LLMSubsets main files:
  Processed\LLMSubsets\<DATASET>\main\<DATASET>_<INPUT>_main_samples.pkl

Usage:

  cd C:\path\to\LLm-pipelin\experiment_4x3x2

  $env:OPENAI_API_KEY = "your_new_openai_api_key"
  $env:PROCESSED_ROOT = "D:\DATA\experiment_4x3x2\Processed"
  $env:LLM_MODEL = "gpt-5.4-mini"
  $env:SUBSET_LEVEL = "main"
  $env:FEW_SHOT_TRAIN_SUBSET_LEVEL = "pilot"
  $env:FEW_SHOT_EXAMPLE_SUBJECTS = "3"
  $env:CONCURRENCY = "1"

  powershell -ExecutionPolicy Bypass -File .\run_3datasets_72_local-gpt.ps1

Notes:
  - BASE should point to the directory containing main.py.
  - PROCESSED_ROOT should point to the Processed directory.
  - This script avoids using $input because $input is a PowerShell automatic variable.
  - API key is read from OPENAI_API_KEY / API_KEY and also passed to main.py by --api-key.
  - multi_agent is enabled because the original uploaded script runs 72 combinations.
#>

$ErrorActionPreference = 'Stop'

$BASE = $env:BASE
if (-not $BASE) {
    $BASE = (Get-Location).Path
}

$PROCESSED_ROOT = $env:PROCESSED_ROOT
if (-not $PROCESSED_ROOT) {
    $PROCESSED_ROOT = "D:\DATA\experiment_4x3x2\Processed"
}

$LOGROOT = $env:LOGROOT
if (-not $LOGROOT) {
    $LOGROOT = Join-Path $BASE ('logs\llm_72_gpt_main_' + (Get-Date -Format "yyyyMMddHHmmss"))
}

$API_URL = $env:API_URL
if (-not $API_URL) {
    $API_URL = "https://api.openai.com/v1"
}

$API_KEY = $env:API_KEY
if (-not $API_KEY) {
    $API_KEY = $env:OPENAI_API_KEY
}

$LLM_MODEL = $env:LLM_MODEL
if (-not $LLM_MODEL) {
    $LLM_MODEL = "gpt-5.4-mini"
}

$CONCURRENCY = $env:CONCURRENCY
if (-not $CONCURRENCY) {
    $CONCURRENCY = 1
}

$SUBSET_LEVEL = $env:SUBSET_LEVEL
if (-not $SUBSET_LEVEL) {
    $SUBSET_LEVEL = "main"
}

$FEW_SHOT_TRAIN_SUBSET_LEVEL = $env:FEW_SHOT_TRAIN_SUBSET_LEVEL
if (-not $FEW_SHOT_TRAIN_SUBSET_LEVEL) {
    $FEW_SHOT_TRAIN_SUBSET_LEVEL = "pilot"
}

$FEW_SHOT_EXAMPLE_SUBJECTS = $env:FEW_SHOT_EXAMPLE_SUBJECTS
if (-not $FEW_SHOT_EXAMPLE_SUBJECTS) {
    $FEW_SHOT_EXAMPLE_SUBJECTS = 3
}

$LOG_EVERY = $env:LOG_EVERY
if (-not $LOG_EVERY) {
    $LOG_EVERY = 10
}

if (-not $API_KEY) {
    Write-Error "ERROR: OPENAI_API_KEY is not set. Set OPENAI_API_KEY before running this script."
    exit 1
}

$env:OPENAI_API_KEY = $API_KEY
$env:API_KEY = $API_KEY

if (-not (Test-Path $BASE)) {
    Write-Error "ERROR: BASE does not exist: $BASE"
    exit 1
}

if (-not (Test-Path $PROCESSED_ROOT)) {
    Write-Error "ERROR: PROCESSED_ROOT does not exist: $PROCESSED_ROOT"
    exit 1
}

$mainPy = Join-Path $BASE "main.py"
if (-not (Test-Path $mainPy)) {
    Write-Error "ERROR: main.py not found in BASE: $mainPy"
    exit 1
}

New-Item -ItemType Directory -Path $LOGROOT -Force | Out-Null
Set-Location $BASE

Write-Host "Job started at $(Get-Date)"
Write-Host "Base: $BASE"
Write-Host "Processed root: $PROCESSED_ROOT"
Write-Host "Log dir: $LOGROOT"
Write-Host "API URL: $API_URL"
Write-Host "LLM model: $LLM_MODEL"
Write-Host "Client concurrency: $CONCURRENCY"
Write-Host "Evaluation subset: $SUBSET_LEVEL"
Write-Host "Few-shot train subset: $FEW_SHOT_TRAIN_SUBSET_LEVEL"
Write-Host "Few-shot example subjects: $FEW_SHOT_EXAMPLE_SUBJECTS"

$venvActivate = Join-Path $BASE ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "Activated virtualenv: $venvActivate"
}
else {
    Write-Warning "Virtualenv activate not found at $venvActivate - continuing without activating."
}

$datasets = @("WESAD", "HHAR", "DREAMT")
$inputTypes = @("raw_data", "feature_description", "encoded_time_series", "extra_knowledge")
$lms = @("direct", "few_shot", "multi_agent")
$outputs = @("label_only", "label_explanation")

foreach ($dataset in $datasets) {
    foreach ($inputType in $inputTypes) {
        $subsetCache = Join-Path $PROCESSED_ROOT "LLMSubsets\$dataset\$SUBSET_LEVEL\${dataset}_${inputType}_${SUBSET_LEVEL}_samples.pkl"

        if (-not (Test-Path $subsetCache)) {
            Write-Error "Missing fixed evaluation subset cache: $subsetCache"
            Write-Error "Expected MAIN/full-subset file. Run prepare_data_subsets.py and prepare_subset_inputs.py for SUBSET_LEVEL=$SUBSET_LEVEL before this script."
            exit 1
        }
    }
}

$statusCsv = Join-Path $LOGROOT "status.csv"
"dataset,input,lm,output,status,start_time,end_time,log" | Out-File -FilePath $statusCsv -Encoding utf8

$script:failures = 0

function Invoke-One {
    param(
        [string]$dataset,
        [string]$inputType,
        [string]$lm,
        [string]$output
    )

    $startTime = Get-Date -Format o

    $logFile = Join-Path $LOGROOT "${dataset}_${inputType}_${lm}_${output}.log"
    $evalCache = Join-Path $PROCESSED_ROOT "LLMSubsets\$dataset\$SUBSET_LEVEL\${dataset}_${inputType}_${SUBSET_LEVEL}_samples.pkl"
    $trainCache = $evalCache

    Write-Host "=================================================="
    Write-Host "Running: dataset=$dataset input=$inputType lm=$lm output=$output"
    Write-Host "Eval cache: $evalCache"
    Write-Host "Log: $logFile"

    $extraArgs = @()

    if ($lm -eq "few_shot") {
        $trainCache = Join-Path $PROCESSED_ROOT "LLMSubsets\$dataset\$FEW_SHOT_TRAIN_SUBSET_LEVEL\${dataset}_${inputType}_${FEW_SHOT_TRAIN_SUBSET_LEVEL}_samples.pkl"

        if (-not (Test-Path $trainCache)) {
            Write-Host "Missing few-shot train subset cache: $trainCache"
            Write-Host "Refusing to fall back to eval subset cache because that can leak evaluation samples into few-shot examples."

            $endTime = Get-Date -Format o
            $script:failures = $script:failures + 1

            "${dataset},${inputType},${lm},${output},1,${startTime},${endTime},${logFile}" |
                Out-File -FilePath $statusCsv -Append -Encoding utf8

            return
        }

        Write-Host "Train cache: $trainCache"

        $extraArgs += "--few-shot-example-selection"
        $extraArgs += "leave_one_subject_out"
        $extraArgs += "--few-shot-example-subjects"
        $extraArgs += "$FEW_SHOT_EXAMPLE_SUBJECTS"
        $extraArgs += "--few-shot-examples-per-subject-per-label"
        $extraArgs += "1"
        $extraArgs += "--few-shot-n-per-class"
        $extraArgs += "1"
        $extraArgs += "--train-input-cache-file"
        $extraArgs += $trainCache

        if ($inputType -eq "raw_data") {
            $extraArgs += "--few-shot-example-max-chars"
            $extraArgs += "500"
        }
        elseif ($inputType -eq "encoded_time_series") {
            $extraArgs += "--few-shot-example-max-chars"
            $extraArgs += "300"
        }
        else {
            $extraArgs += "--few-shot-example-max-chars"
            $extraArgs += "800"
        }
    }

    if ($lm -eq "multi_agent") {
        $extraArgs += "--multi-agent-intermediate-max-tokens"
        $extraArgs += "128"
    }

    $argList = @(
        "main.py",
        "-dataset", $dataset,
        "-Input", $inputType,
        "-LM", $lm,
        "-output", $output,
        "--use-input-cache",
        "--input-cache-dir", $PROCESSED_ROOT,
        "--subject-split", "all",
        "--eval-input-cache-file", $evalCache,
        "--api-url", $API_URL,
        "--api-key", $API_KEY,
        "-llm", $LLM_MODEL,
        "--concurrency", "$CONCURRENCY",
        "--log-every", "$LOG_EVERY"
    )

    $argList += $extraArgs

    try {
        & python @argList *> $logFile
        $status = $LASTEXITCODE
    }
    catch {
        $status = 1
        $_ | Out-File -FilePath $logFile -Append -Encoding utf8
    }

    $endTime = Get-Date -Format o

    if ($status -ne 0) {
        $script:failures = $script:failures + 1
        Write-Host "FAILED: dataset=$dataset input=$inputType lm=$lm output=$output status=$status"
    }
    else {
        Write-Host "OK: dataset=$dataset input=$inputType lm=$lm output=$output"
    }

    "${dataset},${inputType},${lm},${output},${status},${startTime},${endTime},${logFile}" |
        Out-File -FilePath $statusCsv -Append -Encoding utf8
}

foreach ($dataset in $datasets) {
    foreach ($inputType in $inputTypes) {
        foreach ($lm in $lms) {
            foreach ($output in $outputs) {
                Invoke-One -dataset $dataset -inputType $inputType -lm $lm -output $output
            }
        }
    }
}

Write-Host "=================================================="
Write-Host "All 72 MAIN/full-subset combinations finished at $(Get-Date)"
Write-Host "Failures: $script:failures"
Write-Host "Logs saved in: $LOGROOT"
Write-Host "Status CSV: $statusCsv"

if ($script:failures -ne 0) {
    exit 1
}
else {
    exit 0
}
