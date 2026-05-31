<#
Retry only failed combinations from a previous Gemini MAIN/no-agent run.

Set SOURCE_STATUS_CSV to the status CSV produced by the earlier run.

This script:
  - reads rows with status != 0
  - reruns only those combinations
  - writes retry logs to a new retry log directory
  - writes retry_status.csv
  - does not rerun successful combinations
  - does not delete or modify previous logs/results

Usage:

  cd C:\path\to\LLm-pipelin\experiment_4x3x2

  $env:GEMINI_API_KEY="your_gemini_api_key"
  $env:PROCESSED_ROOT="D:\DATA\experiment_4x3x2\Processed"
  $env:SUBSET_LEVEL="main"
  $env:FEW_SHOT_TRAIN_SUBSET_LEVEL="pilot"
  $env:FEW_SHOT_EXAMPLE_SUBJECTS="5"
  $env:CONCURRENCY="1"

  powershell -ExecutionPolicy Bypass -File .\resume_gemini35flash_main_from_wesad_feature_fewshot.ps1

Optional:
  $env:SOURCE_STATUS_CSV="C:\path\to\resume_status.csv"
  $env:RETRY_LOGROOT="C:\path\to\retry_logs"
#>

$ErrorActionPreference = 'Continue'

$BASE = $env:BASE
if (-not $BASE) {
    $BASE = (Get-Location).Path
}

$PROCESSED_ROOT = $env:PROCESSED_ROOT
if (-not $PROCESSED_ROOT) {
    $PROCESSED_ROOT = "D:\DATA\experiment_4x3x2\Processed"
}

$SOURCE_STATUS_CSV = $env:SOURCE_STATUS_CSV
if (-not $SOURCE_STATUS_CSV) {
    $SOURCE_STATUS_CSV = Join-Path $BASE "logs\resume_status.csv"
}

$RETRY_LOGROOT = $env:RETRY_LOGROOT
if (-not $RETRY_LOGROOT) {
    $RETRY_LOGROOT = Join-Path $BASE ('logs\retry_failed_gemini35flash_main_' + (Get-Date -Format "yyyyMMddHHmmss"))
}

$GEMINI_API_KEY = $env:GEMINI_API_KEY

$LM_PROVIDER = "gemini"

$LLM_MODEL = $env:LLM_MODEL
if (-not $LLM_MODEL) {
    $LLM_MODEL = "gemini-3.5-flash"
}
if ($LLM_MODEL -notmatch '^gemini') {
    Write-Error "ERROR: LLM_MODEL must be a Gemini model for this script. Current LLM_MODEL='$LLM_MODEL'. Set `$env:LLM_MODEL='gemini-3.5-flash' or remove the stale environment variable."
    exit 1
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
    $FEW_SHOT_EXAMPLE_SUBJECTS = 5
}

$CONCURRENCY = $env:CONCURRENCY
if (-not $CONCURRENCY) {
    $CONCURRENCY = 1
}

$LOG_EVERY = $env:LOG_EVERY
if (-not $LOG_EVERY) {
    $LOG_EVERY = 10
}

if (-not $GEMINI_API_KEY) {
    Write-Error "ERROR: GEMINI_API_KEY is not set."
    exit 1
}

$env:GEMINI_API_KEY = $GEMINI_API_KEY

if (-not (Test-Path $BASE)) {
    Write-Error "ERROR: BASE does not exist: $BASE"
    exit 1
}

if (-not (Test-Path $PROCESSED_ROOT)) {
    Write-Error "ERROR: PROCESSED_ROOT does not exist: $PROCESSED_ROOT"
    exit 1
}

if (-not (Test-Path $SOURCE_STATUS_CSV)) {
    Write-Error "ERROR: SOURCE_STATUS_CSV does not exist: $SOURCE_STATUS_CSV"
    exit 1
}

$mainPy = Join-Path $BASE "main.py"
if (-not (Test-Path $mainPy)) {
    Write-Error "ERROR: main.py not found in BASE: $mainPy"
    exit 1
}

New-Item -ItemType Directory -Path $RETRY_LOGROOT -Force | Out-Null
Set-Location $BASE

$venvActivate = Join-Path $BASE ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

$retryStatusCsv = Join-Path $RETRY_LOGROOT "retry_status.csv"
"dataset,input,lm,output,status,start_time,end_time,log" | Out-File -FilePath $retryStatusCsv -Encoding utf8

Write-Host "Retry started at $(Get-Date)"
Write-Host "Base: $BASE"
Write-Host "Processed root: $PROCESSED_ROOT"
Write-Host "Source status CSV: $SOURCE_STATUS_CSV"
Write-Host "Retry log dir: $RETRY_LOGROOT"
Write-Host "Model: $LLM_MODEL"
Write-Host "Subset level: $SUBSET_LEVEL"
Write-Host "Few-shot train subset: $FEW_SHOT_TRAIN_SUBSET_LEVEL"
Write-Host "Few-shot example subjects: $FEW_SHOT_EXAMPLE_SUBJECTS"
Write-Host "Concurrency: $CONCURRENCY"

$failedRows = Import-Csv $SOURCE_STATUS_CSV | Where-Object {
    $_.status -ne "0" -and
    $_.dataset -ne "" -and
    $_.input -ne "" -and
    $_.lm -ne "" -and
    $_.output -ne ""
}

if (-not $failedRows -or $failedRows.Count -eq 0) {
    Write-Host "No failed rows found in source status CSV."
    exit 0
}

Write-Host "Failed combinations to retry: $($failedRows.Count)"

$script:failures = 0

foreach ($row in $failedRows) {
    $dataset = $row.dataset
    $inputType = $row.input
    $lm = $row.lm
    $output = $row.output

    $startTime = Get-Date -Format o
    $logFile = Join-Path $RETRY_LOGROOT "${dataset}_${inputType}_${lm}_${output}_retry.log"
    $evalCache = Join-Path $PROCESSED_ROOT "LLMSubsets\$dataset\$SUBSET_LEVEL\${dataset}_${inputType}_${SUBSET_LEVEL}_samples.pkl"

    Write-Host "=================================================="
    Write-Host "Retrying: dataset=$dataset input=$inputType lm=$lm output=$output"
    Write-Host "Eval cache: $evalCache"
    Write-Host "Log: $logFile"

    if (-not (Test-Path $evalCache)) {
        Write-Host "Missing eval cache: $evalCache"
        $endTime = Get-Date -Format o
        "${dataset},${inputType},${lm},${output},1,${startTime},${endTime},${logFile}" |
            Out-File -FilePath $retryStatusCsv -Append -Encoding utf8
        $script:failures += 1
        continue
    }

    $extraArgs = @()

    if ($lm -eq "few_shot") {
        $trainCache = Join-Path $PROCESSED_ROOT "LLMSubsets\$dataset\$FEW_SHOT_TRAIN_SUBSET_LEVEL\${dataset}_${inputType}_${FEW_SHOT_TRAIN_SUBSET_LEVEL}_samples.pkl"

        if (-not (Test-Path $trainCache)) {
            Write-Host "Missing few-shot train subset cache: $trainCache"
            Write-Host "Refusing to fall back to eval subset cache because that can leak evaluation samples into few-shot examples."

            $endTime = Get-Date -Format o
            "${dataset},${inputType},${lm},${output},1,${startTime},${endTime},${logFile}" |
                Out-File -FilePath $retryStatusCsv -Append -Encoding utf8
            $script:failures += 1
            continue
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
        "--lm-provider", $LM_PROVIDER,
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
        $script:failures += 1
        Write-Host "FAILED again: dataset=$dataset input=$inputType lm=$lm output=$output status=$status"
    }
    else {
        Write-Host "OK after retry: dataset=$dataset input=$inputType lm=$lm output=$output"
    }

    "${dataset},${inputType},${lm},${output},${status},${startTime},${endTime},${logFile}" |
        Out-File -FilePath $retryStatusCsv -Append -Encoding utf8
}

Write-Host "=================================================="
Write-Host "Retry finished at $(Get-Date)"
Write-Host "Failed combinations retried: $($failedRows.Count)"
Write-Host "Failures after retry: $script:failures"
Write-Host "Retry logs saved in: $RETRY_LOGROOT"
Write-Host "Retry status CSV: $retryStatusCsv"

if ($script:failures -ne 0) {
    exit 1
}
else {
    exit 0
}
