<#
Resume missing Gemini 3.5 Flash MAIN/no-agent combinations from an existing
result directory.

This script scans an existing results folder, treats every *_metrics.json file
as a completed combination, skips those combinations, and runs only the missing
ones from the 48 no-agent grid:

  3 datasets x 4 inputs x 2 LM usages x 2 outputs = 48

After each successful run, it copies the new CSV/config/metrics files from
Results/ into EXISTING_RESULTS_DIR so the script can be stopped and restarted
without rerunning completed combinations.

Usage:

  cd C:\path\to\LLm-pipelin\experiment_4x3x2

  $env:GEMINI_API_KEY = "your_gemini_api_key"
  $env:PROCESSED_ROOT = "D:\DATA\experiment_4x3x2\Processed"
  $env:EXISTING_RESULTS_DIR = "Results_by_model\gemini35flash_main_noagent_20260531_clean"
  $env:SUBSET_LEVEL = "main"
  $env:FEW_SHOT_TRAIN_SUBSET_LEVEL = "pilot"
  $env:FEW_SHOT_EXAMPLE_SUBJECTS = "3"
  $env:CONCURRENCY = "1"

  powershell -ExecutionPolicy Bypass -File .\resume_missing_gemini35flash_main_noagent.ps1
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

$EXISTING_RESULTS_DIR = $env:EXISTING_RESULTS_DIR
if (-not $EXISTING_RESULTS_DIR) {
    $EXISTING_RESULTS_DIR = Join-Path $BASE "Results_by_model\gemini35flash_main_noagent_20260531_clean"
}
elseif (-not [System.IO.Path]::IsPathRooted($EXISTING_RESULTS_DIR)) {
    $EXISTING_RESULTS_DIR = Join-Path $BASE $EXISTING_RESULTS_DIR
}

$LOGROOT = $env:LOGROOT
if (-not $LOGROOT) {
    $LOGROOT = Join-Path $BASE ('logs\resume_missing_gemini35flash_main_noagent_' + (Get-Date -Format "yyyyMMddHHmmss"))
}

$GEMINI_API_KEY = $env:GEMINI_API_KEY
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
    $FEW_SHOT_EXAMPLE_SUBJECTS = 3
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
    Write-Error "ERROR: GEMINI_API_KEY is not set. Set GEMINI_API_KEY before running this script."
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

$mainPy = Join-Path $BASE "main.py"
if (-not (Test-Path $mainPy)) {
    Write-Error "ERROR: main.py not found in BASE: $mainPy"
    exit 1
}

New-Item -ItemType Directory -Path $EXISTING_RESULTS_DIR -Force | Out-Null
New-Item -ItemType Directory -Path $LOGROOT -Force | Out-Null
Set-Location $BASE

$venvActivate = Join-Path $BASE ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "Activated virtualenv: $venvActivate"
}
else {
    Write-Warning "Virtualenv activate not found at $venvActivate - continuing without activating."
}

function Get-ComboPrefix {
    param(
        [string]$dataset,
        [string]$inputType,
        [string]$lm,
        [string]$output
    )
    return "${dataset}_${inputType}_${lm}_${output}"
}

function Get-CompletedPrefixes {
    param([string]$resultsDir)

    $completed = @{}
    if (-not (Test-Path $resultsDir)) {
        return $completed
    }

    Get-ChildItem -Path $resultsDir -File -Filter "*_metrics.json" | ForEach-Object {
        $name = $_.Name -replace "_metrics\.json$", ""
        $prefix = $name -replace "_\d{14}$", ""
        $completed[$prefix] = $true
    }
    return $completed
}

function Copy-LatestResultFiles {
    param(
        [string]$prefix,
        [datetime]$runStart,
        [string]$targetDir
    )

    $resultsDir = Join-Path $BASE "Results"
    $metrics = Get-ChildItem -Path $resultsDir -File -Filter "${prefix}_*_metrics.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $runStart.AddSeconds(-5) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $metrics) {
        Write-Warning "Could not find new metrics file for completed prefix: $prefix"
        return
    }

    $stem = $metrics.Name -replace "_metrics\.json$", ""
    $candidates = @(
        (Join-Path $resultsDir "${stem}.csv"),
        (Join-Path $resultsDir "${stem}_config.json"),
        $metrics.FullName
    )

    foreach ($source in $candidates) {
        if (Test-Path $source) {
            Copy-Item -LiteralPath $source -Destination $targetDir -Force
        }
    }
}

$datasets = @("WESAD", "HHAR", "DREAMT")
$inputTypes = @("raw_data", "feature_description", "encoded_time_series", "extra_knowledge")
$lms = @("direct", "few_shot")
$outputs = @("label_only", "label_explanation")

foreach ($dataset in $datasets) {
    foreach ($inputType in $inputTypes) {
        $subsetCache = Join-Path $PROCESSED_ROOT "LLMSubsets\$dataset\$SUBSET_LEVEL\${dataset}_${inputType}_${SUBSET_LEVEL}_samples.pkl"
        if (-not (Test-Path $subsetCache)) {
            Write-Error "Missing fixed evaluation subset cache: $subsetCache"
            exit 1
        }
    }
}

$statusCsv = Join-Path $LOGROOT "resume_missing_status.csv"
"dataset,input,lm,output,action,status,start_time,end_time,log" | Out-File -FilePath $statusCsv -Encoding utf8

Write-Host "Resume missing Gemini run started at $(Get-Date)"
Write-Host "Existing results dir: $EXISTING_RESULTS_DIR"
Write-Host "Log dir: $LOGROOT"
Write-Host "Model: $LLM_MODEL"
Write-Host "Subset level: $SUBSET_LEVEL"
Write-Host "Few-shot train subset: $FEW_SHOT_TRAIN_SUBSET_LEVEL"
Write-Host "Few-shot example subjects: $FEW_SHOT_EXAMPLE_SUBJECTS"
Write-Host "Concurrency: $CONCURRENCY"

$script:failures = 0
$script:ran = 0
$script:skipped = 0

foreach ($dataset in $datasets) {
    foreach ($inputType in $inputTypes) {
        foreach ($lm in $lms) {
            foreach ($output in $outputs) {
                $completed = Get-CompletedPrefixes -resultsDir $EXISTING_RESULTS_DIR
                $prefix = Get-ComboPrefix -dataset $dataset -inputType $inputType -lm $lm -output $output
                $startTime = Get-Date -Format o

                if ($completed.ContainsKey($prefix)) {
                    $script:skipped += 1
                    Write-Host "SKIP completed: $prefix"
                    "${dataset},${inputType},${lm},${output},skip,0,${startTime},${startTime}," |
                        Out-File -FilePath $statusCsv -Append -Encoding utf8
                    continue
                }

                $runStart = Get-Date
                $logFile = Join-Path $LOGROOT "${prefix}.log"
                $evalCache = Join-Path $PROCESSED_ROOT "LLMSubsets\$dataset\$SUBSET_LEVEL\${dataset}_${inputType}_${SUBSET_LEVEL}_samples.pkl"
                $trainCache = $evalCache
                $extraArgs = @()

                Write-Host "=================================================="
                Write-Host "RUN missing: $prefix"
                Write-Host "Eval cache: $evalCache"
                Write-Host "Log: $logFile"

                if ($lm -eq "few_shot") {
                    $trainCache = Join-Path $PROCESSED_ROOT "LLMSubsets\$dataset\$FEW_SHOT_TRAIN_SUBSET_LEVEL\${dataset}_${inputType}_${FEW_SHOT_TRAIN_SUBSET_LEVEL}_samples.pkl"
                    if (-not (Test-Path $trainCache)) {
                        Write-Host "Missing few-shot train subset cache: $trainCache"
                        $endTime = Get-Date -Format o
                        "${dataset},${inputType},${lm},${output},run,1,${startTime},${endTime},${logFile}" |
                            Out-File -FilePath $statusCsv -Append -Encoding utf8
                        $script:failures += 1
                        continue
                    }

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
                    "--lm-provider", "gemini",
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
                    Write-Host "FAILED: $prefix status=$status"
                }
                else {
                    $script:ran += 1
                    Write-Host "OK: $prefix"
                    Copy-LatestResultFiles -prefix $prefix -runStart $runStart -targetDir $EXISTING_RESULTS_DIR
                }

                "${dataset},${inputType},${lm},${output},run,${status},${startTime},${endTime},${logFile}" |
                    Out-File -FilePath $statusCsv -Append -Encoding utf8
            }
        }
    }
}

Write-Host "=================================================="
Write-Host "Resume missing Gemini run finished at $(Get-Date)"
Write-Host "Skipped completed: $script:skipped"
Write-Host "New runs completed: $script:ran"
Write-Host "Failures: $script:failures"
Write-Host "Status CSV: $statusCsv"

if ($script:failures -ne 0) {
    exit 1
}
exit 0
