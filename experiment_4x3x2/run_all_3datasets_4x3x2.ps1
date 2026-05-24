param(
    [switch]$FullData,
    [int]$BalancedPerLabel = 1,
    [int]$LogEvery = 1,
    [string]$HharDataDir = "",
    [string]$DreamtDataDir = "",
    [int]$HharMaxRows = 3500000,
    [int]$FewShotNPerClass = 1,
    [int]$FewShotExampleMaxChars = 800,
    [int]$MultiAgentIntermediateMaxTokens = 512,
    [switch]$UseInputCache,
    [string]$InputCacheDir = "Processed",
    [string[]]$WesadDirectSubjects = @("S2", "S3"),
    [string[]]$WesadFewShotTrainSubjects = @("S2"),
    [string[]]$WesadFewShotTestSubjects = @("S3"),
    [string[]]$HharFewShotTrainSubjects = @("a"),
    [string[]]$HharFewShotTestSubjects = @("b"),
    [string[]]$DreamtFewShotTrainSubjects = @("S099"),
    [string[]]$DreamtFewShotTestSubjects = @("S100"),
    [string]$StartDataset = "",
    [string]$StartInput = "",
    [string]$StartLM = "",
    [string]$StartOutput = ""
)

$ErrorActionPreference = "Stop"

# Official 4 x 3 x 2 combinations.
$inputs = @(
    "raw_data",
    "feature_description",
    "encoded_time_series",
    "extra_knowledge"
)

$lms = @(
    "direct",
    "few_shot",
    "multi_agent"
)

$outputs = @(
    "label_only",
    "label_explanation"
)

# Adjust these subject splits if your local data has different available subjects.
# For few-shot, train and test subjects must not overlap.
$datasetSettings = @{
    "WESAD" = @{
        "DataDir" = $null
        # In debug mode, direct/multi_agent use a small subject subset.
        # In -FullData mode this subject filter is omitted, so WESAD uses all
        # available cached/loaded subjects.
        "DirectSubjects" = if ($FullData) { @() } else { $WesadDirectSubjects }
        "FewShotTrainSubjects" = $WesadFewShotTrainSubjects
        "FewShotTestSubjects" = $WesadFewShotTestSubjects
        "MaxRows" = $null
    }
    "HHAR" = @{
        "DataDir" = if ($HharDataDir) { $HharDataDir } else { $null }
        "DirectSubjects" = @()
        # HHAR users are often letter IDs such as a, b, c...; edit if needed.
        "FewShotTrainSubjects" = $HharFewShotTrainSubjects
        "FewShotTestSubjects" = $HharFewShotTestSubjects
        "MaxRows" = $HharMaxRows
    }
    "DREAMT" = @{
        "DataDir" = if ($DreamtDataDir) { $DreamtDataDir } else { $null }
        "DirectSubjects" = @()
        # Use two different DREAMT subject files. Edit if your folder uses other IDs.
        "FewShotTrainSubjects" = $DreamtFewShotTrainSubjects
        "FewShotTestSubjects" = $DreamtFewShotTestSubjects
        "MaxRows" = $null
    }
}

$useResumePoint = $StartDataset -or $StartInput -or $StartLM -or $StartOutput
$resumeReached = -not $useResumePoint

function Add-CommonArgs {
    param(
        [System.Collections.ArrayList]$ArgsList,
        [string]$DatasetName,
        [hashtable]$Settings
    )

    if ($Settings.DataDir) {
        [void]$ArgsList.Add("--data-dir")
        [void]$ArgsList.Add($Settings.DataDir)
    }

    if ($Settings.MaxRows -and -not $FullData) {
        [void]$ArgsList.Add("--max-rows")
        [void]$ArgsList.Add([string]$Settings.MaxRows)
    }

    if (-not $FullData) {
        [void]$ArgsList.Add("--balanced-per-label")
        [void]$ArgsList.Add([string]$BalancedPerLabel)
    }

    if ($UseInputCache) {
        [void]$ArgsList.Add("--use-input-cache")
        [void]$ArgsList.Add("--input-cache-dir")
        [void]$ArgsList.Add($InputCacheDir)
    }

    [void]$ArgsList.Add("--log-every")
    [void]$ArgsList.Add([string]$LogEvery)
}

foreach ($datasetName in @("WESAD", "HHAR", "DREAMT")) {
    $settings = $datasetSettings[$datasetName]

    foreach ($inputName in $inputs) {
        foreach ($lmName in $lms) {
            foreach ($outputName in $outputs) {
                if (-not $resumeReached) {
                    $matchesResumePoint =
                        ($datasetName -eq $StartDataset) -and
                        ($inputName -eq $StartInput) -and
                        ($lmName -eq $StartLM) -and
                        ($outputName -eq $StartOutput)

                    if ($matchesResumePoint) {
                        $resumeReached = $true
                    }
                    else {
                        Write-Host "Skipping before resume point: $datasetName | $inputName | $lmName | $outputName"
                        continue
                    }
                }

                Write-Host ""
                Write-Host "========================================"
                Write-Host "Running: $datasetName | $inputName | $lmName | $outputName"
                Write-Host "========================================"

                $argsList = [System.Collections.ArrayList]@(
                    "main.py",
                    "-dataset", $datasetName,
                    "-Input", $inputName,
                    "-LM", $lmName,
                    "-output", $outputName
                )

                Add-CommonArgs -ArgsList $argsList -DatasetName $datasetName -Settings $settings

                if ($lmName -eq "few_shot") {
                    [void]$argsList.Add("--train-subjects")
                    foreach ($subject in $settings.FewShotTrainSubjects) {
                        [void]$argsList.Add($subject)
                    }

                    [void]$argsList.Add("--test-subjects")
                    foreach ($subject in $settings.FewShotTestSubjects) {
                        [void]$argsList.Add($subject)
                    }

                    [void]$argsList.Add("--few-shot-n-per-class")
                    [void]$argsList.Add([string]$FewShotNPerClass)
                    [void]$argsList.Add("--few-shot-example-max-chars")
                    [void]$argsList.Add([string]$FewShotExampleMaxChars)
                }
                elseif ($settings.DirectSubjects.Count -gt 0) {
                    [void]$argsList.Add("--subjects")
                    foreach ($subject in $settings.DirectSubjects) {
                        [void]$argsList.Add($subject)
                    }
                }

                if ($lmName -eq "multi_agent") {
                    [void]$argsList.Add("--multi-agent-intermediate-max-tokens")
                    [void]$argsList.Add([string]$MultiAgentIntermediateMaxTokens)
                }

                & python @argsList

                if ($LASTEXITCODE -ne 0) {
                    throw "Failed experiment: $datasetName | $inputName | $lmName | $outputName"
                }
            }
        }
    }
}
