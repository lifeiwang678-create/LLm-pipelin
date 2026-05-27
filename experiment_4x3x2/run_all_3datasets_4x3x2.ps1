param(
    [switch]$FullData,
    [int]$BalancedPerLabel = 1,
    [int]$LogEvery = 1,
    [int]$Concurrency = 1,
    [string]$HharDataDir = "",
    [string]$DreamtDataDir = "",
    [int]$HharMaxRows = 3500000,
    [int]$FewShotNPerClass = 1,
    [int]$FewShotExampleMaxChars = 800,
    [int]$MultiAgentIntermediateMaxTokens = 512,
    [switch]$UseInputCache,
    [string]$InputCacheDir = "Processed",
    [string[]]$WesadDirectSubjects = @("S2", "S3"),
    [string[]]$WesadFewShotTrainSubjects = @(),
    [string[]]$WesadFewShotTestSubjects = @(),
    [string[]]$HharFewShotTrainSubjects = @(),
    [string[]]$HharFewShotTestSubjects = @(),
    [string[]]$DreamtFewShotTrainSubjects = @(),
    [string[]]$DreamtFewShotTestSubjects = @(),
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

# By default, main.py uses the subject-independent split from Dataset/registry.py.
# Pass the *FewShot* arrays only when overriding those defaults.
$datasetSettings = @{
    "WESAD" = @{
        "DataDir" = $null
        # In debug mode, direct/multi_agent use a small subject subset.
        # In -FullData mode this subject filter is omitted; main.py then uses
        # the held-out test subjects from the subject-independent split.
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
    [void]$ArgsList.Add("--concurrency")
    [void]$ArgsList.Add([string]$Concurrency)
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
                    if ($settings.FewShotTrainSubjects.Count -gt 0) {
                        [void]$argsList.Add("--train-subjects")
                        foreach ($subject in $settings.FewShotTrainSubjects) {
                            [void]$argsList.Add($subject)
                        }
                    }

                    if ($settings.FewShotTestSubjects.Count -gt 0) {
                        [void]$argsList.Add("--test-subjects")
                        foreach ($subject in $settings.FewShotTestSubjects) {
                            [void]$argsList.Add($subject)
                        }
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
