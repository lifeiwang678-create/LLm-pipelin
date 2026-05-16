$ErrorActionPreference = "Stop"

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

foreach ($inputName in $inputs) {
    foreach ($lmName in $lms) {
        foreach ($outputName in $outputs) {

            Write-Host ""
            Write-Host "========================================"
            Write-Host "Running: WESAD | $inputName | $lmName | $outputName"
            Write-Host "========================================"

            $argsList = @(
                "main.py",
                "-dataset", "WESAD",
                "-Input", $inputName,
                "-LM", $lmName,
                "-output", $outputName,
                "--labels", "1", "2", "3",
                "--balanced-per-label", "1",
                "--log-every", "1"
            )

            if ($lmName -eq "few_shot") {
                $argsList += @(
                    "--train-subjects", "S2",
                    "--test-subjects", "S3",
                    "--few-shot-n-per-class", "1",
                    "--few-shot-example-max-chars", "1200"
                )
            }
            else {
                $argsList += @(
                    "--subjects", "S2"
                )
            }

            & python @argsList

            if ($LASTEXITCODE -ne 0) {
                throw "Failed experiment: WESAD | $inputName | $lmName | $outputName"
            }
        }
    }
}