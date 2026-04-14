param(
    [string]$SourceDir = $(Join-Path $PSScriptRoot "data\Original from tiago"),
    [string]$PythonExe = "C:\Users\Arriscado\envs\crushenv\Scripts\python.exe",
    [string]$OptimizerScript = $(Join-Path $PSScriptRoot "ribOptimizer(v7).py"),
    [ValidateSet("maxCF", "minP", "maxFPP", "boxMaxCF")]
    [string]$Mode = "minP",
    [string[]]$ExcludeNames = @(
        "partition_data_layer31_z157.6.mat",
        "partition_data_layer32_z167.6.mat",
        "partition_data_layer33_z177.6.mat",
        "partition_data_layer34_z187.6.mat",
        "partition_data_layer35_z197.6.mat",
        "partition_data_layer36_z207.6.mat"
    ),
    [string[]]$ExcludePatterns = @(),
    [switch]$ContinueOnError,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw "SourceDir not found: $SourceDir"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $OptimizerScript)) {
    throw "Optimizer script not found: $OptimizerScript"
}

$files = Get-ChildItem -LiteralPath $SourceDir -Filter "partition_data_*.mat" -File |
    Sort-Object Name

if ($ExcludeNames.Count -gt 0) {
    $normalizedExcludeNames = @()
    foreach ($item in $ExcludeNames) {
        if ([string]::IsNullOrWhiteSpace($item)) {
            continue
        }

        $trimmed = $item.Trim()
        $normalizedExcludeNames += $trimmed

        if (-not $trimmed.EndsWith('.mat', [System.StringComparison]::OrdinalIgnoreCase)) {
            $normalizedExcludeNames += "$trimmed.mat"
        }

        $stem = [System.IO.Path]::GetFileNameWithoutExtension($trimmed)
        if (-not [string]::IsNullOrWhiteSpace($stem)) {
            $normalizedExcludeNames += $stem
        }
    }

    $normalizedExcludeNames = $normalizedExcludeNames |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Unique

    $files = $files | Where-Object {
        $name = $_.Name
        $stem = $_.BaseName
        ($normalizedExcludeNames -notcontains $name) -and
        ($normalizedExcludeNames -notcontains $stem)
    }
}

if ($ExcludePatterns.Count -gt 0) {
    $files = $files | Where-Object {
        $name = $_.Name
        -not ($ExcludePatterns | Where-Object { $name -like $_ })
    }
}

if (-not $files) {
    throw "No partition_data_*.mat files found after exclusions in: $SourceDir"
}

Write-Host ""
Write-Host "Batch ribOptimizer(v7) run" -ForegroundColor Cyan
Write-Host "  SourceDir : $SourceDir"
Write-Host "  Mode      : $Mode"
Write-Host "  File count : $($files.Count)"
if ($ExcludeNames.Count -gt 0) {
    Write-Host "  ExcludeNames    : $($ExcludeNames -join ', ')"
}
if ($ExcludePatterns.Count -gt 0) {
    Write-Host "  ExcludePatterns : $($ExcludePatterns -join ', ')"
}
Write-Host ""

$results = New-Object System.Collections.Generic.List[object]
$index = 0

foreach ($file in $files) {
    $index += 1
    Write-Host ("[{0}/{1}] {2}" -f $index, $files.Count, $file.Name) -ForegroundColor Yellow

    $commandPreview = "& `"$PythonExe`" `"$OptimizerScript`" `"$($file.FullName)`" $Mode"
    Write-Host "  $commandPreview"

    if ($DryRun) {
        $results.Add([pscustomobject]@{
            File   = $file.Name
            Status = "DRY_RUN"
        })
        continue
    }

    try {
        & $PythonExe $OptimizerScript $file.FullName $Mode
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            throw "Optimizer exited with code $exitCode"
        }

        $results.Add([pscustomobject]@{
            File   = $file.Name
            Status = "OK"
        })
    }
    catch {
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $results.Add([pscustomobject]@{
            File   = $file.Name
            Status = "FAILED"
            Error  = $_.Exception.Message
        })

        if (-not $ContinueOnError) {
            break
        }
    }

    Write-Host ""
}

Write-Host ""
Write-Host "Summary" -ForegroundColor Cyan
$results | Format-Table -AutoSize
