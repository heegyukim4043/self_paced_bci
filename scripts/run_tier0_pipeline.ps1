param(
    [string]$DataRoot = "data/raw/lee2019_mi",
    [string]$CacheRoot = "data/interim/lee2019_tier0",
    [string]$OutputRoot = "data/processed"
)

$ErrorActionPreference = "Stop"
py scripts/prepare_tier0_cache.py `
    --data-root $DataRoot `
    --output-root $CacheRoot `
    --all `
    --sessions 1 2
if ($LASTEXITCODE -ne 0) {
    throw "Tier 0 feature preparation failed."
}

foreach ($session in 1..2) {
    $result = Join-Path $OutputRoot "tier0_loso_session${session}.json"
    $summary = Join-Path $OutputRoot "tier0_loso_session${session}_summary.json"
    py scripts/run_tier0_loso.py `
        --cache-root $CacheRoot `
        --session $session `
        --target-fp 1 2 4 `
        --decisions threshold ma_dwell accumulator `
        --decision-step 0.1 `
        --output $result
    if ($LASTEXITCODE -ne 0) {
        throw "LOSO failed for session $session."
    }
    py scripts/summarize_tier0_loso.py $result --output $summary
    if ($LASTEXITCODE -ne 0) {
        throw "Summary failed for session $session."
    }
}

Write-Output "Tier 0 Lee2019 pipeline completed."
