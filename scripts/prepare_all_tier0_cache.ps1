param(
    [string]$DataRoot = "data/raw/lee2019_mi",
    [string]$CacheRoot = "data/interim/lee2019_tier0"
)

$ErrorActionPreference = "Stop"
& py scripts/prepare_tier0_cache.py `
    --data-root $DataRoot `
    --output-root $CacheRoot `
    --all `
    --sessions 1 2
if ($LASTEXITCODE -ne 0) {
    throw "Tier 0 feature preparation failed with exit code $LASTEXITCODE."
}
