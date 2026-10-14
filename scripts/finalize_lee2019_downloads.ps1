param(
    [string]$OutputRoot = "data/raw/lee2019_mi"
)

$ErrorActionPreference = "Stop"
$root = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputRoot))
$partials = Get-ChildItem -LiteralPath $root -Recurse -File -Filter "*.part"
$promoted = 0

foreach ($partial in $partials) {
    $destination = $partial.FullName.Substring(0, $partial.FullName.Length - 5)
    $relative = $destination.Substring($root.TrimEnd("\").Length).TrimStart("\").Replace("\", "/")
    $url = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/$relative"
    $headers = & curl.exe --silent --show-error --fail --head $url
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "HEAD failed: $relative"
        continue
    }
    $lengthLine = $headers | Where-Object { $_ -match "^Content-Length:" } | Select-Object -First 1
    if (-not $lengthLine) {
        Write-Warning "No Content-Length: $relative"
        continue
    }
    $remoteLength = [int64](($lengthLine -split ":", 2)[1].Trim())
    if ($partial.Length -eq $remoteLength) {
        Move-Item -LiteralPath $partial.FullName -Destination $destination -Force
        $promoted++
        Write-Output "promoted: $relative"
    } else {
        Write-Output "partial: $relative ($($partial.Length)/$remoteLength)"
    }
}

Write-Output "Promoted $promoted completed files."
