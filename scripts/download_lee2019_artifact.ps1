param(
    [string]$OutputRoot = "data/raw/lee2019_artifact",
    [int]$FirstSubject = 1,
    [int]$LastSubject = 54,
    [int]$Parallel = 6
)

$ErrorActionPreference = "Stop"

if ($FirstSubject -lt 1 -or $LastSubject -gt 54 -or $FirstSubject -gt $LastSubject) {
    throw "Subject range must satisfy 1 <= FirstSubject <= LastSubject <= 54."
}
if ($Parallel -lt 1 -or $Parallel -gt 8) {
    throw "Parallel must be between 1 and 8."
}

$baseUrl = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542"
$root = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputRoot))
$logRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) "logs/lee2019_artifact"))
New-Item -ItemType Directory -Force -Path $root, $logRoot | Out-Null

$queue = [System.Collections.Generic.Queue[object]]::new()
foreach ($session in 1..2) {
    foreach ($subject in $FirstSubject..$LastSubject) {
        $relative = "session{0}/s{1}/sess{0:d2}_subj{1:d2}_EEG_Artifact.mat" -f $session, $subject
        $queue.Enqueue([PSCustomObject]@{
            Session = $session
            Subject = $subject
            Relative = $relative
            Url = "$baseUrl/$relative"
        })
    }
}

$active = @()
$failures = @()
$completed = 0
$total = $queue.Count

function Start-Download {
    param([object]$Item)

    $destination = Join-Path $root $Item.Relative
    $partial = "$destination.part"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null

    if (Test-Path -LiteralPath $destination) {
        return [PSCustomObject]@{ Skipped = $true; Item = $Item }
    }

    $logBase = "sess{0:d2}_subj{1:d2}" -f $Item.Session, $Item.Subject
    $stdout = Join-Path $logRoot "$logBase.out.log"
    $stderr = Join-Path $logRoot "$logBase.err.log"
    $arguments = @(
        "--location", "--fail", "--retry", "5", "--retry-delay", "3",
        "--continue-at", "-", "--output", $partial, $Item.Url
    )
    $process = Start-Process -FilePath "curl.exe" -ArgumentList $arguments `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr

    return [PSCustomObject]@{
        Skipped = $false
        Item = $Item
        Process = $process
        Partial = $partial
        Destination = $destination
    }
}

while ($queue.Count -gt 0 -or $active.Count -gt 0) {
    while ($queue.Count -gt 0 -and $active.Count -lt $Parallel) {
        $entry = Start-Download -Item $queue.Dequeue()
        if ($entry.Skipped) {
            $completed++
        } else {
            $active += $entry
        }
    }

    $stillActive = @()
    foreach ($entry in $active) {
        if ($entry.Process.HasExited) {
            $entry.Process.WaitForExit()
            $entry.Process.Refresh()
            if ($entry.Process.ExitCode -eq 0) {
                Move-Item -LiteralPath $entry.Partial -Destination $entry.Destination -Force
                $completed++
                Write-Output "[$completed/$total] $($entry.Item.Relative)"
            } else {
                $failures += $entry.Item.Relative
                Write-Warning "failed: $($entry.Item.Relative)"
            }
        } else {
            $stillActive += $entry
        }
    }
    $active = $stillActive
    if ($active.Count -gt 0) {
        Start-Sleep -Seconds 1
    }
}

if ($failures.Count -gt 0) {
    throw ("Artifact downloads failed:`n" + ($failures -join "`n"))
}
Write-Output "Lee2019 Artifact download complete: $completed/$total files."
