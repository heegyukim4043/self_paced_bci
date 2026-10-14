param(
    [string]$OutputRoot = "data/raw/lee2019_mi",
    [int]$FirstSubject = 1,
    [int]$LastSubject = 54,
    [int]$Parallel = 4
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
$logRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) "logs/lee2019_mi"))
New-Item -ItemType Directory -Force -Path $root, $logRoot | Out-Null

$queue = [System.Collections.Generic.Queue[object]]::new()
foreach ($session in 1..2) {
    foreach ($subject in $FirstSubject..$LastSubject) {
        $relative = "session{0}/s{1}/sess{0:d2}_subj{1:d2}_EEG_MI.mat" -f $session, $subject
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
    $destinationDir = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

    if (Test-Path -LiteralPath $destination) {
        return [PSCustomObject]@{ Skipped = $true; Item = $Item; Destination = $destination }
    }

    $logBase = "sess{0:d2}_subj{1:d2}" -f $Item.Session, $Item.Subject
    $stdout = Join-Path $logRoot "$logBase.out.log"
    $stderr = Join-Path $logRoot "$logBase.err.log"
    $arguments = @(
        "--location",
        "--fail",
        "--retry", "5",
        "--retry-delay", "5",
        "--continue-at", "-",
        "--output", $partial,
        $Item.Url
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
            Write-Output "[$completed/$total] exists: $($entry.Item.Relative)"
        } else {
            $active += $entry
            Write-Output "started: $($entry.Item.Relative)"
        }
    }

    $stillActive = @()
    foreach ($entry in $active) {
        if ($entry.Process.HasExited) {
            $entry.Process.WaitForExit()
            $entry.Process.Refresh()
            $exitCode = $entry.Process.ExitCode
            $curlCompleted = $exitCode -eq 0
            if ($null -eq $exitCode -and (Test-Path -LiteralPath $entry.Partial)) {
                $curlCompleted = Select-String -LiteralPath (
                    Join-Path $logRoot (
                        "sess{0:d2}_subj{1:d2}.err.log" -f `
                            $entry.Item.Session, $entry.Item.Subject
                    )
                ) -Pattern "^100 " -Quiet
            }
            if ($curlCompleted) {
                Move-Item -LiteralPath $entry.Partial -Destination $entry.Destination -Force
                $completed++
                Write-Output "[$completed/$total] complete: $($entry.Item.Relative)"
            } else {
                $failures += $entry.Item.Relative
                Write-Warning "failed (exit $exitCode): $($entry.Item.Relative)"
            }
        } else {
            $stillActive += $entry
        }
    }
    $active = $stillActive

    if ($active.Count -gt 0) {
        Start-Sleep -Seconds 2
    }
}

if ($failures.Count -gt 0) {
    Write-Error ("Downloads failed:`n" + ($failures -join "`n"))
}

Write-Output "Lee2019 MI download complete: $completed/$total files."
