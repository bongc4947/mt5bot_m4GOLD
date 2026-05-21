# watch_mt5.ps1 - tail the MT5 terminal log and surface only the lines
# you actually care about for a MetaTrend field-test demo run.
#
# Usage:
#   .\tools\watch_mt5.ps1                  # tail mode, follows latest log
#   .\tools\watch_mt5.ps1 -Summary         # one-shot summary of today, no follow
#   .\tools\watch_mt5.ps1 -Since "06:00"   # only events after wallclock HH:MM
#
# What it shows:
#   - EA load / remove / init failures
#   - Broker connect / disconnect / authorisation events
#   - All MetaTrend log lines (entries, exits, trend flips)
#   - All Trades and Trade events (deals + position modifies)
#   - Any line with severity > 0 (warnings + errors), with ONNX-CUDA noise filtered
#
# Run it whenever you want a quick read on what the live EA has been doing.

[CmdletBinding()]
param(
  [switch]$Summary,
  [string]$Since = "",
  [int]$Lines = 200
)

$ErrorActionPreference = "Stop"

$root = "$env:APPDATA\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"
# Two log streams to merge:
#   Logs/        - terminal events (auth, EA load/remove, network)
#   MQL5/Logs/   - EA Print() output ([MetaGate], [MetaTrend], trade prints)
$logDirTerminal = Join-Path $root "Logs"
$logDirMql5     = Join-Path $root "MQL5\Logs"
if (-not (Test-Path $logDirTerminal)) {
  Write-Error "Terminal log dir not found: $logDirTerminal"
  exit 1
}

# Patterns we keep (whitelist) and patterns we drop (blacklist over the keeps)
$Keep = @(
  'MT5bot_m4Gold',
  '\[MetaGate\]',
  '\[MetaTrend\]',
  '\tExperts\t',
  '\tTrades\t',
  '\tTrade\t',
  'authorized on',
  'connection to .* lost',
  'trading has been enabled',
  'trading has been disabled',
  'terminal synchronized',
  'expert .* loaded successfully',
  'expert .* removed',
  'expert .* failed to load'
) -join "|"

$DropNoise = @(
  '\[ort_env\]: CUDA failure 801',
  'use MetaTrader VPS Hosting Service',
  'scanning network for access points',
  'scanning network finished',
  'LiveUpdate\s+\d',
  'auto connecting to a better access point'
) -join "|"

function Get-LatestLog {
  param([string]$dir)
  Get-ChildItem $dir -Filter "*.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

function Format-LogLine {
  param([string]$line)
  # Lines look like:  ID<tab>SEV<tab>HH:MM:SS.mmm<tab>SRC<tab>TEXT
  $parts = $line -split "`t", 5
  if ($parts.Count -lt 5) { return $line }
  $sev = $parts[1]
  $time = $parts[2]
  $src = $parts[3]
  $msg = $parts[4]
  $tag = switch -Regex ($msg) {
    '\[MetaTrend\] LIVE'           { '[GO]'   ; break }
    '\[MetaTrend\] trend flipped'  { '[FLIP]' ; break }
    '\[MetaTrend\] .*entered'      { '[IN]'   ; break }
    '\[MetaTrend\] .*closed'       { '[OUT]'  ; break }
    'expert .* loaded successfully'{ '[EA+]'  ; break }
    'expert .* removed'            { '[EA-]'  ; break }
    'expert .* failed'             { '[EA!]'  ; break }
    'deal #\d+ buy'                { '[BUY]'  ; break }
    'deal #\d+ sell'               { '[SELL]' ; break }
    'authorized on'                { '[AUTH]' ; break }
    'connection .* lost'           { '[DCON]' ; break }
    default                         { if ($sev -ne "0") { "[WARN]" } else { "" } }
  }
  $sevTag = if ($sev -eq "0") { "  " } elseif ($sev -eq "1") { "! " } else { "!!" }
  return ("{0} {1} {2,-6} {3}" -f $sevTag, $time, $tag, $msg.TrimEnd())
}

function Test-LineRelevant {
  param([string]$line, [string]$since)
  if ($line -notmatch $Keep) { return $false }
  if ($line -match $DropNoise) { return $false }
  if ($since -ne "") {
    if ($line -match '^\w\w\t\d\t(\d\d:\d\d:\d\d)') {
      if ($Matches[1] -lt $since) { return $false }
    }
  }
  return $true
}

function Show-Summary {
  param($termLog, $mqlLog, [string]$since)
  Write-Output "=== Summary ==="
  Write-Output ("Terminal log: " + $(if ($termLog) {$termLog.Name + ' (' + $termLog.LastWriteTime + ')'} else {'(none)'}))
  Write-Output ("MQL5 log:     " + $(if ($mqlLog)  {$mqlLog.Name  + ' (' + $mqlLog.LastWriteTime  + ')'} else {'(none)'}))
  if ($since) { Write-Output "Filter: events >= $since" }
  Write-Output ""

  $content = @()
  if ($termLog) { $content += Get-Content $termLog.FullName }
  if ($mqlLog)  { $content += Get-Content $mqlLog.FullName }
  $matched = $content | Where-Object { Test-LineRelevant $_ $since }

  $eaLoads = ($matched | Where-Object { $_ -match 'expert .* loaded successfully' }).Count
  $eaRemoves = ($matched | Where-Object { $_ -match 'expert .* removed' }).Count
  $authOks = ($matched | Where-Object { $_ -match 'authorized on' }).Count
  $disconnects = ($matched | Where-Object { $_ -match 'connection .* lost' }).Count
  $buys = ($matched | Where-Object { $_ -match 'deal #\d+ buy' }).Count
  $sells = ($matched | Where-Object { $_ -match 'deal #\d+ sell' }).Count
  $warnsErrs = ($matched | Where-Object { $_ -match '^\w\w\t[12]\t' }).Count

  Write-Output ("  EA loaded         : {0}" -f $eaLoads)
  Write-Output ("  EA removed        : {0}" -f $eaRemoves)
  Write-Output ("  broker authorised : {0}" -f $authOks)
  Write-Output ("  disconnects       : {0}" -f $disconnects)
  Write-Output ("  buy deals         : {0}" -f $buys)
  Write-Output ("  sell deals        : {0}" -f $sells)
  Write-Output ("  warnings + errors : {0}" -f $warnsErrs)
  Write-Output ""
  Write-Output "=== Last $Lines relevant events ==="
  $matched | Select-Object -Last $Lines | ForEach-Object { Format-LogLine $_ }
}

function Start-Tail {
  param($termLog, $mqlLog)
  Write-Output ("Tailing both streams (Ctrl+C to stop):")
  Write-Output ("  terminal: " + $(if($termLog){$termLog.Name}else{'(none)'}))
  Write-Output ("  MQL5:     " + $(if($mqlLog){$mqlLog.Name}else{'(none)'}))
  Write-Output "Filtering for MetaTrend / Experts / Trades / auth / errors. Noise suppressed."
  Write-Output ""
  # Recent context from both
  $recent = @()
  if ($termLog) { $recent += Get-Content $termLog.FullName -Tail 400 }
  if ($mqlLog)  { $recent += Get-Content $mqlLog.FullName  -Tail 400 }
  $recent | Where-Object { Test-LineRelevant $_ $Since } |
    Select-Object -Last 20 | ForEach-Object { Format-LogLine $_ }
  Write-Output ("--- live tail since {0} ---" -f (Get-Date -Format "HH:mm:ss"))
  # Tail both concurrently via a runspace. Simpler: tail each in turn.
  # PowerShell can't easily Get-Content -Wait on two files without jobs.
  # We start a background job for the MQL5 log and tail terminal in foreground.
  $job = $null
  if ($mqlLog) {
    $job = Start-Job -ScriptBlock {
      param($p, $since)
      Get-Content $p -Tail 0 -Wait | Where-Object {
        if ($since -ne "" -and $_ -match '^\w\w\t\d\t(\d\d:\d\d:\d\d)') {
          if ($Matches[1] -lt $since) { return $false }
        }
        $_ -match 'MT5bot_m4Gold|\[MetaGate\]|\[MetaTrend\]|deal #|position|ONNX api'
      }
    } -ArgumentList $mqlLog.FullName, $Since
  }
  try {
    if ($termLog) {
      Get-Content $termLog.FullName -Tail 0 -Wait |
        Where-Object { Test-LineRelevant $_ $Since } |
        ForEach-Object {
          Format-LogLine $_
          # also drain MQL5 job output without blocking
          if ($job) {
            Receive-Job $job | ForEach-Object { Format-LogLine $_ }
          }
        }
    }
  } finally {
    if ($job) { Stop-Job $job; Remove-Job $job -Force }
  }
}

$termLog = Get-LatestLog $logDirTerminal
$mqlLog  = if (Test-Path $logDirMql5) { Get-LatestLog $logDirMql5 } else { $null }
if (-not $termLog -and -not $mqlLog) { Write-Error "No log files in $logDirTerminal or $logDirMql5"; exit 1 }

if ($Summary) { Show-Summary $termLog $mqlLog $Since } else { Start-Tail $termLog $mqlLog }
