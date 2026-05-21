# ab_runner.ps1 - run 4 isolated v1.20 feature tests on the same window
# and capture wall-clock bounds so we can parse each from the agent log.
#
# All runs use:  GOLD, M5, 2025.11.20 -> 2026.05.21, deposit $10k, maxStack=1

$reportDir = "C:\Users\Angela Ramos\.openclaw\workspace\MT5bot_m4Gold\tester_reports"
$mt5 = "C:\Program Files\MetaTrader 5\terminal64.exe"
$from = "2025.11.20"; $to = "2026.05.21"

# Each run differs only in the [TesterInputs] block
$runs = @(
  @{ name="A_baseline";   desc="all v1.20 features OFF (should recover v1.10 PF 1.250)"; tunes="
InpUseBreakeven=false
InpUsePartialClose=false
InpUseQuantiles=false
InpTpAtr=0.0
" },
  @{ name="B_breakeven";  desc="ONLY breakeven move at +1 ATR"; tunes="
InpUseBreakeven=true
InpBreakevenAtr=1.0
InpBreakevenBuffer=0.05
InpUsePartialClose=false
InpUseQuantiles=false
InpTpAtr=0.0
" },
  @{ name="C_partial";    desc="ONLY partial close 50% at +1.5 ATR"; tunes="
InpUseBreakeven=false
InpUsePartialClose=true
InpPartialAtr=1.5
InpPartialPct=0.5
InpUseQuantiles=false
InpTpAtr=0.0
" },
  @{ name="D_quantile";   desc="ONLY quantile gate (Kelly forced 1.0x, no q50 filter, tail veto 2.5 ATR)"; tunes="
InpUseBreakeven=false
InpUsePartialClose=false
InpUseQuantiles=true
InpQ10VetoAtr=2.5
InpUseQ50Filter=false
InpKellyFraction=0.25
InpKellyMin=1.0
InpKellyMax=1.0
InpSlBufferAtr=0.0
InpTpAtr=0.0
" }
)

$baseInputs = @"
InpBaseLot=0.01
InpMaxStack=1
InpRespectDeploy=true
InpVerboseLog=false
InpSlAtr=3.0
InpUseTrailing=true
InpTrailStartAtr=2.0
InpTrailAtr=3.0
InpMaxHoldBars=288
InpExitOnFlip=true
"@

$results = @()
foreach ($r in $runs) {
  $iniPath = Join-Path $reportDir ("ab_" + $r.name + ".ini")
  $reportPath = Join-Path $reportDir ("ab_" + $r.name + ".htm")
  if (Test-Path $reportPath) { Remove-Item $reportPath -Force }

  $ini = @"
[Tester]
Expert=MT5bot_m4Gold\ea\MT5bot_m4Gold_MetaTrend.ex5
Symbol=GOLD
Period=M5
Optimization=0
Model=1
FromDate=$from
ToDate=$to
ForwardMode=0
Deposit=10000
Currency=USD
Leverage=1:100
ExecutionMode=0
ShutdownTerminal=true
Report=$reportPath
ReplaceReport=1

[TesterInputs]
$baseInputs
$($r.tunes)
"@
  [System.IO.File]::WriteAllText($iniPath, $ini, [System.Text.UnicodeEncoding]::new($false, $true))

  $start = Get-Date
  Write-Output ("[" + $r.name + "] " + $r.desc)
  Write-Output ("    start at " + $start.ToString('HH:mm:ss'))

  # be sure nothing else is alive
  Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2

  $p = Start-Process -FilePath $mt5 -ArgumentList "/config:`"$iniPath`"" -PassThru
  # Wait up to 5 min - the terminal exits itself on ShutdownTerminal=true
  $exited = $p.WaitForExit(300000)
  if (-not $exited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  $end = Get-Date
  $elapsed = ($end - $start).TotalSeconds
  Write-Output ("    end   at " + $end.ToString('HH:mm:ss') + "  elapsed " + [math]::Round($elapsed,1) + "s")

  # parse the tester log for "final balance N USD"
  $tlogDir = "$env:APPDATA\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs"
  $tl = Get-ChildItem $tlogDir -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $finalBal = $null
  $tradeCount = $null
  Get-Content $tl.FullName -Tail 200 | ForEach-Object {
    if ($_ -match 'final balance ([\d.]+) USD') { $finalBal = [double]$Matches[1] }
    if ($_ -match 'GOLD,M5:.*?bars generated') { $tradeCount = $_ }
  }
  $results += [PSCustomObject]@{
    name = $r.name
    desc = $r.desc
    final_balance = $finalBal
    net_pnl = if ($finalBal) { $finalBal - 10000 } else { $null }
    return_pct = if ($finalBal) { ($finalBal - 10000) / 10000 * 100 } else { $null }
    wc_from = $start.ToString('HH:mm:ss')
    wc_to = $end.ToString('HH:mm:ss')
  }
  Write-Output ("    final balance: $finalBal USD")
}

Write-Output ""
Write-Output "=" * 70
Write-Output "  A/B SUMMARY  (window 2025.11.20 -> 2026.05.21, deposit \$10k)"
Write-Output "=" * 70
"{0,-15} {1,15} {2,12} {3,8}  {4}" -f "run", "final_bal", "net", "%", "desc"
foreach ($r in $results) {
  $bal = if ($r.final_balance) { ('${0:N2}' -f $r.final_balance) } else { 'FAIL' }
  $net = if ($r.net_pnl -ne $null) { ('{0:+0.00;-0.00}' -f $r.net_pnl) } else { '?' }
  $pct = if ($r.return_pct -ne $null) { ('{0:+0.00;-0.00}%' -f $r.return_pct) } else { '?' }
  "{0,-15} {1,15} {2,12} {3,8}  {4}" -f $r.name, $bal, $net, $pct, $r.desc
}
Write-Output ""
Write-Output "Wall-clock windows (for agent-log deal parsing if you want PF):"
foreach ($r in $results) {
  "  {0,-15} {1} -> {2}" -f $r.name, $r.wc_from, $r.wc_to
}
