# determinism_test.ps1 - 4 Tester runs to verify whether Tester is
# deterministic on the 18-feat path vs the 24-feat sr_fib path.
#
# Hypothesis: if 18-feat run twice produces matching balances but 24-feat
# differs across runs, the sr_fib MQL5 code has state-dependent behaviour
# (most likely H1 history pull timing or swing-array reuse).

$workspace = "C:\Users\Angela Ramos\.openclaw\workspace\MT5bot_m4Gold"
$common    = "$env:APPDATA\MetaQuotes\Terminal\Common\Files"
$repo      = Join-Path $workspace "onnx_out"
$mt5       = "C:\Program Files\MetaTrader 5\terminal64.exe"
$tlog      = "$env:APPDATA\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260602.log"
$alog      = "$env:APPDATA\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\logs\20260602.log"

# Single tester.ini used for all runs - lot=0.01, maxStack=1, identical inputs
$ini = Join-Path $workspace "tester_reports\determinism.ini"
$iniContent = @"
[Tester]
Expert=MT5bot_m4Gold\ea\MT5bot_m4Gold_MetaTrend.ex5
Symbol=GOLD
Period=M5
Optimization=0
Model=1
FromDate=2025.11.20
ToDate=2026.05.21
ForwardMode=0
Deposit=10000
Currency=USD
Leverage=1:100
ExecutionMode=0
ShutdownTerminal=true

[TesterInputs]
InpBaseLot=0.01
InpMaxStack=1
InpRespectDeploy=true
InpVerboseLog=false
InpSlAtr=3.0
InpTpAtr=0.0
InpUseBreakeven=true
InpBreakevenAtr=1.0
InpBreakevenBuffer=0.05
InpUsePartialClose=false
InpUseTrailing=true
InpTrailStartAtr=2.0
InpTrailAtr=3.0
InpMaxHoldBars=288
InpExitOnFlip=true
InpUseQuantiles=false
"@
[System.IO.File]::WriteAllText($ini, $iniContent, [System.Text.UnicodeEncoding]::new($false, $true))

function Stage-Bundle {
    param([string]$variant)
    # variant: "18feat" or "24feat"
    if ($variant -eq "18feat") {
        # the 18-feat bundle is in onnx_out/M4GOLD_METATREND_GOLD.{onnx,_spec.json}
        # (most recent train_h7_metatrend.py without --with-sr-fib)
        Copy-Item (Join-Path $repo "M4GOLD_METATREND_GOLD.onnx")      (Join-Path $common "M4GOLD_METATREND_GOLD.onnx") -Force
        Copy-Item (Join-Path $repo "M4GOLD_METATREND_GOLD_spec.json") (Join-Path $common "M4GOLD_METATREND_GOLD_spec.json") -Force
    } else {
        # the 24-feat bundle is in onnx_out/M4GOLD_METATREND_GOLD_srfib.{onnx,_spec.json}
        Copy-Item (Join-Path $repo "M4GOLD_METATREND_GOLD_srfib.onnx")      (Join-Path $common "M4GOLD_METATREND_GOLD.onnx") -Force
        Copy-Item (Join-Path $repo "M4GOLD_METATREND_GOLD_srfib_spec.json") (Join-Path $common "M4GOLD_METATREND_GOLD_spec.json") -Force
    }
    $size = (Get-Item (Join-Path $common "M4GOLD_METATREND_GOLD.onnx")).Length
    Write-Output ("    staged $variant : onnx=$size bytes")
}

function Run-Tester {
    param([string]$label)
    Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $start = Get-Date
    $p = Start-Process -FilePath $mt5 -ArgumentList "/config:`"$ini`"" -PassThru
    # Poll up to 6 min
    $deadline = $start.AddMinutes(6)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 15
        if (-not (Get-Process -Id $p.Id -ErrorAction SilentlyContinue)) { break }
    }
    $elapsed = ((Get-Date) - $start).TotalSeconds
    # Kill any leftover
    Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    # Pull latest final balance
    $bal = $null
    if (Test-Path $alog) {
        $lines = Get-Content $alog
        for ($i = $lines.Count - 1; $i -ge 0; $i--) {
            if ($lines[$i] -match "final balance ([\d.]+) USD") { $bal = [double]$Matches[1]; break }
        }
    }
    Write-Output ("    [$label] elapsed " + [math]::Round($elapsed,0) + "s  final = $" + $bal)
    return $bal
}

Write-Output "================================================================"
Write-Output "  Determinism test: 4 Tester runs, same window, same inputs"
Write-Output "================================================================"
Write-Output ""

Write-Output "Phase A: 18-feature baseline (v1.20 path)"
Stage-Bundle "18feat"
$a1 = Run-Tester "18feat A"
$a2 = Run-Tester "18feat B"

Write-Output ""
Write-Output "Phase B: 24-feature sr_fib (v1.30 path)"
Stage-Bundle "24feat"
$b1 = Run-Tester "24feat A"
$b2 = Run-Tester "24feat B"

Write-Output ""
Write-Output "================================================================"
Write-Output "  RESULTS"
Write-Output "================================================================"
$delta_18 = if ($a1 -and $a2) { [math]::Abs($a1 - $a2) } else { -1 }
$delta_24 = if ($b1 -and $b2) { [math]::Abs($b1 - $b2) } else { -1 }
"  18-feat A : `$$a1"
"  18-feat B : `$$a2     delta=`$$delta_18"
"  24-feat A : `$$b1"
"  24-feat B : `$$b2     delta=`$$delta_24"
Write-Output ""
if ($delta_18 -lt 0.01 -and $delta_24 -lt 0.01) {
    Write-Output "VERDICT: Both paths deterministic. Earlier `$10,404 vs `$11,134 was config drift, not a bug."
} elseif ($delta_18 -lt 0.01 -and $delta_24 -gt 1) {
    Write-Output "VERDICT: 18-feat deterministic, 24-feat NON-DETERMINISTIC. Bug in sr_fib code (likely H1 history)."
} elseif ($delta_18 -gt 1) {
    Write-Output "VERDICT: Even 18-feat is non-deterministic. MT5 Tester sim has state-dependent variance on this build."
} else {
    Write-Output "VERDICT: ambiguous - delta_18=`$$delta_18  delta_24=`$$delta_24"
}
