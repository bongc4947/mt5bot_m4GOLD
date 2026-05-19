@echo off
REM train.bat — local Windows entrypoint for MT5bot_m4Gold.
REM
REM Picks up GPU automatically if a CUDA-capable Python + torch is on PATH;
REM falls back to CPU otherwise. Set FORCE_CPU=1 to override.
REM
REM Usage:
REM   train.bat                ^&^& trains MetaTrend — the validated edge (default)
REM   train.bat strategies     ^&^& LEGACY H1/H4/H5/H6 rule stack (superseded)
REM   train.bat ai             ^&^& LEGACY AI direction head (no edge — retired)
REM   train.bat aurum          ^&^& AURUM v2 research pipeline
REM   train.bat help           ^&^& show options

setlocal
set "MODE=%~1"
REM Default = metatrend: the one validated, deployable edge (PF ~1.41,
REM leak-free purged CV). strategies/ai/aurum are legacy/research paths
REM that the edge search superseded — run them explicitly if needed.
if "%MODE%"=="" set "MODE=metatrend"
if "%MODE%"=="help" goto :help

if not defined EPOCHS set "EPOCHS=60"
if not defined SEED   set "SEED=42"
if not defined STRATEGIES set "STRATEGIES=H1,H4,H5,H6"

REM Hardware probe
python -c "import sys; sys.path.insert(0,'python'); from hardware_detector import detect; hw=detect(); print(f'[m4Gold] device={hw.device} batch={hw.batch_size} tier={hw.tier}')"

if /I "%MODE%"=="metatrend"  goto :metatrend
if /I "%MODE%"=="aurum"      goto :aurum
if /I "%MODE%"=="strategies" goto :strategies
if /I "%MODE%"=="ai"         goto :ai
if /I "%MODE%"=="both"       goto :strategies

:metatrend
echo === MetaTrend (the validated edge — EMA trend + XGBoost meta-gate) ===
python python\train_h7_metatrend.py
if errorlevel 1 exit /b 1
goto :done

:aurum
echo === AURUM v2 (full phased pipeline) ===
python python\train_aurum.py all --epochs %EPOCHS%
if errorlevel 1 exit /b 1
goto :done

:strategies
echo === strategies (%STRATEGIES%) ===
python python\train_strategies.py GOLD --strategies %STRATEGIES% --seed %SEED%
if errorlevel 1 exit /b 1
if /I "%MODE%"=="strategies" goto :done

:ai
echo === AI direction head ===
python python\train.py gold --epochs %EPOCHS% --seed %SEED% --skip-extract
if errorlevel 1 exit /b 1

:done
echo Done. Artifacts in onnx_out\ (or MT5 Common Files if MT5_COMMON_DIR set).
exit /b 0

:help
echo MT5bot_m4Gold local trainer
echo.
echo Usage: train.bat [metatrend^|strategies^|ai^|aurum^|help]
echo.
echo   metatrend - the VALIDATED edge, PF ~1.41 (default; docs/DEPLOY_METATREND.md)
echo   strategies- LEGACY H1/H4/H5/H6 rule stack (superseded — 0/4 had edge)
echo   ai        - LEGACY M5-direction AI head (retired — no edge)
echo   aurum     - AURUM v2 transformer research pipeline (docs/DESIGN_AURUM.md)
echo.
echo Env vars:
echo   EPOCHS=60          AI head epochs
echo   SEED=42            RNG seed
echo   STRATEGIES=H1,H4,H5,H6   subset
echo   FORCE_CPU=1        skip GPU detection
exit /b 0
