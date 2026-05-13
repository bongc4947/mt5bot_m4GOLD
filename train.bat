@echo off
REM train.bat — local Windows entrypoint for MT5bot_m4Gold.
REM
REM Picks up GPU automatically if a CUDA-capable Python + torch is on PATH;
REM falls back to CPU otherwise. Set FORCE_CPU=1 to override.
REM
REM Usage:
REM   train.bat                ^&^& trains all strategies + AI head on GOLD
REM   train.bat strategies     ^&^& strategies only
REM   train.bat ai             ^&^& AI direction head only
REM   train.bat help           ^&^& show options

setlocal
set "MODE=%~1"
if "%MODE%"=="" set "MODE=both"
if "%MODE%"=="help" goto :help

if not defined EPOCHS set "EPOCHS=60"
if not defined SEED   set "SEED=42"
if not defined STRATEGIES set "STRATEGIES=H1,H4,H5,H6"

REM Hardware probe
python -c "import sys; sys.path.insert(0,'python'); from hardware_detector import detect; hw=detect(); print(f'[m4Gold] device={hw.device} batch={hw.batch_size} tier={hw.tier}')"

if /I "%MODE%"=="strategies" goto :strategies
if /I "%MODE%"=="ai"         goto :ai
if /I "%MODE%"=="both"       goto :strategies

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
echo Usage: train.bat [strategies^|ai^|both^|help]
echo.
echo Env vars:
echo   EPOCHS=60          AI head epochs
echo   SEED=42            RNG seed
echo   STRATEGIES=H1,H4,H5,H6   subset
echo   FORCE_CPU=1        skip GPU detection
exit /b 0
