@echo off
setlocal enabledelayedexpansion

rem === Configuration ========================================================
set "GPUS=2"
set "NNODES=1"
if "%NODE_RANK%"=="" set "NODE_RANK=0"
if "%PORT%"=="" set "PORT=29958"
if "%MASTER_ADDR%"=="" set "MASTER_ADDR=127.0.0.1"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

rem === Python path setup ====================================================
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%") do set "PARENT_DIR=%%~dpI"
if defined PARENT_DIR set "PARENT_DIR=%PARENT_DIR:~0,-1%"
if defined PYTHONPATH (
    set "PYTHONPATH=%PARENT_DIR%;%SCRIPT_DIR%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%PARENT_DIR%;%SCRIPT_DIR%"
)

rem === Launch ===============================================================
call :run ^
    --nnodes=%NNODES% ^
    --node_rank=%NODE_RANK% ^
    --master_addr=%MASTER_ADDR% ^
    --nproc_per_node=%GPUS% ^
    --master_port=%PORT% ^
    utils/infer.py ^
    --config=local_configs.NYUDepthv2.DFormer_Large ^
    --continue_fpath=checkpoints/trained/NYUv2_DFormer_Large.pth ^
    --save_path=output ^
    --gpus=%GPUS%

exit /b %errorlevel%

:run
setlocal
set "ARGS="
:collect
if "%~1"=="" goto launch
set "ARGS=%ARGS% %~1"
shift
goto collect
:launch
"%PYTHON_CMD%" -m torch.distributed.run %ARGS%
if errorlevel 1 exit /b 1
exit /b 0
