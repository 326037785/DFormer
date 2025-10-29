@echo off
setlocal enabledelayedexpansion

rem === Configuration ========================================================
set "GPUS=8"
set "NNODES=1"
if "%NODE_RANK%"=="" set "NODE_RANK=0"
if "%PORT%"=="" set "PORT=29158"
if "%MASTER_ADDR%"=="" set "MASTER_ADDR=127.0.0.1"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

set "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
set "TORCHDYNAMO_VERBOSE=1"

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
    utils/eval.py ^
    --config=local_configs.NYUDepthv2.DFormerv2_S ^
    --gpus=%GPUS% ^
    --sliding ^
    --no-compile ^
    --syncbn ^
    --mst ^
    --compile_mode=reduce-overhead ^
    --amp ^
    --pad_SUNRGBD ^
    --continue_fpath=checkpoints/trained/DFormerv2_Small_NYU.pth

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
