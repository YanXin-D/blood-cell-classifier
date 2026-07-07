@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM 激活项目统一 Python 环境（conda base）
call "C:\Users\lenovo\anaconda3\Scripts\activate.bat" base
set KMP_DUPLICATE_LIB_OK=TRUE
