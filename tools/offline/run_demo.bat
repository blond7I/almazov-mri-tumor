@echo off
chcp 65001 >nul
rem Рабочее место врача без интернета: откроется в браузере по адресу http://127.0.0.1:7860
cd /d "%~dp0\..\.."
set PYTHONPATH=src
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_OFFLINE=1
set NO_PROXY=127.0.0.1,localhost
.venv\Scripts\python -m almazov.app
pause
