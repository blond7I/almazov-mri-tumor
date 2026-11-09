@echo off
chcp 65001 >nul
rem Установка офлайн-демо (один раз, при хорошем интернете): Python-окружение + проверка, что модели собираются.
cd /d "%~dp0\..\.."
where py >nul 2>nul
if errorlevel 1 (
  echo Нужен Python 3.11 или 3.12: https://www.python.org/downloads/  ^(при установке поставьте галочку "Add python.exe to PATH"^)
  pause
  exit /b 1
)
if not exist .venv ( py -3 -m venv .venv )
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 ( echo Не удалось поставить PyTorch. Проверьте интернет и запустите ещё раз. & pause & exit /b 1 )
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 ( echo Не удалось поставить зависимости. Проверьте интернет и запустите ещё раз. & pause & exit /b 1 )
set PYTHONPATH=src
.venv\Scripts\python tools\offline\check_offline.py
if errorlevel 1 ( echo Проверка не прошла — пришлите скриншот этого окна. & pause & exit /b 1 )
echo.
echo Готово. Демо запускается файлом tools\offline\run_demo.bat — интернет больше не нужен.
pause
