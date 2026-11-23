@echo off
cd /d "%~dp0"
set NO_PROXY=localhost,127.0.0.1
set GRADIO_ANALYTICS_ENABLED=False
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" app.py
pause
