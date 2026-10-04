@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: py -3.12 scripts\bootstrap.py --device cuda
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m basketball_cv.cli ui
pause
