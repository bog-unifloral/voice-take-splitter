@echo off
rem Double-click this file to open the Voice take splitter window.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Install it from https://www.python.org/downloads/
  echo During setup, tick "Add python.exe to PATH". Then double-click this file again.
  pause
  exit /b 1
)
echo Checking the required packages (first run takes a minute)...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
python split_takes.py
if errorlevel 1 pause
