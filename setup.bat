@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"

echo Creating venv...
py -3 -m venv .venv

echo Upgrading pip...
".\.venv\Scripts\python.exe" -m pip install -U pip

echo Installing requirements...
".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
".\.venv\Scripts\python.exe" -m pip install python-dotenv

echo Setup Done.
pause