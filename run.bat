@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"

".\.venv\Scripts\python.exe" src\main.py

pause