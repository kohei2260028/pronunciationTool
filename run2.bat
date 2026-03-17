@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"

call ".\.venv\Scripts\activate.bat"

".\.venv\Scripts\python.exe" -m streamlit run src\dashboard.py

pause