@echo off
REM Launch the local Gemma 3 chat using the project's virtual environment.
cd /d "%~dp0"
".venv\Scripts\python.exe" chat_app.py
