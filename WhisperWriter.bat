@echo off
cd /d "%~dp0"
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1
echo Lancement de WhisperWriter... chargement du modele 10-15 s
".venv\Scripts\python.exe" run.py
echo.
echo Fenetre fermee. Code %errorlevel%
pause
