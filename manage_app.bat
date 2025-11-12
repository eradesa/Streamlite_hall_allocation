@echo off
setlocal enabledelayedexpansion

:: Configuration
set "VENV_ACTIVATE=myenv\Scripts\activate.bat"
set "APP_DIR=Interface"
set "app.py"
set "PID_FILE=streamlit_app.pid"
set "LOCK_FILE=app_running.lock"
:: streamlit run app.py

:main_menu
cls
echo.
echo  ===================================
echo    STREAMLIT APPLICATION CONTROLLER
echo  ===================================
echo.
echo   [1] Start Application
echo   [2] Stop Application
echo   [3] Restart Application
echo   [4] Exit
echo.
choice /c 1234 /n /m "Select option: "

if errorlevel 4 exit /b
if errorlevel 3 goto restart_app
if errorlevel 2 goto stop_app
if errorlevel 1 goto start_app

:start_app
if exist "%LOCK_FILE%" (
    echo.
    echo Error: Application is already running!
    timeout /t 2 >nul
    goto main_menu
)

echo.
echo Starting application...
echo Creating lock file...
echo. > "%LOCK_FILE%"

:: Start in new window and capture PID
start "Streamlit_APP" /MIN cmd /c "(%VENV_ACTIVATE% && cd %APP_DIR% && streamlit run %APP_SCRIPT%) && (del "%LOCK_FILE%" 2>nul)"

echo Application started successfully!
echo.
timeout /t 2 >nul
goto main_menu

:stop_app
if not exist "%LOCK_FILE%" (
    echo.
    echo Error: No running application found!
    timeout /t 2 >nul
    goto main_menu
)

echo.
echo Stopping application...
taskkill /FI "WINDOWTITLE eq Streamlit_APP*" /T /F >nul 2>&1
del "%LOCK_FILE%" 2>nul

echo Application stopped successfully!
echo.
timeout /t 2 >nul
goto main_menu

:restart_app
call :stop_app
echo.
echo Restarting application...
timeout /t 2 >nul
call :start_app
goto main_menu