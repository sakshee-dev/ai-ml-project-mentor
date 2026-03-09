@echo off
setlocal

echo ==========================================
echo AI ML Project Mentor - Environment Setup
echo ==========================================

REM -------------------------------------------------
REM Check if Python 3.12 exists
REM -------------------------------------------------

echo Checking for Python 3.12...

py -3.12 --version >nul 2>&1

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Python 3.12 is not installed.
    echo Please install Python 3.12 from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Python 3.12 detected.

REM Delete existing venv

IF EXIST venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

REM Create new venv using Python 3.12

echo Creating virtual environment...

py -3.12 -m venv venv

IF %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

REM -------------------------------------------------
REM Activate environment
REM -------------------------------------------------

echo Activating virtual environment...

call venv\Scripts\activate

REM -------------------------------------------------
REM Verify Python version
REM -------------------------------------------------

echo Verifying Python version...

python --version

FOR /F "tokens=2 delims= " %%G IN ('python --version') DO set PYVER=%%G

echo Detected Python version: %PYVER%

echo %PYVER% | findstr /B "3.12" >nul

IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Virtual environment is not using Python 3.12
    pause
    exit /b 1
)

REM -------------------------------------------------
REM Upgrade pip
REM -------------------------------------------------

echo Upgrading pip...

python -m pip install --upgrade pip

REM -------------------------------------------------
REM Install requirements
REM -------------------------------------------------

echo Installing dependencies...

pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Environment setup complete!
echo ==========================================
echo.

echo Virtual environment is ACTIVE.
echo To activate later run:
echo venv\Scripts\activate

pause