@echo off
echo ========================================
echo BUILDING DOCKER IMAGE
echo ========================================
echo.

echo Step 1: Checking Docker installation...
docker --version
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo.
echo Step 2: Building Docker image...
docker build -t ultimate-cv-system:latest .

if %errorlevel% neq 0 (
    echo ERROR: Docker build failed!
    echo.
    echo Troubleshooting:
    echo 1. Make sure Docker Desktop is running
    echo 2. Check Dockerfile exists in current directory
    echo 3. Try running as Administrator
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD COMPLETE!
echo ========================================
echo.
echo To run the application, execute: run.bat
echo.
pause