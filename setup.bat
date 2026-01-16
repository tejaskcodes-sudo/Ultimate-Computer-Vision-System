@echo off
echo ========================================
echo ULTIMATE CV SYSTEM - SETUP SCRIPT
echo ========================================
echo.

echo Checking for required files...
if not exist "ultimate_cv_final.py" (
    echo ERROR: ultimate_cv_final.py not found!
    echo Please save your Python code with this name.
    pause
    exit /b 1
)

echo Creating Dockerfile...
(
echo # Use Python 3.10
echo FROM python:3.10-slim
echo.
echo # Set working directory
echo WORKDIR /app
echo.
echo # Install system dependencies
echo RUN apt-get update ^&^& apt-get install -y \
echo     libopencv-dev \
echo     libgl1-mesa-glx \
echo     libglib2.0-0 \
echo     libsm6 \
echo     libxext6 \
echo     libxrender-dev \
echo     wget \
echo     ^&^& rm -rf /var/lib/apt/lists/*
echo.
echo # Copy requirements file
echo COPY requirements.txt .
echo.
echo # Install Python packages
echo RUN pip install --no-cache-dir --upgrade pip ^&^& \
echo     pip install --no-cache-dir -r requirements.txt
echo.
echo # Copy application code
echo COPY . .
echo.
echo # Create non-root user
echo RUN useradd -m -u 1000 appuser
echo USER appuser
echo.
echo # Run the application
echo CMD ["python", "ultimate_cv_final.py"]
) > Dockerfile

echo Creating requirements.txt...
(
echo numpy==1.24.3
echo opencv-python==4.9.0.80
echo mediapipe==0.10.14
) > requirements.txt

echo Creating docker-compose.yml...
(
echo version: '3.8'
echo.
echo services:
echo   cv-system:
echo     build: .
echo     image: ultimate-cv-system:latest
echo     container_name: ultimate-cv-system
echo     volumes:
echo       - ./screenshots:/app/screenshots
echo     devices:
echo       - /dev/video0:/dev/video0
echo     privileged: true
echo     stdin_open: true
echo     tty: true
echo     restart: unless-stopped
) > docker-compose.yml

echo.
echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Files created:
echo   - Dockerfile
echo   - requirements.txt
echo   - docker-compose.yml
echo.
echo Next steps:
echo   1. Run build.bat to build Docker image
echo   2. Run run.bat to start the application
echo.
pause