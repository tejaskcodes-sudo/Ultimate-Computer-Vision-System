@echo off
echo ========================================
echo ULTIMATE COMPUTER VISION SYSTEM
echo Running locally on Windows
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting application...
echo Make sure your webcam is connected!
echo.
echo Controls:
echo   Q - Quit
echo   R - Reset blink counter
echo   S - Save screenshot
echo   F - Toggle fullscreen
echo.
python app.py

echo.
pause