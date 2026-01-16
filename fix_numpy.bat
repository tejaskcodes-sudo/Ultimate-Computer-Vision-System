@echo off
echo =========================================
echo FIXING NUMPY VERSION CONFLICT
echo =========================================

echo Current NumPy version:
python -c "import numpy; print(numpy.__version__)"

echo.
echo Downgrading NumPy to 1.24.3...
pip uninstall numpy -y
pip install numpy==1.24.3

echo.
echo Fixing OpenCV...
pip uninstall opencv-contrib-python opencv-python -y
pip install opencv-python==4.9.0.80

echo.
echo Verifying installation...
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mediapipe as mp; print('MediaPipe:', mp.__version__)"

echo.
echo =========================================
echo FIX COMPLETE!
echo Press any key to run the program...
pause > nul

python "app.py"