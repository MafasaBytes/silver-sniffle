@echo off
echo Starting GPU extraction...
echo Output will be logged to extraction_output.log
echo.
echo Check progress with: type extraction_output.log
echo Count files with: dir /b data\processed\train\*.npy | find /c ".npy"
echo.

python src\extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split all > extraction_output.log 2>&1

echo.
echo Extraction complete! Check extraction_output.log for details.
pause
