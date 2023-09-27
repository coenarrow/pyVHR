@echo off
cd C:\Users\20759193\source\repos\pyVHR\scripts

REM Activate the Conda environment
call C:\Users\20759193\AppData\Local\anaconda3\Scripts\activate.bat pyvhr

REM Run the Python script with the specified parameters
python crop_color_frames.py --input K1.mkv --output K1_Cropped_Color.mkv --rect_size 500 500

REM Run the Python script with the specified parameters
python crop_color_frames.py --input K2.mkv --output K2_Cropped_Color.mkv --rect_size 500 500

REM Optionally, deactivate the Conda environment
call conda deactivate

echo Script execution complete
pause
