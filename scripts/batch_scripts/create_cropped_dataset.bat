@echo off
SETLOCAL EnableDelayedExpansion

REM Activate the conda environment
call C:\Users\20759193\AppData\Local\anaconda3\Scripts\activate.bat pyvhr

REM Set the directories for the input dataset, output dataset, and Python script
SET INPUT_DATASET=Z:\Dataset
SET OUTPUT_DATASET=C:\Users\20759193\source\repos\pyVHR\data
SET PYTHON_SCRIPT=C:\Users\20759193\source\repos\pyVHR\scripts\crop_color_frames.py

REM Execute the Python script with the provided dataset directories
python "%PYTHON_SCRIPT%" --raw_dataset "%INPUT_DATASET%" --processed_dataset "%OUTPUT_DATASET%"

REM End the script
ENDLOCAL
