@echo off
:: Absolute path to the data directory
set "dataPath=C:\Users\20759193\source\repos\pyVHR\data"

:: This batch file iterates through all index folders inside the 'data' directory.

setlocal enabledelayedexpansion

for /d %%i in ("%dataPath%\*") do (
    set "index=%%~ni"
    set "success=1"
    
    if exist "%%i\K1_Cropped_Colour.mkv" (
        mkdir "%%i_K1"
        if exist "%%i\data.csv" (
            copy "%%i\data.csv" "%%i_K1\"
        ) else (
            set "success=0"
        )
        move "%%i\K1_Cropped_Colour.mkv" "%%i_K1\"
    )
    
    if exist "%%i\K2_Cropped_Colour.mkv" (
        mkdir "%%i_K2"
        if exist "%%i\data.csv" (
            copy "%%i\data.csv" "%%i_K2\"
        ) else (
            set "success=0"
        )
        move "%%i\K2_Cropped_Colour.mkv" "%%i_K2\"
    )
    
    if not exist "%%i_K1\data.csv" if exist "%%i\K1_Cropped_Colour.mkv" (
        set "success=0"
    )
    
    if not exist "%%i_K2\data.csv" if exist "%%i\K2_Cropped_Colour.mkv" (
        set "success=0"
    )
    
    if "!success!"=="1" (
        rd /s /q "%%i"
    ) else (
        echo Warning: An error occurred with index !index!, skipping deletion
    )
)

echo Done
pause
