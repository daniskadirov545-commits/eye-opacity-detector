@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Не найден Python в .venv
    pause
    exit /b 1
)

if not exist "unet_corneal_opacity.pt" (
    echo Не найден unet_corneal_opacity.pt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install pyinstaller
".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --windowed ^
  --name EyeOpacityApp ^
  --add-data "unet_corneal_opacity.pt;." ^
  app_build_stable.py

pause