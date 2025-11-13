@echo off
chcp 65001 >nul
echo ============================================================
echo    PROIECT: DETECȚIE PISICI VS CÂINI - YOLOv8
echo    GPU: NVIDIA RTX 3060
echo ============================================================
echo.

REM Verifică dacă mediul virtual există
if not exist ".venv1\" (
    echo ❌ Mediul virtual nu există!
    echo.
    echo 📦 Configurare automată...
    echo.
    python setup_environment.py
    echo.
    if errorlevel 1 (
        echo ❌ Eroare la configurare!
        pause
        exit /b 1
    )
)

echo ✅ Activare mediu virtual...
call .venv1\Scripts\activate.bat

echo.
echo ============================================================
echo    Mediul virtual este activat!
echo ============================================================
echo.

REM Test rapid GPU
echo 🔍 Verificare GPU...
python test_gpu.py

echo.
echo ============================================================
echo    PORNIRE APLICAȚIE
echo ============================================================
echo.

python main.py

echo.
echo ============================================================
echo    Aplicația s-a închis
echo ============================================================
pause
