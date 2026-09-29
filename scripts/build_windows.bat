@echo off
rem CalculatoRTX - compilation Windows (Visual Studio 2022 + CUDA Toolkit + Vulkan SDK).
rem A lancer depuis "x64 Native Tools Command Prompt for VS 2022" ou un terminal normal.
setlocal
cd /d "%~dp0\.."

where cmake >nul 2>nul
if errorlevel 1 (
    echo CMake introuvable : installez CMake 3.24+ ^(ou utilisez celui de Visual Studio^).
    exit /b 1
)
where nvcc >nul 2>nul
if errorlevel 1 (
    echo nvcc introuvable : installez le CUDA Toolkit 12.4+ ^(13.x recommande^).
    exit /b 1
)
if "%VULKAN_SDK%"=="" (
    echo Attention : VULKAN_SDK non defini. Installez le SDK Vulkan LunarG : https://vulkan.lunarg.com/
)

cmake --preset windows-release %*
if errorlevel 1 exit /b 1
cmake --build --preset windows-release
if errorlevel 1 exit /b 1
ctest --preset windows-release

echo.
echo Compilation terminee : build\windows\bin\Release\CalculatoRTX.exe
endlocal
