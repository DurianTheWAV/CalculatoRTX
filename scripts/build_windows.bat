@echo off
rem CalculatoRTX - Windows build (Visual Studio 2022 + Vulkan SDK [+ CUDA Toolkit for NVIDIA]).
rem Run from "x64 Native Tools Command Prompt for VS 2022" or a regular terminal.
rem Without the CUDA Toolkit (AMD / Intel GPUs), only the Vulkan backend is built.
setlocal
cd /d "%~dp0\.."

where cmake >nul 2>nul
if errorlevel 1 (
    echo CMake not found: install CMake 3.24+ ^(or use the one shipped with Visual Studio^).
    exit /b 1
)
if "%VULKAN_SDK%"=="" (
    echo VULKAN_SDK is not set: install the LunarG Vulkan SDK ^(https://vulkan.lunarg.com/^),
    echo it provides the Vulkan headers, the loader library and glslangValidator.
    exit /b 1
)

set PRESET=windows-release
where nvcc >nul 2>nul
if errorlevel 1 (
    echo nvcc not found: building the Vulkan backend only ^(AMD / Intel / NVIDIA^).
    set PRESET=windows-vulkan
)

cmake --preset %PRESET% %*
if errorlevel 1 exit /b 1
cmake --build --preset %PRESET%
if errorlevel 1 exit /b 1
ctest --preset %PRESET%

echo.
echo Build finished: build\%PRESET%\bin\Release\CalculatoRTX.exe
endlocal
