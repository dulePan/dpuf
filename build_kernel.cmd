@echo off
rem Compiles the native AVX2 + FMA + OpenMP Zero-Decode DLL on Windows using MSVC BuildTools
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo MSVC vcvars64.bat not found. Please run from a Visual Studio x64 Developer Prompt.
  exit /b 1
)
cl /O2 /arch:AVX2 /openmp dpuf-cli.c dpuf_avx2.c dpuf_engine.c dpuf_vision.c ws2_32.lib /Fe:dpuf-cli.exe /nologo
if errorlevel 1 exit /b 1
rem Optional DLL build:
rem cl /O2 /arch:AVX2 /openmp /LD dpuf_avx2.c dpuf_engine.c dpuf_vision.c /Fe:dpuf_avx2.dll /nologo
if errorlevel 1 exit /b 1
del /q dpuf_avx2.obj dpuf_engine.obj dpuf_vision.obj dpuf-cli.obj dpuf_avx2.lib dpuf_avx2.exp dpuf-cli.lib dpuf-cli.exp 2>nul
echo Built dpuf_avx2.dll and dpuf-cli.exe successfully!
