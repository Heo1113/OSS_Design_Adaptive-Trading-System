@echo off
cd /d "%~dp0"
echo === ATS build ===

where javac >nul 2>nul || goto :nojdk
javac -version

rem Oracle javapath exposes javac but not jar; locate real JDK home via java.home
set "JH="
for /f "tokens=2 delims==" %%a in ('java -XshowSettings:properties -version 2^>^&1 ^| findstr /c:"java.home"') do set "JH=%%a"
for /f "tokens=* delims= " %%a in ("%JH%") do set "JH=%%a"

set "JARCMD=jar"
if exist "%JH%\bin\jar.exe" set "JARCMD=%JH%\bin\jar.exe"

if exist out rmdir /s /q out
if exist ATS.jar del ATS.jar
mkdir out

echo Compiling...
javac --release 17 -encoding UTF-8 -d out -sourcepath src src\ats\Main.java || goto :fail

echo Packaging with: %JARCMD%
"%JARCMD%" --create --file ATS.jar --main-class ats.Main -C out . || goto :fail

echo.
echo Build OK - ATS.jar   (run: java -jar ATS.jar)
pause
exit /b 0

:nojdk
echo [ERROR] javac not found. Install JDK 17+ and add it to PATH.
echo         Download: https://adoptium.net/
pause
exit /b 1

:fail
echo [ERROR] Build failed. See messages above.
pause
exit /b 1
