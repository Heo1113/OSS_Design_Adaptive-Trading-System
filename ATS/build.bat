@echo off
rem ATS 빌드 스크립트 - 요구사항: JDK 17 이상 (그 외 외부 의존성 없음)
cd /d %~dp0
if exist out rmdir /s /q out
if exist ATS.jar del ATS.jar
mkdir out
dir /s /b src\*.java > .sources.txt
javac --release 17 -encoding UTF-8 -d out @.sources.txt
jar --create --file ATS.jar --main-class ats.Main -C out .
del .sources.txt
echo Build OK - ATS.jar   (run: java -jar ATS.jar)
