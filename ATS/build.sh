#!/usr/bin/env bash
# ATS 빌드 스크립트 — 요구사항: JDK 17 이상 (그 외 외부 의존성 없음)
set -e
cd "$(dirname "$0")"
rm -rf out ATS.jar
mkdir -p out
find src -name "*.java" > .sources.txt
javac --release 17 -encoding UTF-8 -d out @.sources.txt
jar --create --file ATS.jar --main-class ats.Main -C out .
rm -f .sources.txt
echo "Build OK -> ATS.jar   (run: java -jar ATS.jar)"
