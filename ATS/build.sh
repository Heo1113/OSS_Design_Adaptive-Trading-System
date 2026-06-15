#!/usr/bin/env bash
# ATS 빌드 스크립트 — 요구사항: JDK 17 이상 (그 외 외부 의존성 없음)
set -e
cd "$(dirname "$0")"

if ! command -v javac >/dev/null 2>&1; then
  echo "[ERROR] javac not found. Install JDK 17+ (a JRE alone is not enough)."
  echo "        Verify with: javac -version"
  exit 1
fi
javac -version

rm -rf out ATS.jar
mkdir -p out

echo "Compiling..."
javac --release 17 -encoding UTF-8 -d out -sourcepath src src/ats/Main.java

echo "Packaging..."
jar --create --file ATS.jar --main-class ats.Main -C out .

echo "Build OK -> ATS.jar   (run: java -jar ATS.jar)"
