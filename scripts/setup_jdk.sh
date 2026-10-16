#!/usr/bin/env bash
# Download a Temurin 17 JDK into ./.jdk so Spark has a JVM without touching the
# rest of the machine. Nothing outside this directory is modified; remove .jdk to undo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JDK_DIR="$REPO_ROOT/.jdk"

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)  OS=mac;   ARCH=aarch64 ;;
  Darwin/x86_64) OS=mac;   ARCH=x64 ;;
  Linux/aarch64) OS=linux; ARCH=aarch64 ;;
  Linux/x86_64)  OS=linux; ARCH=x64 ;;
  *) echo "unsupported platform: $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac

if [ -x "$JDK_DIR/Contents/Home/bin/java" ] || [ -x "$JDK_DIR/bin/java" ]; then
  echo "JDK already present in $JDK_DIR"
  exit 0
fi

URL="https://api.adoptium.net/v3/binary/latest/17/ga/${OS}/${ARCH}/jdk/hotspot/normal/eclipse"
TARBALL="$(mktemp -t temurin17.XXXXXX).tar.gz"
trap 'rm -f "$TARBALL"' EXIT

echo "downloading Temurin 17 for ${OS}/${ARCH}"
curl -sS -L --fail --max-time 900 -o "$TARBALL" "$URL"
mkdir -p "$JDK_DIR"
tar -xzf "$TARBALL" -C "$JDK_DIR" --strip-components=1

if [ -x "$JDK_DIR/Contents/Home/bin/java" ]; then
  "$JDK_DIR/Contents/Home/bin/java" -version
else
  "$JDK_DIR/bin/java" -version
fi
