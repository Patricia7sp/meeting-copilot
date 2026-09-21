#!/bin/bash
# Build do overlay desktop (Linux + macOS). Roda na máquina alvo, não no container.
set -e
cd "$(dirname "$0")/desktop"
OS=$(uname -s)
echo "[desktop] OS=$OS"
if ! command -v npm >/dev/null; then echo "instale Node 18+ primeiro"; exit 1; fi
if ! command -v cargo >/dev/null; then echo "instale Rust (rustup) primeiro"; exit 1; fi
if [ "$OS" = "Linux" ]; then
  echo "[desktop] Linux precisa: libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev patchelf"
  echo "  Ubuntu/Debian: sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file libssl-dev libayatana-appindicator3-dev librsvg2-dev patchelf"
fi
if [ "$OS" = "Darwin" ]; then
  echo "[desktop] macOS precisa: Xcode Command Line Tools (xcode-select --install)"
fi
npm install
npx tauri build
echo "[desktop] bundle em src-tauri/target/release/bundle/"
