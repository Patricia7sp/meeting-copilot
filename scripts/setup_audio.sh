#!/bin/bash
# Setup guiado de áudio: Linux (trabalho) e macOS (pessoal). Idempotente.
set -e
cd "$(dirname "$0")/.."
OS=$(uname -s)
echo "[setup-audio] OS detectado: $OS"

python3 apps/audio/check_audio.py || true

if [ "$OS" = "Linux" ]; then
  echo ""
  echo "--- Linux: nada para instalar na maioria dos casos ---"
  echo "1. Durante uma call, rode: pactl list short sources | grep monitor"
  echo "2. Anote o nome do .monitor e exporte: export LOOPBACK_DEVICE='<nome>'"
  echo "3. Deps Python: pip install -r apps/audio/requirements.txt -r apps/stt/requirements.txt"
elif [ "$OS" = "Darwin" ]; then
  echo ""
  echo "--- macOS: requer BlackHole 2ch (1x) ---"
  if ! system_profiler SPAudioDataType 2>/dev/null | grep -q BlackHole; then
    echo "BlackHole não detectado."
    if command -v brew >/dev/null; then
      read -p "Instalar agora via brew? [S/n] " ans
      if [ "$ans" != "n" ]; then brew install --cask blackhole-2ch; fi
    else
      echo "Instale manualmente: https://existential.audio/blackhole/ (versão 2ch)"
    fi
    echo "Depois: Audio MIDI Setup -> + -> Create Multi-Output Device -> marque BlackHole + seus alto-falantes."
    echo "No Meet/Teams, mantenha saída nos alto-falantes normais; o app captura do BlackHole."
  else
    echo "BlackHole OK."
  fi
  echo "Deps Python: pip install -r apps/audio/requirements.txt -r apps/stt/requirements.txt"
else
  echo "SO ainda não coberto na Entrega 2."
fi
