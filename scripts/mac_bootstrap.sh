#!/bin/bash
# Provisiona o cliente Mac do Meeting Copilot: repo + .venv + config do WS remoto.
# Idempotente — pode rodar de novo a qualquer momento.
#
#   bash scripts/mac_bootstrap.sh [DESTINO]
#
# DESTINO default: ~/meeting-copilot-client/client
set -euo pipefail

DEST="${1:-$HOME/meeting-copilot-client/client}"
REPO_URL="https://github.com/Patricia7sp/meeting-copilot.git"
WS_URL="ws://100.87.25.101:8000/ws"
MODELS="small,medium,large-v3-turbo,distil-large-v3"

say() { printf '\n\033[1m[mac] %s\033[0m\n' "$*"; }

say "1/5 checando ambiente"
[ "$(uname -s)" = "Darwin" ] || { echo "precisa ser macOS"; exit 1; }
command -v python3 >/dev/null || { echo "python3 ausente"; exit 1; }
PYVER=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
echo "  python3 = $PYVER"
case "$PYVER" in
  3.9|3.10|3.11|3.12|3.13) ;;
  *) echo "  AVISO: python $PYVER não validado (faster-whisper quer 3.9+)" ;;
esac

say "2/5 repositório em $DEST"
mkdir -p "$(dirname "$DEST")"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch --quiet origin
  git -C "$DEST" checkout --quiet main
  git -C "$DEST" pull --ff-only --quiet
  echo "  atualizado para $(git -C "$DEST" rev-parse --short HEAD)"
else
  [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ] && { echo "  $DEST existe e não é vazio — sair"; exit 1; }
  git clone --quiet "$REPO_URL" "$DEST"
  echo "  clonado"
fi

cd "$DEST"

say "3/5 .venv com faster-whisper, sounddevice, numpy"
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet -r apps/audio/requirements.txt -r apps/stt/requirements.txt
./.venv/bin/python - <<'PY'
import importlib.metadata as md
for p in ("faster-whisper", "sounddevice", "numpy"):
    try:
        print(f"  {p}=={md.version(p)}")
    except Exception:
        print(f"  {p}: NAO INSTALADO")
PY

say "4/5 config do cliente ($WS_URL)"
[ -f config/mac-client.env ] || { echo "  config/mac-client.env ausente no repo"; exit 1; }
# grava o destino do WS a partir da variável de ambiente, sem editar à mão
sed "s|^export API_WS_URL=.*|export API_WS_URL=\"$WS_URL\"|" config/mac-client.env > config/mac-client.local.env
grep '^export API_WS_URL' config/mac-client.local.env
echo "  ajuste MIC_DEVICE/LOOPBACK_DEVICE se check_audio.py mostrar outros nomes"

say "5/5 checando fontes de áudio"
./.venv/bin/python apps/audio/check_audio.py || true

say "pronto"
cat <<EOF

Ative o ambiente:
  cd "$DEST" && source .venv/bin/activate && source config/mac-client.local.env

1) DIAGNÓSTICO + captura de 60s (deixe a aula/video tocando no loopback durante a gravação):
   python3 apps/audio/diag_capture.py --seconds 60 --source both \\
       --ref transcripts/aula.txt --out data/bench/capture

2) BENCHMARK dos quatro modelos no WAV gerado:
   python3 apps/stt/benchmark.py --wav data/bench/capture/loopback_<ts>.wav \\
       --ref transcripts/aula.txt --meta data/bench/capture/meta_<ts>.json \\
       --models $MODELS --passes 3 --save apps/stt/BENCH_REPORT.md

3) INTERPRETAR:
   cat apps/stt/BENCH_REPORT.md

UI (no servidor, abra no navegador do Mac): http://100.87.25.101:8000/ui
EOF
