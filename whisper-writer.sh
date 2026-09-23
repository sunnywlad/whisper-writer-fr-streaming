#!/bin/bash
cd "$(dirname "$0")"
if pgrep -f "[w]hisper-writer/venv/bin/python src/main.py" >/dev/null; then
  notify-send "WhisperWriter" "Déjà en cours d'exécution"
  exit 0
fi
# venv/bin/python directly rather than `source venv/bin/activate`: activate
# hard-codes the venv's absolute path and breaks once the folder moves.
export PYTHONUNBUFFERED=1 DOTOOL_XKB_LAYOUT=fr DOTOOL_XKB_VARIANT=oss
exec venv/bin/python run.py
