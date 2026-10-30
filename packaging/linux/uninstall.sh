#!/usr/bin/env bash
# Removes the app, its menu entry and its command. Your saved computers, settings and keys are kept
# unless you pass --purge.
set -euo pipefail

APP_DIR="$HOME/.local/opt/selector-modelos-ia"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"

[ -x "$APP_DIR/parar.sh" ] && "$APP_DIR/parar.sh" --silencio || true

if [ "${1:-}" = "--purge" ] && [ -x "$APP_DIR/venv/bin/python" ]; then
  # keys live in the system keyring; remove them with the app's own Python before deleting it
  "$APP_DIR/venv/bin/python" -c 'from selector_ia import credentials as c
for name in (c.OPENROUTER_KEY, c.HF_TOKEN):
    c.delete_secret(name)' || true
fi

rm -rf "$APP_DIR"
rm -f "$HOME/.local/bin/selector-modelos-ia" "$HOME/.local/bin/selector-ia"
rm -f "$DATA_DIR/applications/selector-modelos-ia.desktop"
rm -f "$DATA_DIR/icons/hicolor/scalable/apps/selector-modelos-ia.svg"
command -v update-desktop-database >/dev/null && update-desktop-database "$DATA_DIR/applications" >/dev/null 2>&1 || true
echo "Aplicación desinstalada."

if [ "${1:-}" = "--purge" ]; then
  rm -rf "$DATA_DIR/selector-modelos-ia"
  echo "También se han borrado tus equipos guardados, los ajustes, el catálogo descargado y las claves."
else
  echo "Tus equipos guardados, ajustes y claves se conservan. Usa --purge para borrarlos también."
fi
