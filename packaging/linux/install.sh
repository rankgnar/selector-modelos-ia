#!/usr/bin/env bash
# Installs «Selector de Modelos IA» for the current user (no sudo). Following ~/proyectos/iconos/README.md:
#   app source (git clone, used for automatic updates) -> ~/.local/opt/selector-modelos-ia/src
#   its own Python environment                         -> ~/.local/opt/selector-modelos-ia/venv
#   launcher (Chrome --app window) and stop script     -> ~/.local/opt/selector-modelos-ia/{lanzar,parar}.sh
#   menu entry + icon                                  -> ~/.local/share/applications, ~/.local/share/icons
# Students: git clone https://github.com/rankgnar/selector-modelos-ia ~/.local/opt/selector-modelos-ia/src
#           && ~/.local/opt/selector-modelos-ia/src/packaging/linux/install.sh
# From another copy of the repo (the teacher's project folder) it installs that copy's last commit.
# --update: used by actualizar.sh after pulling a new version (no clone, no questions).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM="https://github.com/rankgnar/selector-modelos-ia.git"
APP_DIR="$HOME/.local/opt/selector-modelos-ia"
SRC="$APP_DIR/src"
VENV="$APP_DIR/venv"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
ID="selector-modelos-ia"
UPDATE=0
[ "${1:-}" = "--update" ] && UPDATE=1

say() { [ "$UPDATE" = 1 ] || printf '\n==> %s\n' "$*"; }

# copy with an atomic replace: a running script keeps reading its old copy
put() { install -m "$3" "$1" "$2.tmp.$$" && mv -f "$2.tmp.$$" "$2"; }

say "Comprobando requisitos"
PY="$(command -v python3 || true)"
if [ -z "$PY" ] || ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  echo "Hace falta Python 3.11 o más reciente. En Ubuntu: sudo apt install python3 python3-venv"
  exit 1
fi
if ! "$PY" -c 'import ensurepip' 2>/dev/null; then
  echo "Falta el módulo venv de Python. Instálalo con: sudo apt install python3-venv"
  exit 1
fi
command -v git >/dev/null || { echo "Hace falta git. En Ubuntu: sudo apt install git"; exit 1; }
command -v curl >/dev/null || { echo "Hace falta curl. En Ubuntu: sudo apt install curl"; exit 1; }
CHROME="$(command -v google-chrome || command -v google-chrome-stable || command -v chromium || command -v chromium-browser || true)"

mkdir -p "$APP_DIR"
if [ "$UPDATE" = 0 ] && [ "$REPO" != "$SRC" ]; then
  say "Copiando la aplicación (último commit de $REPO)"
  if [ -d "$SRC/.git" ]; then
    git -C "$SRC" fetch --quiet "$REPO" HEAD
    git -C "$SRC" reset --quiet --hard FETCH_HEAD
  else
    rm -rf "$SRC"
    git clone --quiet "$REPO" "$SRC"
  fi
  git -C "$SRC" remote set-url origin "$UPSTREAM"     # updates come from GitHub
fi

say "Creando el entorno de Python en $VENV"
if compgen -G "$VENV/lib/python3*/site-packages/PySide6" >/dev/null; then
  rm -rf "$VENV"                                          # environment of the old Qt version: start clean
fi
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --upgrade "$SRC"

# installed version, shown in Ajustes
PKG="$("$VENV/bin/python" -c 'import os, selector_ia; print(os.path.dirname(selector_ia.__file__))')"
git -C "$SRC" log -1 --format='{"commit": "%h", "date": "%cI", "source": "github"}' > "$PKG/VERSION.json"

say "Instalando el lanzador"
put "$SRC/packaging/linux/lanzar.sh" "$APP_DIR/lanzar.sh" 755
put "$SRC/packaging/linux/parar.sh" "$APP_DIR/parar.sh" 755
put "$SRC/packaging/linux/actualizar.sh" "$APP_DIR/actualizar.sh" 755
mkdir -p "$BIN_DIR"
rm -f "$BIN_DIR/selector-ia"                              # old terminal command (removed)
ln -sf "$APP_DIR/lanzar.sh" "$BIN_DIR/$ID"

say "Añadiendo la aplicación a «Mostrar aplicaciones»"
mkdir -p "$DESKTOP_DIR" "$ICON_DIR"
put "$SRC/packaging/linux/$ID.svg" "$ICON_DIR/$ID.svg" 644
cat > "$DESKTOP_DIR/$ID.desktop.tmp.$$" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Selector de Modelos IA
Comment=Busca el modelo de IA adecuado para lo que necesitas y dónde lo vas a usar
Exec=$APP_DIR/lanzar.sh
Icon=$ICON_DIR/$ID.svg
Terminal=false
StartupNotify=true
StartupWMClass=chrome-selector.localhost__-Default
Categories=Utility;
Actions=parar;

[Desktop Action parar]
Name=Parar ahora
Exec=$APP_DIR/parar.sh
DESKTOP
mv -f "$DESKTOP_DIR/$ID.desktop.tmp.$$" "$DESKTOP_DIR/$ID.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
if command -v desktop-file-validate >/dev/null && [ "$UPDATE" = 0 ]; then
  desktop-file-validate "$DESKTOP_DIR/$ID.desktop" || true
fi

say "Comprobando que la aplicación arranca"
"$VENV/bin/selector-ia-server" --smoke-test

[ "$UPDATE" = 1 ] && exit 0
if [ -z "$CHROME" ]; then
  echo
  echo "AVISO: no se ha encontrado Google Chrome ni Chromium. La app se abrirá en tu navegador normal."
  echo "Para tenerla en su propia ventana instala Chrome (google.com/chrome) o: sudo snap install chromium"
fi
say "Instalado ($(git -C "$SRC" log -1 --format=%h)). Ábrelo desde «Mostrar aplicaciones»: «Selector de Modelos IA»."
echo "Se actualiza sola desde GitHub cada vez que la abres (si hay conexión)."
