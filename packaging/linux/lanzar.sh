#!/usr/bin/env bash
# Opens «Selector de Modelos IA» in its own Chrome window (--app, own profile, maximized).
# Based on ~/proyectos/herramientas/iconos/plantillas/lanzar-app-web.sh. Before starting the server it updates the
# app from GitHub when there is a new version (actualizar.sh), and it stops the server when the window
# closes, waiting for a catalog update in progress to finish.
set -u
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

APP="selector"                                   # also the host name: selector.localhost
TITULO="Selector de Modelos IA"
CLASE="chrome-selector.localhost__-Default"      # StartupWMClass in the .desktop (Wayland window name)
BASE_PUERTO="${APP_PORT:-8790}"
DATOS="${XDG_DATA_HOME:-$HOME/.local/share}/selector-modelos-ia"
LOG="$DATOS/servidor.log"
PIDFILE="$DATOS/servidor.pid"
PERFIL="$DATOS/ventana-chrome"
CERROJO="$DATOS/lanzador.lock"
SERVIDOR="$APP_DIR/venv/bin/selector-ia-server"
mkdir -p "$DATOS"

avisar() {
  notify-send -a "$TITULO" "$TITULO" "$1" 2>/dev/null || true
}

error() {
  avisar "$1"
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="$TITULO" --text="$1" 2>/dev/null || true
  fi
}

# Port of THIS app's server (other apps may use nearby ports).
puerto_activo() {
  for p in $(seq "$BASE_PUERTO" $((BASE_PUERTO + 14))); do
    if curl -s -m 1 -H "Host: 127.0.0.1:$p" "http://127.0.0.1:$p/api/ocupado" 2>/dev/null | grep -q "\"app\": \"$APP\""; then
      echo "$p"; return 0
    fi
  done
  return 1
}

motivo_ocupado() {
  curl -s -m 2 -H "Host: 127.0.0.1:$1" "http://127.0.0.1:$1/api/ocupado" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("motivo", ""))' 2>/dev/null
}

CHROME="$(command -v google-chrome || command -v google-chrome-stable || command -v chromium || command -v chromium-browser || true)"

# On Wayland, Chrome names an --app window "chrome-<host>__-Default": with its own host
# (selector.localhost, which Chrome resolves to 127.0.0.1) the dock groups it with its icon.
abrir_ventana() {
  local url="http://$APP.localhost:$1"
  if [ -z "$CHROME" ]; then
    xdg-open "http://127.0.0.1:$1" >/dev/null 2>&1 &
    return
  fi
  mkdir -p "$PERFIL"
  setsid "$CHROME" --user-data-dir="$PERFIL" --app="$url" --class="$CLASE" --start-maximized \
    --no-first-run --no-default-browser-check >/dev/null 2>&1 &
}

ventana_abierta() {
  pgrep -f -- "--user-data-dir=$PERFIL( |$)" >/dev/null 2>&1
}

# 1) server: reuse it, or update the app and start it
if ! P=$(puerto_activo); then
  [ -x "$APP_DIR/actualizar.sh" ] && "$APP_DIR/actualizar.sh"
  if [ ! -x "$SERVIDOR" ]; then
    error "La aplicación no está bien instalada. Vuelve a ejecutar el instalador (packaging/linux/install.sh)."
    exit 1
  fi
  : > "$LOG"
  APP_PORT="$BASE_PUERTO" setsid nohup "$SERVIDOR" >>"$LOG" 2>&1 &
  echo $! > "$PIDFILE"

  P=""
  for _ in $(seq 1 100); do      # up to 20 s
    if P=$(puerto_activo); then break; fi
    sleep 0.2
  done
  if [ -z "$P" ]; then
    rm -f "$PIDFILE"
    detalle=$(tail -n 5 "$LOG" 2>/dev/null | tr '\n' ' ')
    error "No se pudo arrancar $TITULO. Revisa: $LOG ${detalle:+( $detalle )}"
    exit 1
  fi
fi

# 2) window
abrir_ventana "$P"
[ -n "$CHROME" ] || exit 0

# 3) watcher: only one at a time (if another one runs, it will stop the server)
exec 9>"$CERROJO"
flock -n 9 || exit 0

sleep 3                          # give Chrome time to start its process
avisado=""
while true; do
  if ventana_abierta; then
    avisado=""
    sleep 2
    continue
  fi
  P=$(puerto_activo) || exit 0   # already stopped (parar.sh)
  motivo=$(motivo_ocupado "$P")
  if [ -n "$motivo" ]; then
    if [ -z "$avisado" ]; then
      avisar "Sigue trabajando en segundo plano ($motivo). Se cerrará solo al terminar."
      avisado=1
    fi
    sleep 10
    continue
  fi
  "$APP_DIR/parar.sh" --silencio
  exit 0
done
