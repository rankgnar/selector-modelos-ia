#!/usr/bin/env bash
# Stops the «Selector de Modelos IA» server.  --silencio: no notification (used by lanzar.sh).
set -u
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
DATOS="${XDG_DATA_HOME:-$HOME/.local/share}/selector-modelos-ia"
PIDFILE="$DATOS/servidor.pid"
SERVIDOR="$APP_DIR/venv/bin/selector-ia-server"

parado=0

if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "${PID:-}" ] && [ -r "/proc/$PID/cmdline" ] && tr '\0' ' ' < "/proc/$PID/cmdline" | grep -q "selector-ia-server"; then
    kill "$PID" 2>/dev/null && parado=1
    sleep 0.5
  fi
  rm -f "$PIDFILE"
fi

# only the server of THIS installation (its full path is in the command line)
for p in /proc/[0-9]*; do
  [ -r "$p/cmdline" ] || continue
  c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
  case "$c" in
    *"$SERVIDOR"*) kill "${p#/proc/}" 2>/dev/null && parado=1;;
  esac
done

if [ "${1:-}" != "--silencio" ] && command -v notify-send >/dev/null 2>&1; then
  if [ "$parado" = "1" ]; then
    notify-send "Selector de Modelos IA" "Servidor detenido"
  else
    notify-send "Selector de Modelos IA" "No estaba en marcha"
  fi
fi
