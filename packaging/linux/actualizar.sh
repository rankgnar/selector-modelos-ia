#!/usr/bin/env bash
# Updates the installed app from GitHub when there is a new version on `main` (called by lanzar.sh
# before starting the server). Never leaves the app broken: if installing the new version fails, it
# goes back to the previous commit. Offline, without git or without access: it just does nothing.
set -u
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$APP_DIR/src"
TITULO="Selector de Modelos IA"

avisar() { notify-send -a "$TITULO" "$TITULO" "$1" 2>/dev/null || true; }

command -v git >/dev/null 2>&1 || exit 0
[ -d "$SRC/.git" ] || exit 0
cd "$SRC" || exit 0

export GIT_TERMINAL_PROMPT=0      # never ask for a password (private repo, no network…)
timeout 10 git fetch --quiet origin main 2>/dev/null || exit 0

ANTES=$(git rev-parse HEAD) || exit 0
NUEVA=$(git rev-parse origin/main) || exit 0
[ "$ANTES" = "$NUEVA" ] && exit 0
FALLIDA="$APP_DIR/version-fallida"                        # do not retry a broken version on every launch
[ "$(cat "$FALLIDA" 2>/dev/null)" = "$NUEVA" ] && exit 0
git merge-base --is-ancestor HEAD origin/main || exit 0    # only plain fast-forward updates

avisar "Actualizando a la versión nueva…"
if git merge --ff-only --quiet origin/main && "$SRC/packaging/linux/install.sh" --update >>"$APP_DIR/actualizacion.log" 2>&1; then
  avisar "Actualizado a la versión $(git log -1 --format=%h) ($(git log -1 --format=%cd --date=format:%d/%m/%Y))."
  exit 0
fi

# something failed: back to the version that worked (and remember not to retry this one)
echo "$NUEVA" > "$FALLIDA"
git reset --quiet --hard "$ANTES"
"$SRC/packaging/linux/install.sh" --update >>"$APP_DIR/actualizacion.log" 2>&1 || true
avisar "No se ha podido actualizar; sigues con la versión anterior. Detalles: $APP_DIR/actualizacion.log"
exit 0
