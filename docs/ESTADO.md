# Estado del proyecto — Selector de Modelos IA

> **Léelo al empezar una sesión nueva.** Resume qué se ha hecho, qué se ha decidido, dónde estamos y
> qué queda pendiente. Actualízalo al terminar cada cambio importante.
>
> Última actualización: **30/10/2026**

## Qué es

Un **buscador de modelos de IA actuales** para los **alumnos** del usuario (profesor), en español. El
alumno marca qué necesita que haga el modelo (varias cosas a la vez), dónde lo va a usar (API de
OpenRouter con su programa, su equipo, un VPS, un servidor con GPU, otro proveedor) y requisitos; salen
**solo los modelos que cumplen todo**, ordenados como elija, con pros y contras, dónde usarlos y cómo
configurarlos. Debe ser profesional, sencillo y fiable: de ello depende la credibilidad del usuario.

**Regla principal: CERO INVENCIÓN.** Todo dato sale de OpenRouter, Hugging Face o la documentación
oficial de cada programa, con su fecha. Lo que falta es «sin datos». Solo se estima la memoria en el
equipo, siempre marcada como «estimación» con su fórmula.

## Dónde está todo

| Qué | Dónde |
|---|---|
| Código | `~/proyectos/Private/selector-modelos-ia` |
| Repositorio | GitHub `rankgnar/selector-modelos-ia` (rama `main`), **público** desde el 30/10/2026 con el historial reiniciado en un solo commit (decisión del usuario). El historial anterior completo está solo en la rama local `historial-completo` (no se sube) |
| App instalada | `~/.local/opt/selector-modelos-ia/` = `src/` (clon de git) + `venv/` + `lanzar.sh`, `parar.sh`, `actualizar.sh` |
| Acceso | «Mostrar aplicaciones» → `~/.local/share/applications/selector-modelos-ia.desktop` (inventario en `~/proyectos/iconos/README.md`) |
| Datos del usuario | `~/.local/share/selector-modelos-ia/` (`cache.sqlite3` con catálogo, equipos y ajustes; `servidor.log`; perfil de Chrome de la ventana) |
| Claves | Llavero del sistema (servicio `selector-modelos-ia`), nunca en archivos |
| Documentación técnica | `CLAUDE.md` · para el usuario: `README.md` |

## Historia breve

1. Fases 1-6 (app Qt con un «cerebro» LLM que interpretaba texto libre): terminadas, pero al usuario no
   le convencía (lenta, de pago, respuestas distintas a la misma pregunta).
2. Primer rediseño (pantalla guiada con una sola tarea): rechazado («no es un filtro real», «solo puedes
   elegir una cosa», «no es profesional»).
3. **Rediseño actual (30/10/2026), aprobado con maqueta:** buscador web con filtros combinables al estilo
   openrouter.ai/models. Se borró todo lo anterior (Qt, cerebro, CLI, SSH; quedan en el historial de git:
   última versión Qt = commit `a63486b`).

## Decisiones tomadas (no volver a discutir)

- **Solo Linux (Ubuntu 24.04).** Nada de Mac, Windows ni versión web pública por ahora.
- **App web local en su ventana** (Chrome `--app`, host `selector.localhost`), como las apps Cursos y
  Subtítulos del usuario. Servidor solo en `127.0.0.1`.
- **Sin LLM para elegir:** filtros y orden deterministas (mismas casillas → mismo resultado, gratis).
- **Filtros AND:** el modelo debe cumplir todo lo marcado. Al lado de cada opción, cuántos quedarían.
- **Si nada cumple**, se sugiere cambiar de sitio, requisito o precio antes que quitar una capacidad (lo
  que el alumno necesita que haga el modelo no se le quita primero).
- **Orden «mejor calidad»** = benchmarks de lo marcado: Programar → coding_index; Herramientas →
  agentic_index; Escuchar audio → WER del Open ASR Leaderboard; Generar/editar imágenes → ELO de Design
  Arena; resto → intelligence_index (Artificial Analysis vía OpenRouter). Sin dato → detrás, por uso.
- **«Antiguo»** = publicado y sin actualizar desde hace más de 2 años (oculto por defecto). Se descartó
  «superado por una versión más nueva»: marcaba como antiguos modelos pequeños actuales que son justo los
  que caben en el PC de un alumno.
- **Originales por defecto; versiones de la comunidad bajo petición.** Fuera siempre: LoRA/adaptadores,
  copias cuantizadas normales, reempaquetados, CoreML/MLX (solo Mac) y diarización (pyannote). Los ajustes,
  mezclas y versiones sin censura hechos por terceros se guardan marcados como `community` (con su modelo
  base) y solo se ven con «Incluir versiones de la comunidad» o «Sin censura».
- **«Sin censura» (requisito):** el autor o proveedor lo declara (nombre, etiquetas o descripción con
  uncensored/abliterated/heretic…) y el proveedor de OpenRouter no modera. Aviso de calidad variable y
  licencia del original; los marcados por su autor como adultos llevan la etiqueta «+18». (Se quitó un
  tiempo por un malentendido y el usuario lo pidió de vuelta el 30/10/2026.)
- **OpenRouter completo:** se piden todas las modalidades (`output_modalities=all`): vídeo, embeddings, voz,
  transcripción, rerank e imagen (antes faltaban 236 modelos; lo detectó el usuario al ver casillas a 0).
  Casilla nueva «Reordenar resultados (RAG)». Precio de vídeo solo cuando OpenRouter lo da explícito en
  céntimos por segundo; si no, «sin datos» y los SKU tal cual en el detalle. Sin benchmark ni uso, el orden
  «mejor calidad» va del más nuevo al más antiguo y la tarjeta dice «Destacado».
- **Método para elegir modelo local (lo más importante para el usuario, 30/10/2026).** Bloque «Cómo lo vas a
  usar» en Mis equipos: chat/agente, contexto (8K-262K), agentes a la vez (1-8), caché KV q8_0. La caché KV
  cuenta solo las capas que crecen con el contexto (`layer_types`, `full_attention_interval`, ventana
  deslizante) × agentes; q8_0 la divide entre 2. Agente → solo Q5_K_M o mejor, o IQ4_XS (regla del método,
  mostrada como «regla general, no universal»); si no cabe ninguna, se dice cuál cabe por debajo. Orden por
  índice de agentes en uso agente. Benchmarks de ficha por categoría marcados «del autor» (ninguno viene
  verificado). Sin censura: citas literales de la ficha (negativas, KL, pruebas de tool calling,
  aggressive/balanced). Velocidad: no se estima; MTP + `llama-bench`. Constantes IQ4_XS (0,559) e IQ4_NL
  (0,585) medidas en unsloth/Qwen3-8B-GGUF. Flags de llama.cpp/Ollama comprobados en su documentación.
- **Buscar por nombre busca también en todo Hugging Face** (`/api/hfsearch`, 40 resultados): lo encontrado
  se ve aunque sea antiguo o de la comunidad, con sus etiquetas y «de Hugging Face».
- **Arneses (programas que usan la API de OpenRouter):** solo agentes de código (decisión del usuario):
  Claude Code, Codex CLI, OpenCode, Pi, OMP, Cline, Roo Code, Kilo Code, Aider, Continue y «otro
  compatible con OpenAI». Datos en `src/selector_ia/web/harnesses.json` con URL de la documentación
  oficial y fecha. Solo se aplican requisitos que la documentación dice (Claude Code: modelos de
  Anthropic, con opción de ver el resto «sin garantía»).
- **Mis equipos:** detectar este PC, a mano o desde plantilla (VPS 4/8/16/32 GB; PC con GPU NVIDIA
  8/12/16/24 GB; servidor GPU 48/80 GB). Se guardan en la caché.
- **Siempre actualizado:** catálogo cada 6 h (al abrir y mientras está abierta; reintento cada 30 min sin
  internet). La app se actualiza sola desde GitHub en cada arranque, con vuelta atrás si falla.
- **«Probar»** se mantiene: cada alumno pone su propia clave de OpenRouter (solo para eso).
- **Interfaz:** tema claro y oscuro (el usuario usa oscuro), estilo profesional; los filtros quedan en la
  dirección (`#caps=pdf,tools&app=cline…`) para compartir enlaces.
- La ventana se abre maximizada; no se toca la barra de favoritos del usuario.

## Cómo trabajar en una sesión nueva

```bash
cd ~/proyectos/Private/selector-modelos-ia
.venv/bin/pytest && node --test tests/js/*.test.mjs     # todo debe pasar
.venv/bin/pytest -m live                                # contra las APIs reales (gratis)
SELECTOR_IA_DATA_DIR=/tmp/x .venv/bin/selector-ia-server --port 8799   # probar sin tocar los datos del usuario
git commit … && ./packaging/linux/install.sh            # el instalador usa el ÚLTIMO COMMIT
```
Después de cada cambio: tests → commit → `install.sh` → push → actualizar este archivo.
Capturas sin ventana: `google-chrome --headless=new --screenshot=… "http://selector.localhost:PUERTO/?open=ID#caps=…"`
(`?open=pc` abre «Mis equipos», `?open=settings` los Ajustes, `?open=<id>` el detalle de un modelo).

## Avisos conocidos

- El puerto 8790 también lo usa la app Subtítulos del usuario: el lanzador encuentra el servidor por
  `/api/ocupado`, nunca solo por el puerto.
- La configuración de privacidad de la cuenta de OpenRouter puede bloquear algunos modelos en «Probar»
  (error 404 «data policy»): la app lo explica.
- El tamaño de muchos modelos de imagen y vídeo no se publica: salen como «sin datos de tamaño» y no
  cuentan como «caben».
- Los equipos guardados por la versión Qt (por ejemplo «Mi PC») siguen funcionando.

## Pendiente

- [x] Repositorio público (30/10/2026). Auditoría: ninguna clave en el historial; datos personales menores
      solo en versiones antiguas de este archivo → se publicó con el historial limpio. **No subir nunca** la
      rama `historial-completo` ni volver a poner aquí saldos, IPs o datos del PC del usuario.
- [ ] Pedir al usuario que confirme que la ventana se agrupa con su icono en el dock (en Wayland no se
      puede comprobar desde la terminal).
- [ ] Posibles mejoras: más arneses si los alumnos usan otros (Cursor, Hermes, Junie, OpenClaw tienen
      guía oficial en OpenRouter); benchmark fiable para voz, música y vídeo (hoy se ordenan por uso).
