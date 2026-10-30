# Selector de Modelos IA

Un buscador de **modelos de IA actuales** para elegir bien sin pasar horas mirando cientos de modelos.
Marcas lo que necesitas (leer PDF, usar herramientas, programar, generar imágenes, transcribir audio…),
dónde lo vas a usar (la API de OpenRouter con tu programa favorito, tu ordenador, un VPS o un servidor
con GPU) y te enseña **solo los modelos que cumplen todo**, ordenados como elijas, con sus pros y
contras y cómo ponerlos en marcha.

**Regla principal: cero invención.** Todo dato (modelos, precios, licencias, benchmarks) sale de las
fuentes oficiales y lleva su fecha. Si falta un dato, verás «sin datos». Lo único que se estima es la
memoria que necesita un modelo en tu equipo, y siempre se marca como «estimación».

> Para **Linux (Ubuntu)**. Se instala con un comando, sin contraseña de administrador, y **se actualiza
> sola** cada vez que la abres.

## Instalación (Ubuntu)

1. Abre una terminal (tecla Super, escribe «Terminal» y pulsa Intro).
2. Copia y pega esto, y pulsa Intro:
   ```bash
   git clone https://github.com/rankgnar/selector-modelos-ia ~/.local/opt/selector-modelos-ia/src && ~/.local/opt/selector-modelos-ia/src/packaging/linux/install.sh
   ```
3. Ábrela desde **«Mostrar aplicaciones»**: «Selector de Modelos IA». Se abre en su propia ventana.

Necesita `git`, `curl`, Python 3.11 o más reciente y Google Chrome (o Chromium). Si falta algo, el
instalador te dice el comando exacto para instalarlo.

**Actualizar:** no hace falta hacer nada. Cada vez que la abres comprueba si hay una versión nueva en
GitHub y se actualiza sola (si algo fallara, sigue con la versión anterior).

**Desinstalar:** `~/.local/opt/selector-modelos-ia/src/packaging/linux/uninstall.sh` (añade `--purge` para
borrar también tus equipos guardados, los ajustes y las claves).

## Cómo se usa

**A la izquierda, los filtros** (todo se puede combinar; al lado de cada opción ves cuántos modelos
quedarían):

| Grupo | Qué eliges |
|---|---|
| **Dónde lo vas a usar** | **OpenRouter** (por internet, pagando por uso) · **Mis equipos** (tu ordenador, un VPS o un servidor con GPU: solo salen los modelos que caben) · **Otro proveedor** (fal.ai, Replicate, Together…) · **Cualquiera** |
| **¿Con qué programa?** (con OpenRouter) | Claude Code, Codex CLI, OpenCode, Pi, OMP, Cline, Roo Code, Kilo Code, Aider, Continue u otro compatible con OpenAI. Se aplican los requisitos que indica su documentación oficial y el detalle de cada modelo te da la configuración lista para copiar |
| **Qué tiene que hacer** | Escribir, programar, usar herramientas, razonar, leer PDF, ver imágenes, escuchar audio, ver vídeo, generar imágenes, editar imágenes, generar voz, música o vídeo, buscar en documentos |
| **Requisitos** | Habla español, sin censura, gratis, uso comercial, se puede descargar, textos muy largos |
| **Más opciones** | Incluir modelos antiguos · Incluir versiones de la comunidad (ajustes y versiones de un modelo hechas por otras personas) |
| **Precio máximo** | Con OpenRouter |

**Arriba a la derecha, el orden:** mejor calidad (con el benchmark de lo que has marcado: programación,
agentes, error de transcripción…), mejor calidad-precio, más barato, más usado, más nuevo o más ligero.

**Los 3 primeros** salen destacados con el porqué; debajo está la lista completa. Al pulsar un modelo ves:
a favor y en contra, dónde usarlo, si cabe en tu equipo (con el comando de Ollama, llama.cpp, LM Studio o
vLLM) y cómo configurarlo en tu programa. **Probar** envía el mismo mensaje a hasta 3 modelos y compara
las respuestas (con tu clave de OpenRouter; suele costar menos de 0,01 USD).

**Buscar por nombre** (arriba) busca también en todo Hugging Face, no solo en el catálogo: escribe
«abliterated», «dolphin», «qwen»… y verás también lo que encuentre allí.

Si ningún modelo cumple todo, te dice qué quitar con un botón. Los filtros quedan en la dirección de la
página: puedes compartir un enlace con los filtros ya marcados.

**Mis equipos:** detecta tu ordenador o añade otros a mano o desde una plantilla (VPS sin GPU, PC con GPU
NVIDIA, servidor GPU alquilado). Se guardan para la próxima vez.

**Ajustes:** tu clave de OpenRouter (solo para «Probar»; se guarda en el llavero del sistema, nunca en un
archivo), el token de Hugging Face (opcional) y cada cuánto se actualiza el catálogo.

## Fuentes de datos

| Fuente | Qué aporta |
|---|---|
| [OpenRouter](https://openrouter.ai/models) | Modelos por API: precio, contexto, entradas y salidas, herramientas, razonamiento. Incluye los índices de [Artificial Analysis](https://artificialanalysis.ai) y el ELO de [Design Arena](https://www.designarena.ai). |
| [Hugging Face](https://huggingface.co/models) | Modelos abiertos de todo tipo: licencia, idiomas, descargas, tamaño, proveedores que los sirven y resultados publicados en su ficha (por ejemplo, el [Open ASR Leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)). |
| Documentación oficial de cada programa | Cómo conectar Claude Code, Codex CLI, OpenCode, Pi, OMP, Cline, Roo Code, Kilo Code, Aider y Continue a OpenRouter (enlace y fecha en `src/selector_ia/web/harnesses.json`). |

El catálogo se actualiza solo al abrir la app si tiene más de 6 horas, y cada 6 horas mientras está
abierta (se cambia en Ajustes). Arriba ves «Datos hace…»; púlsalo para actualizar ahora.

## Dónde se guardan los datos

`~/.local/share/selector-modelos-ia/` (catálogo descargado, tus equipos y ajustes). Borrar el archivo
`cache.sqlite3` es seguro: se vuelve a descargar todo.

## Problemas frecuentes

- **«Tu configuración de privacidad de OpenRouter bloquea este modelo»** al probar: algunos modelos solo
  están en proveedores que pueden usar tus datos para entrenar. Cámbialo en
  <https://openrouter.ai/settings/privacy> si te parece bien, o elige otro modelo.
- **«No tienes saldo suficiente»**: recarga créditos en OpenRouter.
- **«Has alcanzado el límite de peticiones de Hugging Face»**: espera unos minutos o añade un token de
  Hugging Face en Ajustes.
- **No se abre**: mira `~/.local/share/selector-modelos-ia/servidor.log`. Para pararla a mano: clic derecho
  en el icono → «Parar ahora».
