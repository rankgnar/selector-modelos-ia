"""Which tool to run a model with, and an example command built only from real repo IDs."""

from pydantic import BaseModel

from selector_ia.data.models import HFModel
from selector_ia.hardware.profile import HardwareProfile
from selector_ia.hardware.variants import Variant

CHAT_TASKS = {"text-generation", "image-text-to-text", "text2text-generation", None}


class RunnerSuggestion(BaseModel):
    tool: str
    command: str
    note: str
    url: str


def suggest_runners(model: HFModel, variant: Variant | None, profile: HardwareProfile,
                    context_tokens: int = 8192) -> list[RunnerSuggestion]:
    os_name = (profile.os or "").lower()
    is_windows_or_mac = "windows" in os_name or "macos" in os_name
    has_nvidia = "nvidia" in profile.gpu_vendors
    has_gpu = bool(profile.dedicated_vram_bytes) or profile.has_unified_memory
    task = model.pipeline_tag
    tips: list[RunnerSuggestion] = []

    if variant and variant.format == "GGUF" and variant.repo_id and task in CHAT_TASKS:
        ref = f"{variant.repo_id}:{variant.label}"
        tips.append(RunnerSuggestion(
            tool="Ollama", command=f"ollama run hf.co/{ref}",
            note="La opción más sencilla: descarga y arranca el modelo con una sola orden.",
            url="https://ollama.com/download"))
        gpu_flag = " -ngl 99" if has_gpu else ""
        tips.append(RunnerSuggestion(
            tool="llama.cpp", command=f"llama-server -hf {ref} -c {context_tokens}{gpu_flag}",
            note="Más control y rendimiento; abre un chat web en http://localhost:8080.",
            url="https://github.com/ggml-org/llama.cpp/releases"))
        if is_windows_or_mac or profile.kind == "local":
            tips.append(RunnerSuggestion(
                tool="LM Studio", command=f"Buscar «{variant.repo_id}» en la pestaña Descubrir y elegir {variant.label}",
                note="Programa con ventanas, sin terminal.", url="https://lmstudio.ai"))
        return tips

    if task in CHAT_TASKS and has_nvidia and "windows" not in os_name and variant and variant.repo_id:
        tips.append(RunnerSuggestion(
            tool="vLLM", command=f"vllm serve {variant.repo_id} --max-model-len {context_tokens}",
            note="Servidor rápido para GPU NVIDIA en Linux, compatible con la API de OpenAI.",
            url="https://docs.vllm.ai"))

    repo = variant.repo_id if variant and variant.repo_id else model.id
    tags = {t.lower() for t in model.tags}
    if task == "automatic-speech-recognition" and ("ctranslate2" in tags or model.library_name == "ctranslate2"):
        tips.append(RunnerSuggestion(
            tool="faster-whisper",
            command=f'pip install faster-whisper && python -c "from faster_whisper import WhisperModel; '
                    f"m = WhisperModel('{repo}'); print(' '.join(s.text for s in m.transcribe('audio.mp3')[0]))\"",
            note="Transcripción rápida en CPU o GPU.", url="https://github.com/SYSTRAN/faster-whisper"))
    elif task == "automatic-speech-recognition" and "nemo" in tags:
        tips.append(RunnerSuggestion(
            tool="NVIDIA NeMo",
            command=f'pip install -U "nemo_toolkit[asr]" && python -c "import nemo.collections.asr as asr; '
                    f"print(asr.models.ASRModel.from_pretrained('{repo}').transcribe(['audio.wav'])[0].text)\"",
            note="Librería oficial de NVIDIA para estos modelos.", url="https://github.com/NVIDIA/NeMo"))
    elif model.library_name == "diffusers":
        tips.append(RunnerSuggestion(
            tool="Diffusers",
            command=f"pip install diffusers transformers accelerate torch && python -c \"from diffusers import "
                    f"DiffusionPipeline; p = DiffusionPipeline.from_pretrained('{repo}')\"",
            note="Librería oficial de Hugging Face para imagen y vídeo; la ficha trae el ejemplo completo.",
            url=f"https://huggingface.co/{repo}"))
        tips.append(RunnerSuggestion(
            tool="ComfyUI", command="",
            note="Programa con ventanas para generar imagen y vídeo; carga el modelo desde su gestor.",
            url="https://www.comfy.org"))
    elif model.library_name in ("diffusion-single-file", "gguf") or "comfyui" in tags:
        tips.append(RunnerSuggestion(
            tool="ComfyUI", command="",
            note="Descarga el archivo que indica la ficha y cárgalo en ComfyUI (programa con ventanas).",
            url="https://www.comfy.org"))
    elif model.library_name == "sentence-transformers":
        tips.append(RunnerSuggestion(
            tool="sentence-transformers",
            command=f"pip install sentence-transformers && python -c \"from sentence_transformers import "
                    f"SentenceTransformer; print(SentenceTransformer('{repo}').encode(['hola']).shape)\"",
            note="Para generar embeddings.", url="https://sbert.net"))
    elif model.library_name == "transformers" and task:
        tips.append(RunnerSuggestion(
            tool="Transformers",
            command=f"pip install transformers torch && python -c \"from transformers import pipeline; "
                    f"p = pipeline('{task}', model='{repo}')\"",
            note="Librería oficial de Hugging Face; la ficha del modelo trae ejemplos de uso.",
            url=f"https://huggingface.co/{repo}"))
    if not tips:
        tips.append(RunnerSuggestion(
            tool="Ficha del modelo", command="",
            note="No usa una herramienta estándar: sigue las instrucciones de uso de su ficha.",
            url=f"https://huggingface.co/{repo}"))
    return tips
