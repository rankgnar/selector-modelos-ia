"""Saved computers ("Mis equipos"): this PC, a VPS or a rented GPU server, typed by hand or from a template.

Templates are generic hardware sizes the student edits (not model data): they only pre-fill the form.
"""

from pydantic import BaseModel

from selector_ia.errors import SelectorError
from selector_ia.hardware.profile import GIB, GPU, HardwareProfile

MACHINE_LABELS = {"pc": "PC", "vps": "VPS", "gpu_server": "Servidor GPU"}


class Template(BaseModel):
    id: str
    label: str
    machine: str
    ram_gb: float
    gpu: str = ""
    vram_gb: float | None = None
    gpu_count: int = 0


TEMPLATES = [
    *(Template(id=f"vps-{n}", label=f"VPS {n} GB sin GPU", machine="vps", ram_gb=n) for n in (4, 8, 16, 32)),
    *(Template(id=f"pc-gpu-{v}", label=f"PC con GPU NVIDIA de {v} GB", machine="pc", ram_gb=32,
               gpu=f"NVIDIA (GPU de {v} GB)", vram_gb=v, gpu_count=1) for v in (8, 12, 16, 24)),
    Template(id="gpu-48", label="Servidor GPU 48 GB (tipo L40S)", machine="gpu_server", ram_gb=64,
             gpu="NVIDIA L40S", vram_gb=48, gpu_count=1),
    Template(id="gpu-80", label="Servidor GPU 80 GB (tipo A100 / H100)", machine="gpu_server", ram_gb=128,
             gpu="NVIDIA A100 / H100", vram_gb=80, gpu_count=1),
]


class EquipmentForm(BaseModel):
    name: str
    machine: str = "pc"
    ram_gb: float
    cpu_threads: int | None = None
    gpu: str = ""
    vram_gb: float | None = None
    gpu_count: int = 1


def profile_from_form(form: EquipmentForm) -> HardwareProfile:
    name = form.name.strip()
    if not name:
        raise SelectorError("Escribe un nombre para el equipo.")
    if len(name) > 60:
        raise SelectorError("El nombre es demasiado largo (máximo 60 caracteres).")
    if form.machine not in MACHINE_LABELS:
        raise SelectorError("Tipo de equipo no válido.")
    if not 0.5 <= form.ram_gb <= 4096:
        raise SelectorError("Escribe la memoria RAM en GB (por ejemplo, 16).")
    gpus = []
    if form.gpu.strip() or form.vram_gb:
        if form.vram_gb is not None and not 0 < form.vram_gb <= 1024:
            raise SelectorError("La memoria de la GPU debe estar entre 1 y 1024 GB.")
        count = max(1, min(form.gpu_count or 1, 16))
        gpus = [GPU(name=form.gpu.strip() or "GPU", vendor=_vendor(form.gpu),
                    vram_bytes=int(form.vram_gb * GIB) if form.vram_gb else None) for _ in range(count)]
    return HardwareProfile(name=name, kind="remote_manual", machine=form.machine, os="Linux",
                           cpu_threads=form.cpu_threads, ram_bytes=int(form.ram_gb * GIB), gpus=gpus)


def _vendor(name: str) -> str:
    lowered = name.lower()
    for vendor in ("nvidia", "amd", "intel"):
        if vendor in lowered or (vendor == "nvidia" and any(k in lowered for k in ("rtx", "gtx", "a100", "h100", "l40"))):
            return vendor
    return "other"


def profile_json(p: HardwareProfile) -> dict:
    dedicated = [g for g in p.gpus if not g.integrated]
    vram = p.dedicated_vram_bytes
    return {
        "name": p.name,
        "kind": p.kind,
        "machine": p.machine or ("pc" if p.kind == "local" else None),
        "detected": p.kind == "local",
        "ram_gb": round(p.ram_bytes / GIB, 1) if p.ram_bytes else None,
        "cpu_threads": p.cpu_threads,
        "gpu": dedicated[0].name if dedicated else "",
        "vram_gb": round(vram / GIB / max(len(dedicated), 1), 1) if vram else None,
        "gpu_count": len(dedicated),
        "summary": p.summary(),
        "warnings": p.warnings,
    }
