"""Hardware profile: where a model will run."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

GIB = 1024 ** 3

Vendor = Literal["nvidia", "amd", "intel", "apple", "other"]
ProfileKind = Literal["local", "remote_manual", "remote_ssh"]
Machine = Literal["pc", "vps", "gpu_server"]


class GPU(BaseModel):
    name: str
    vendor: Vendor = "other"
    vram_bytes: int | None = None   # None = sin datos
    integrated: bool = False        # iGPU sharing system RAM (not useful as dedicated VRAM)
    unified_memory: bool = False    # Apple Silicon: GPU uses system RAM directly


class SSHTarget(BaseModel):  # kept only so computers saved by older versions still load
    host: str                       # hostname, IP or ~/.ssh/config alias
    port: int | None = None
    username: str | None = None
    key_path: str | None = None
    use_password: bool = False      # password stored in the keyring, never here


class HardwareProfile(BaseModel):
    name: str
    kind: ProfileKind
    machine: Machine | None = None  # what the student called it: PC, VPS or rented GPU server
    os: str | None = None
    arch: str | None = None
    cpu_model: str | None = None
    cpu_threads: int | None = None
    ram_bytes: int | None = None
    gpus: list[GPU] = Field(default_factory=list)
    disk_free_bytes: int | None = None
    detected_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    ssh: SSHTarget | None = None

    @property
    def dedicated_vram_bytes(self) -> int:
        """Sum over dedicated GPUs with known VRAM (llama.cpp / vLLM can split across GPUs)."""
        return sum(g.vram_bytes or 0 for g in self.gpus if not g.integrated and not g.unified_memory)

    @property
    def has_unified_memory(self) -> bool:
        return any(g.unified_memory for g in self.gpus)

    @property
    def gpu_vendors(self) -> set[str]:
        return {g.vendor for g in self.gpus if not g.integrated}

    def summary(self) -> str:
        parts = [self.cpu_model or "CPU sin datos"]
        parts.append(f"{_gb(self.ram_bytes)} RAM" if self.ram_bytes else "RAM sin datos")
        for g in self.gpus:
            vram = _gb(g.vram_bytes) if g.vram_bytes else "VRAM sin datos"
            kind = " (integrada)" if g.integrated else " (memoria unificada)" if g.unified_memory else ""
            parts.append(f"{g.name} {vram}{kind}")
        if not self.gpus:
            parts.append("sin GPU")
        return " · ".join(parts)


def _gb(n: int) -> str:
    return f"{n / GIB:.1f} GB".replace(".", ",")
