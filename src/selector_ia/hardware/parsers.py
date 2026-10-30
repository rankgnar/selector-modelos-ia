"""Pure parsers for command output, shared by local and SSH detection (easy to test)."""

import re

from selector_ia.hardware.profile import GIB, GPU

MIB = 1024 ** 2
# amdgpu reports a small carve-out (e.g. 512 MB) for APUs; below this we treat the GPU as integrated.
INTEGRATED_VRAM_THRESHOLD = 2 * GIB


def parse_nvidia_smi(text: str) -> list[GPU]:
    """`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` (MiB)."""
    gpus = []
    for line in text.strip().splitlines():
        if "," not in line:
            continue
        name, _, mem = line.rpartition(",")
        try:
            vram = int(float(mem.strip())) * MIB
        except ValueError:
            vram = None
        gpus.append(GPU(name=name.strip(), vendor="nvidia", vram_bytes=vram))
    return gpus


def parse_meminfo_total(text: str) -> int | None:
    match = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def parse_lscpu_model(text: str) -> str | None:
    match = re.search(r"^Model name:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_cpuinfo_model(text: str) -> str | None:
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_df_avail(text: str) -> int | None:
    """`df -B1 --output=avail <path>`: header line then bytes."""
    for line in text.strip().splitlines()[1:]:
        first = line.split()[0] if line.split() else ""
        if first.isdigit():
            return int(first)
    return None


def parse_os_release(text: str) -> str | None:
    match = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_lspci_gpus(text: str) -> list[tuple[str, str]]:
    """(pci slot, device description) for display controllers."""
    found = []
    for line in text.splitlines():
        match = re.match(r"^(\S+)\s+(VGA compatible controller|3D controller|Display controller):\s*(.+)$", line)
        if match:
            found.append((match.group(1), match.group(3).strip()))
    return found


def clean_lspci_name(description: str) -> str:
    """'Advanced Micro Devices, Inc. [AMD/ATI] Rembrandt [Radeon 680M] (rev 12)' -> 'AMD Radeon 680M (Rembrandt)'."""
    text = re.sub(r"\s*\(rev [0-9a-f]+\)\s*$", "", description.strip())
    brackets = re.findall(r"\[([^\]]+)\]", text)
    if not brackets:
        return text
    vendor = _vendor_label(text)
    product = brackets[-1]
    codename = re.sub(r"\[[^\]]+\]", "", text.split("]", 1)[-1]).strip(" ,")
    name = product if product.lower().startswith(vendor.lower()) else f"{vendor} {product}".strip()
    return f"{name} ({codename})" if codename and codename.lower() not in name.lower() else name


def _vendor_label(text: str) -> str:
    return {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}.get(vendor_from_name(text), "")


def amd_gpus_from_vram(vram_values: list[int], names: list[str]) -> list[GPU]:
    """amdgpu sysfs `mem_info_vram_total` values; names from lspci when available."""
    gpus = []
    for i, vram in enumerate(vram_values):
        name = clean_lspci_name(names[i]) if i < len(names) else "GPU AMD"
        integrated = vram < INTEGRATED_VRAM_THRESHOLD
        gpus.append(GPU(name=name, vendor="amd", vram_bytes=vram, integrated=integrated))
    return gpus


def parse_int_lines(text: str) -> list[int]:
    return [int(x) for x in text.split() if x.strip().isdigit()]


def vendor_from_name(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered or "rtx" in lowered:
        return "nvidia"
    if "amd" in lowered or "radeon" in lowered or "ati" in lowered.split():
        return "amd"
    if "intel" in lowered:
        return "intel"
    if "apple" in lowered:
        return "apple"
    return "other"


def is_integrated_name(name: str) -> bool:
    """Heuristic for GPUs that share system RAM (Intel iGPU, AMD APU graphics)."""
    lowered = name.lower()
    if "intel" in lowered and "arc" not in lowered:
        return True
    return bool(re.search(r"radeon\(tm\) graphics|radeon graphics|vega \d+ graphics|\d{3}m\b", lowered))
