"""Detects this computer's hardware (Windows, Linux, macOS). Read-only."""

import glob
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

from selector_ia.hardware import parsers
from selector_ia.hardware.profile import GPU, HardwareProfile

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: no console flash in the .exe


def _run(args: list[str], timeout: float = 10) -> str:
    """Runs a read-only command; returns '' if missing or failing."""
    if not shutil.which(args[0]):
        return ""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def detect_local(name: str = "Mi PC") -> HardwareProfile:
    warnings: list[str] = []
    system = platform.system()
    gpus = _detect_gpus(system, warnings)
    ram = psutil.virtual_memory().total
    if system == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon: the GPU uses system RAM directly.
        gpus = [GPU(name=_cpu_model(system) or "Apple Silicon", vendor="apple", vram_bytes=ram,
                    unified_memory=True)] + gpus
    if not any(not g.integrated for g in gpus):
        warnings.append("No se ha detectado una GPU dedicada: los modelos se ejecutarán en CPU y RAM.")

    try:
        disk_free = psutil.disk_usage(str(Path.home())).free
    except OSError:
        disk_free = None
        warnings.append("No se ha podido leer el espacio libre en disco.")

    return HardwareProfile(
        name=name,
        kind="local",
        os=_os_name(system),
        arch=platform.machine(),
        cpu_model=_cpu_model(system),
        cpu_threads=psutil.cpu_count(logical=True),
        ram_bytes=ram,
        gpus=gpus,
        disk_free_bytes=disk_free,
        detected_at=datetime.now(timezone.utc),
        warnings=warnings,
    )


def _os_name(system: str) -> str:
    if system == "Linux":
        try:
            pretty = parsers.parse_os_release(Path("/etc/os-release").read_text())
            if pretty:
                return pretty
        except OSError:
            pass
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    return f"{system} {platform.release()}"


def _cpu_model(system: str) -> str | None:
    if system == "Linux":
        try:
            return parsers.parse_cpuinfo_model(Path("/proc/cpuinfo").read_text())
        except OSError:
            return None
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or None
    if system == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor() or None
    return platform.processor() or None


def _detect_gpus(system: str, warnings: list[str]) -> list[GPU]:
    nvidia = parsers.parse_nvidia_smi(_run(["nvidia-smi", "--query-gpu=name,memory.total",
                                            "--format=csv,noheader,nounits"]))
    if system == "Linux":
        return nvidia + _linux_amd_gpus()
    if system == "Windows":
        # nvidia-smi is more precise; keep registry entries only for the other vendors.
        others = [g for g in _windows_registry_gpus(warnings) if not (nvidia and g.vendor == "nvidia")]
        return nvidia + others
    return nvidia


def _linux_amd_gpus() -> list[GPU]:
    vram = []
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        try:
            vram.append(int(Path(path).read_text().strip()))
        except (OSError, ValueError):
            continue
    names = [desc for _slot, desc in parsers.parse_lspci_gpus(_run(["lspci"]))
             if parsers.vendor_from_name(desc) == "amd"]
    return parsers.amd_gpus_from_vram(vram, names)


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _windows_registry_gpus(warnings: list[str]) -> list[GPU]:
    """Display adapters from the registry. `HardwareInformation.qwMemorySize` is 64-bit
    (WMI's AdapterRAM is capped at 4 GB, so it is not used)."""
    try:
        import winreg
    except ImportError:
        return []
    gpus = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return []
    with root:
        index = 0
        while True:
            try:
                sub = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                    vram = _registry_vram(winreg, key)
            except OSError:
                continue
            if "basic display" in name.lower() or "virtual" in name.lower():
                continue
            vendor = parsers.vendor_from_name(name)
            if vram is None:
                warnings.append(f"No se ha podido leer la VRAM de «{name}»: introdúcela a mano en el perfil.")
            gpus.append(GPU(name=name, vendor=vendor, vram_bytes=vram,
                            integrated=parsers.is_integrated_name(name)))
    return gpus


def _registry_vram(winreg, key) -> int | None:
    for value_name in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
        try:
            value, _type = winreg.QueryValueEx(key, value_name)
        except OSError:
            continue
        if isinstance(value, bytes):
            value = int.from_bytes(value, "little")
        if isinstance(value, int) and value > 0:
            return value
    return None
