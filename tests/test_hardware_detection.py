from selector_ia.hardware import parsers
from selector_ia.hardware.local import detect_local
from selector_ia.hardware.profile import GIB


def test_nvidia_smi_multi_gpu():
    out = "NVIDIA GeForce RTX 4090, 24564\nNVIDIA A100-SXM4-80GB, 81920\n"
    gpus = parsers.parse_nvidia_smi(out)
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4090", "NVIDIA A100-SXM4-80GB"]
    assert gpus[0].vram_bytes == 24564 * 1024 ** 2 and gpus[0].vendor == "nvidia"


def test_nvidia_smi_empty_or_error():
    assert parsers.parse_nvidia_smi("") == []
    assert parsers.parse_nvidia_smi("NVIDIA-SMI has failed because it couldn't communicate") == []


def test_meminfo_df_lscpu_os_release():
    assert parsers.parse_meminfo_total("MemTotal:       24428484 kB\nMemFree: 1 kB") == 24428484 * 1024
    assert parsers.parse_df_avail("     Avail\n772754575360\n") == 772754575360
    assert parsers.parse_lscpu_model("Architecture: x86_64\nModel name:   AMD EPYC 7763 64-Core Processor\n") \
        == "AMD EPYC 7763 64-Core Processor"
    assert parsers.parse_os_release('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.5 LTS"\n') == "Ubuntu 24.04.5 LTS"


def test_lspci_names_are_cleaned():
    out = ("00:02.0 Host bridge: Intel\n"
           "e5:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Rembrandt [Radeon 680M] (rev 12)\n"
           "01:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 40GB] (rev a1)\n")
    found = parsers.parse_lspci_gpus(out)
    assert [s for s, _ in found] == ["e5:00.0", "01:00.0"]
    assert parsers.clean_lspci_name(found[0][1]) == "AMD Radeon 680M (Rembrandt)"


def test_amd_apu_is_integrated_dedicated_is_not():
    apu, card = parsers.amd_gpus_from_vram([512 * 1024 ** 2, 24 * GIB], [])
    assert apu.integrated and not card.integrated


def test_windows_name_heuristics():
    assert parsers.is_integrated_name("Intel(R) UHD Graphics 770")
    assert not parsers.is_integrated_name("Intel(R) Arc(TM) A770 Graphics")
    assert parsers.is_integrated_name("AMD Radeon(TM) Graphics")
    assert not parsers.is_integrated_name("AMD Radeon RX 7900 XTX")
    assert parsers.vendor_from_name("NVIDIA GeForce RTX 3060") == "nvidia"


def test_detect_local_runs_on_this_machine():
    p = detect_local("Prueba")
    assert p.kind == "local" and p.ram_bytes and p.cpu_threads and p.detected_at
    assert p.disk_free_bytes is not None
