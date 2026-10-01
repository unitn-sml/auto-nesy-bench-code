"""Tie the lifetime of worker processes to their parent."""

import ctypes
import platform
import signal as signal_module

_PR_SET_PDEATHSIG = 1


def set_pdeathsig(sig: int = None) -> None:
    """Ask the kernel to send *sig* (default SIGKILL) to this process when its parent dies.

    Linux only (``prctl(PR_SET_PDEATHSIG)``), a no-op elsewhere. Called first
    in every model-counting worker so that no worker outlives a killed
    evaluation process.
    """
    if platform.system() != "Linux":
        return
    if sig is None:
        sig = signal_module.SIGKILL
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except OSError:
        pass
