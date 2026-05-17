from __future__ import annotations

import platform


def beep(enabled: bool = True) -> None:
    if not enabled:
        return
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(880, 55)
        else:
            print("\a", end="")
    except Exception:
        pass
