from __future__ import annotations

import platform
from typing import Optional


class VolumeController:
    def __init__(self):
        self.available = False
        self.endpoint = None
        if platform.system() == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.endpoint = cast(interface, POINTER(IAudioEndpointVolume))
                self.available = True
            except Exception:
                self.available = False

    def set_volume_scalar(self, scalar: float) -> bool:
        scalar = max(0.0, min(1.0, scalar))
        if self.available and self.endpoint is not None:
            self.endpoint.SetMasterVolumeLevelScalar(float(scalar), None)
            return True
        return False

    def mute_toggle(self) -> bool:
        if self.available and self.endpoint is not None:
            current = self.endpoint.GetMute()
            self.endpoint.SetMute(0 if current else 1, None)
            return True
        return False
