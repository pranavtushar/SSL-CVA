
from __future__ import annotations

import torch

_FLAG = "_latest_child_speech_torch_load_patched"


def _apply() -> None:
    if getattr(torch, _FLAG, False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        try:
            return _orig(*args, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return _orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]
    setattr(torch, _FLAG, True)


_apply()
