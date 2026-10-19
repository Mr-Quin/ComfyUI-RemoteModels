import io

import torch

VERSION = "0.3.0"
# Bump when a request or response format changes; mismatched peers refuse to talk.
PROTOCOL = 2
HEADER = "X-Remote-Models-Protocol"


def pack(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def unpack(data):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu").contiguous()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj
