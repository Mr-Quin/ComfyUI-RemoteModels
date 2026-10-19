import torch

import comfy.sd
import comfy.utils
import folder_paths

from . import client
from .client import CATEGORY, PLACEHOLDER, AnyCombo
from .host import Kind, RequestError, register

TILE_ARGS = {"tile_x", "tile_y", "overlap", "tile_t", "overlap_t"}
OPS = {"decode": {"vae_options"}, "decode_tiled": TILE_ARGS, "encode": set(), "encode_tiled": TILE_ARGS}


def _ints(value):
    if isinstance(value, int):
        return value
    if isinstance(value, tuple) and all(isinstance(v, int) for v in value):
        return value
    return None


def vae_info(vae):
    """Plain attributes nodes read from a VAE. Ratios that are callables (some video VAEs) stay on the server."""
    return {
        "latent_channels": vae.latent_channels,
        "latent_dim": vae.latent_dim,
        "output_channels": vae.output_channels,
        "not_video": bool(getattr(vae, "not_video", False)),
        "downscale_ratio": _ints(vae.downscale_ratio),
        "upscale_ratio": _ints(vae.upscale_ratio),
        "downscale_index_formula": _ints(getattr(vae, "downscale_index_formula", None)),
        "upscale_index_formula": _ints(getattr(vae, "upscale_index_formula", None)),
        "spacial_compression_decode": _ints(vae.spacial_compression_decode()),
        "spacial_compression_encode": _ints(vae.spacial_compression_encode()),
        "temporal_compression_decode": vae.temporal_compression_decode(),
        "vae_dtype": str(vae.vae_dtype).removeprefix("torch."),
    }


@register
class VaeKind(Kind):
    name = "vae"
    folder = "vae"

    def key(self, req):
        name = req.get("vae_name")
        if not isinstance(name, str):
            raise RequestError(400, "vae_name must be a file name.")
        op = req.get("op", "info")
        if op != "info" and (op not in OPS or not isinstance(req.get("input"), torch.Tensor)):
            raise RequestError(400, f"op must be one of info, {', '.join(OPS)} and input must be a tensor.")
        self.require_files([name])
        return (name,)

    def key_to_json(self, key):
        return {"vae_name": key[0]}

    def key_from_json(self, data):
        return (data["vae_name"],)

    def load(self, key):
        sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise(self.folder, key[0]), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    def run(self, vae, req):
        op = req.get("op", "info")
        if op == "info":
            return vae_info(vae)
        args = {k: v for k, v in req.get("args", {}).items() if k in OPS[op]}
        x = req["input"]
        if op.startswith("encode"):
            x = x.float()
        return getattr(vae, op)(x, **args)

    def to_wire(self, result, req):
        # Decoded pixels are the bulk of the traffic; half precision still holds far more than 8-bit output.
        if req.get("op", "").startswith("decode") and isinstance(result, torch.Tensor) and result.is_floating_point():
            return result.half()
        return result


class RemoteVAE:
    """Stands in for comfy.sd.VAE; decoding and encoding happen on the server."""

    def __init__(self, url, vae_name, info):
        self.url = url
        self.vae_name = vae_name
        self.info = info
        for attr in ("latent_channels", "latent_dim", "output_channels", "not_video", "downscale_ratio", "upscale_ratio",
                     "downscale_index_formula", "upscale_index_formula"):
            if info.get(attr) is not None:
                setattr(self, attr, info[attr])
        self.vae_dtype = getattr(torch, info["vae_dtype"], torch.float32)
        self.device = torch.device("cpu")

    def _run(self, op, x, **args):
        if self.url != client.LOCAL:
            x = (x.half() if op.startswith("encode") else x).cpu().contiguous()
        out = client.run(self.url, VaeKind.name, {"vae_name": self.vae_name, "op": op, "input": x,
                                                   "args": {k: v for k, v in args.items() if v is not None}})
        return out.float() if op.startswith("decode") else out

    def decode(self, samples_in, vae_options={}):
        return self._run("decode", samples_in, vae_options=vae_options or None)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        return self._run("decode_tiled", samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap, tile_t=tile_t, overlap_t=overlap_t)

    def encode(self, pixel_samples):
        return self._run("encode", pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        return self._run("encode_tiled", pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap, tile_t=tile_t, overlap_t=overlap_t)

    def spacial_compression_decode(self):
        return self.info["spacial_compression_decode"]

    def spacial_compression_encode(self):
        return self.info["spacial_compression_encode"]

    def temporal_compression_decode(self):
        return self.info["temporal_compression_decode"]

    def throw_exception_if_invalid(self):
        pass

    def is_dynamic(self):
        return False

    def __getattr__(self, name):
        raise AttributeError(f"RemoteVAE has no '{name}'; this node needs the VAE weights locally and does not work with a remote VAE")


class RemoteVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": "", "tooltip": client.URL_TOOLTIP}),
                "vae_name": (AnyCombo([PLACEHOLDER]),),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "VAE that decodes and encodes on another ComfyUI instance."

    def load(self, server_url, vae_name):
        url = client.connect(server_url, VaeKind.name, [vae_name])
        info = client.run(url, VaeKind.name, {"vae_name": vae_name, "op": "info"})
        return (RemoteVAE(url, vae_name, info),)


NODES = {"RemoteVAELoader": (RemoteVAELoader, "Load Remote VAE")}
