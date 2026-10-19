import torch

import comfy.sd
import folder_paths

from . import client
from .client import CATEGORY, PLACEHOLDER, AnyCombo
from .host import Kind, RequestError, register


def clip_types():
    return [t.name.lower() for t in comfy.sd.CLIPType]


@register
class ClipKind(Kind):
    name = "clip"
    folder = "text_encoders"
    cache_results = True

    def options(self):
        return {"files": self.files(), "types": clip_types()}

    def key(self, req):
        files, clip_type, device = req.get("clip_names"), req.get("type"), req.get("device", "default")
        if not isinstance(files, (list, tuple)) or not 1 <= len(files) <= 2 or not all(isinstance(f, str) for f in files):
            raise RequestError(400, "clip_names must hold 1 or 2 file names.")
        if clip_type not in clip_types() or device not in ("default", "cpu"):
            raise RequestError(400, "type must be a known CLIP type and device must be default or cpu.")
        self.require_files(files)
        return tuple(files), clip_type, device

    def key_to_json(self, key):
        return {"clip_names": list(key[0]), "type": key[1], "device": key[2]}

    def key_from_json(self, data):
        return tuple(data["clip_names"]), data["type"], data.get("device", "default")

    def load(self, key):
        files, clip_type, device = key
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        return comfy.sd.load_clip(
            ckpt_paths=[folder_paths.get_full_path_or_raise(self.folder, f) for f in files],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=getattr(comfy.sd.CLIPType, clip_type.upper()),
            model_options=model_options,
        )

    def run(self, clip, req):
        previous_layer = clip.layer_idx
        clip.layer_idx = req.get("layer_idx")
        try:
            tokens = clip.tokenize(req["text"], return_word_ids=bool(req.get("return_word_ids", False)), **req.get("kwargs", {}))
            out = clip.encode_from_tokens(tokens, return_pooled=req.get("return_pooled", True), return_dict=True)
        finally:
            clip.layer_idx = previous_layer
        out.pop("hooks", None)
        return out


class RemoteTokens:
    def __init__(self, text, return_word_ids, kwargs):
        self.text = text
        self.return_word_ids = return_word_ids
        self.kwargs = kwargs

    def __getitem__(self, key):
        raise RuntimeError("This encode node edits token lists directly (for example the SDXL, SD3 or Flux dual-prompt "
                           "encoders), which Load Remote CLIP does not support. Use CLIP Text Encode instead.")

    def __setitem__(self, key, value):
        self[key]


class RemoteCLIP:
    """Stands in for comfy.sd.CLIP; tokenizing and encoding both happen on the server."""

    def __init__(self, url, clip_names, clip_type, device):
        self.url = url
        self.clip_names = list(clip_names)
        self.clip_type = clip_type
        self.device = device
        self.layer_idx = None
        self.tokenizer_options = {}
        self.use_clip_schedule = False
        self.apply_hooks_to_conds = None

    def clone(self, *args, **kwargs):
        n = RemoteCLIP(self.url, self.clip_names, self.clip_type, self.device)
        n.layer_idx = self.layer_idx
        n.tokenizer_options = self.tokenizer_options.copy()
        return n

    def add_patches(self, *args, **kwargs):
        raise RuntimeError("CLIP LoRAs are not supported on remote encoders")

    def set_tokenizer_option(self, option_name, value):
        self.tokenizer_options[option_name] = value

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def add_hooks_to_dict(self, pooled_dict):
        return pooled_dict

    def load_model(self, tokens={}):
        return None

    def is_dynamic(self):
        return False

    def tokenize(self, text, return_word_ids=False, **kwargs):
        options = {**self.tokenizer_options, **kwargs.get("tokenizer_options", {})}
        if options:
            kwargs["tokenizer_options"] = options
        return RemoteTokens(text, return_word_ids, kwargs)

    def _remote_encode(self, tokens, return_pooled):
        if not isinstance(tokens, RemoteTokens):
            raise RuntimeError("Remote CLIP can only encode tokens from its own tokenize(); this node tokenizes elsewhere and is not supported.")
        return client.run(self.url, ClipKind.name, {
            "clip_names": self.clip_names,
            "type": self.clip_type,
            "device": self.device,
            "text": tokens.text,
            "return_word_ids": tokens.return_word_ids,
            "kwargs": tokens.kwargs,
            "return_pooled": return_pooled,
            "layer_idx": self.layer_idx,
        })

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        out = self._remote_encode(tokens, return_pooled)
        if return_dict:
            return out
        if return_pooled:
            return out["cond"], out["pooled_output"]
        return out["cond"]

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        out = self._remote_encode(tokens, "unprojected" if unprojected else True)
        cond = out.pop("cond")
        out.update(add_dict)
        return [[cond, out]]

    def encode(self, text):
        return self.encode_from_tokens(self.tokenize(text))

    def __getattr__(self, name):
        raise AttributeError(f"RemoteCLIP has no '{name}'; this operation is not supported on remote encoders")


class RemoteCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": "", "tooltip": client.URL_TOOLTIP}),
                "clip_name": (AnyCombo([PLACEHOLDER]),),
                "type": (AnyCombo(clip_types()),),
            },
            "optional": {
                "clip_name2": (AnyCombo(["none"]),),
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Text encoder that runs on another ComfyUI instance."

    def load(self, server_url, clip_name, type, clip_name2="none", device="default"):
        names = [clip_name] + ([clip_name2] if clip_name2 not in ("", "none", PLACEHOLDER) else [])
        return (RemoteCLIP(client.prepare(server_url, ClipKind.name, names), names, type, device),)


NODES = {"RemoteCLIPLoader": (RemoteCLIPLoader, "Load Remote CLIP")}
