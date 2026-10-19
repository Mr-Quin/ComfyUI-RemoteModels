"""End-to-end check of a client and a server ComfyUI, both running this pack.

Run on the client machine with ComfyUI's Python, since it compares files from the client's output folder:

  python tests/e2e.py --server http://192.168.1.50:8188 --output-dir /path/to/ComfyUI/output \\
      --clip qwen_3_06b_base.safetensors --type stable_diffusion \\
      --vae qwen_image_vae.safetensors --latent example.latent

The encoder and VAE files must exist on both machines (the comparison runs them locally too), and
--latent names a .latent file in the client's input folder. Serving must be on at the server.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote_models import protocol  # noqa: E402

PROMPT = "A lighthouse on a rocky coast at dusk, waves breaking, warm light in the window."


def http(base, path, payload=None, raw=None, headers=None, timeout=900):
    headers = {"Content-Type": "application/json", **(headers or {})}
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    req = urllib.request.Request(base + path, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class Checker:
    def __init__(self, args):
        self.a = args
        self.run_id = random.randint(10000, 99999)
        self.failures = []

    def expect(self, name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
        if not ok:
            self.failures.append(name)

    def run_graph(self, graph):
        code, body = http(self.a.client, "/prompt", {"prompt": graph})
        if code != 200:
            raise RuntimeError(body.decode()[:800])
        pid = json.loads(body)["prompt_id"]
        while True:
            h = json.loads(http(self.a.client, "/history/" + pid)[1])
            if pid in h and h[pid]["status"].get("completed") is not None:
                break
            time.sleep(0.2)
        if h[pid]["status"]["status_str"] != "success":
            errors = [m[1].get("exception_message", "") for m in h[pid]["status"]["messages"] if m[0] == "execution_error"]
            raise RuntimeError("; ".join(errors)[:800])

    def newest(self, pattern):
        return max(glob.glob(os.path.join(self.a.output_dir, pattern)), key=os.path.getmtime)

    def protocol_checks(self):
        server, proto = self.a.server, {protocol.HEADER: str(protocol.PROTOCOL)}
        code, body = http(server, "/remote_models/status")
        status = json.loads(body) if code == 200 else {}
        self.expect("server status", code == 200 and status.get("protocol") == protocol.PROTOCOL, f"{code} {status.get('version')}")
        good = protocol.pack({"clip_names": [self.a.clip], "type": self.a.type, "text": f"protocol check {self.run_id}"})
        self.expect("no protocol header is refused", http(server, "/remote_models/clip/run", raw=good)[0] == 409)
        self.expect("encode with header", http(server, "/remote_models/clip/run", raw=good, headers=proto)[0] == 200)
        missing = protocol.pack({"vae_name": "missing.safetensors"})
        self.expect("missing file is 404", http(server, "/remote_models/vae/run", raw=missing, headers=proto)[0] == 404)
        self.expect("garbage body is 400", http(server, "/remote_models/vae/run", raw=b"not a torch file", headers=proto)[0] == 400)
        self.expect("unknown kind is 404", http(server, "/remote_models/unet/list")[0] == 404)
        for kind in ("clip", "vae"):
            p = json.loads(http(self.a.client, "/remote_models/probe", {"url": server, "kind": kind})[1])
            self.expect(f"probe {kind}", p.get("ok") is True, str(p.get("error", len(p.get("files", [])))))
        own = json.loads(http(self.a.client, "/remote_models/probe", {"url": self.a.client, "kind": "clip"})[1])
        self.expect("own URL is rejected", own.get("ok") is False and "is this ComfyUI" in own.get("error", ""))

    def clip_checks(self):
        def loader(mode):
            if mode == "core":
                return {"class_type": "CLIPLoader", "inputs": {"clip_name": self.a.clip, "type": self.a.type}}
            url = "local" if mode == "local" else self.a.server
            return {"class_type": "RemoteCLIPLoader", "inputs": {"server_url": url, "clip_name": self.a.clip, "type": self.a.type}}

        for mode in ("core", "remote", "local"):
            self.run_graph({"1": loader(mode), "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 0]}},
                            "3": {"class_type": "SaveConditioning", "inputs": {"conditioning": ["2", 0],
                                                                               "filename_prefix": f"remote_models_e2e/c{self.run_id}_{mode}"}}})
        base = load_file(self.newest(f"remote_models_e2e/c{self.run_id}_core_*.safetensors"))
        for mode, tolerance in (("remote", 0.02), ("local", 0.0)):
            other = load_file(self.newest(f"remote_models_e2e/c{self.run_id}_{mode}_*.safetensors"))
            worst = max(float((base[k].float() - other[k].float()).abs().max()) for k in base if base[k].numel())
            self.expect(f"CLIP {mode} matches core", worst <= tolerance, f"max diff {worst:.5f}")

    def vae_checks(self):
        def loader(mode):
            if mode == "core":
                return {"class_type": "VAELoader", "inputs": {"vae_name": self.a.vae}}
            url = "local" if mode == "local" else self.a.server
            return {"class_type": "RemoteVAELoader", "inputs": {"server_url": url, "vae_name": self.a.vae}}

        decoders = {"decode": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["1", 0]}},
                    "tiled": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["2", 0], "vae": ["1", 0], "tile_size": 512,
                                                                         "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}}}
        for op, node in decoders.items():
            for mode in ("core", "remote", "local"):
                self.run_graph({"1": loader(mode), "2": {"class_type": "LoadLatent", "inputs": {"latent": self.a.latent}}, "3": node,
                                "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0],
                                                                            "filename_prefix": f"remote_models_e2e/v{self.run_id}_{op}_{mode}"}}})
            base = np.asarray(Image.open(self.newest(f"remote_models_e2e/v{self.run_id}_{op}_core_*.png"))).astype(np.int16)
            for mode, tolerance in (("remote", 1.0), ("local", 0.0)):
                other = np.asarray(Image.open(self.newest(f"remote_models_e2e/v{self.run_id}_{op}_{mode}_*.png"))).astype(np.int16)
                mean = float(np.abs(base - other).mean())
                self.expect(f"VAE {op} {mode} matches core", base.shape == other.shape and mean <= tolerance, f"mean diff {mean:.3f}/255")
        for mode in ("core", "remote"):
            self.run_graph({"1": loader(mode), "9": loader("core"), "2": {"class_type": "LoadLatent", "inputs": {"latent": self.a.latent}},
                            "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["9", 0]}},
                            "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 0]}},
                            "5": {"class_type": "SaveLatent", "inputs": {"samples": ["4", 0], "filename_prefix": f"remote_models_e2e/e{self.run_id}_{mode}"}}})
        a = load_file(self.newest(f"remote_models_e2e/e{self.run_id}_core_*.latent"))
        b = load_file(self.newest(f"remote_models_e2e/e{self.run_id}_remote_*.latent"))
        key = next(k for k in a if a[k].numel())
        cos = torch.nn.functional.cosine_similarity(a[key].float().flatten(), b[key].float().flatten(), dim=0).item()
        self.expect("VAE encode remote matches core", cos > 0.999, f"cosine {cos:.6f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--client", default="http://127.0.0.1:8188")
    p.add_argument("--server", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--type", required=True)
    p.add_argument("--vae")
    p.add_argument("--latent")
    c = Checker(p.parse_args())
    c.protocol_checks()
    c.clip_checks()
    if c.a.vae and c.a.latent:
        c.vae_checks()
    print(f"\n{len(c.failures)} failed" if c.failures else "\nall passed")
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
