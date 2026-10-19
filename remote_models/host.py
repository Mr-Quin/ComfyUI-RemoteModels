"""Serves models to other ComfyUI instances. Each model kind plugs in as a Kind."""
import asyncio
import collections
import gc
import hashlib
import json
import logging
import os
import threading
import time
import uuid

import torch
from aiohttp import web

import comfy.model_management as mm
import folder_paths

from . import protocol, settings

RESULT_CACHE_BYTES = 256 * 1024 * 1024
RESULT_CACHE_ENTRIES = 64
RECENT_KEEP = 3
BUSY_WAIT_SECONDS = 600
# Lets a client notice it was pointed at its own ComfyUI, which would wait on itself.
INSTANCE = uuid.uuid4().hex

log = logging.getLogger("remote_models")


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Kind:
    """One kind of model the host can serve.

    A key identifies one loaded model. It must be hashable and round-trip through
    key_to_json/key_from_json so the recently used list survives restarts."""

    name = ""
    folder = ""
    cache_results = False

    def files(self):
        return folder_paths.get_filename_list(self.folder)

    def options(self):
        return {"files": self.files()}

    def key(self, req):
        raise NotImplementedError

    def key_to_json(self, key):
        raise NotImplementedError

    def key_from_json(self, data):
        raise NotImplementedError

    def load(self, key):
        raise NotImplementedError

    def run(self, model, req):
        raise NotImplementedError

    def patcher(self, model):
        return model.patcher

    def to_wire(self, result, req):
        return result

    def require_files(self, names):
        known = set(self.files())
        missing = [n for n in names if n not in known]
        if missing:
            raise RequestError(404, f"Not found in {self.folder} on server: " + ", ".join(missing))


KINDS = {}


def register(cls):
    KINDS[cls.name] = cls()
    return cls


def host_busy():
    """comfy.model_management is not thread-safe, so requests from HTTP threads and the
    housekeeping thread must not load or unload models while the prompt executor is using them."""
    from server import PromptServer

    return PromptServer.instance.prompt_queue.get_tasks_remaining() > 0


def _recent_path():
    return os.path.join(folder_paths.get_user_directory(), "default", "remote_models_recent.json")


def read_recent():
    try:
        with open(_recent_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    recent = []
    for item in data if isinstance(data, list) else []:
        kind = KINDS.get(item.get("kind")) if isinstance(item, dict) else None
        if kind is None:
            continue
        try:
            recent.append((kind.name, kind.key_from_json(item["key"])))
        except (KeyError, TypeError, ValueError):
            continue
    return recent


def remember(kind, key):
    recent = [r for r in read_recent() if r != (kind.name, key)]
    recent.insert(0, (kind.name, key))
    try:
        with open(_recent_path(), "w", encoding="utf-8") as f:
            json.dump([{"kind": k, "key": KINDS[k].key_to_json(v)} for k, v in recent[:RECENT_KEEP]], f, indent=1)
    except OSError:
        log.warning("could not save the recently used model list")


class Pool:
    def __init__(self):
        self.lock = threading.Lock()
        self.entries = {}
        self.results = collections.OrderedDict()
        self.results_bytes = 0

    def _entry(self, kind, key):
        entry = self.entries.get((kind.name, key))
        if entry is None:
            start = time.monotonic()
            entry = {"model": kind.load(key), "last_used": time.monotonic()}
            log.info("loaded %s %s in %.1fs", kind.name, key, time.monotonic() - start)
            self.entries[(kind.name, key)] = entry
            remember(kind, key)
        return entry

    def run(self, kind, req):
        key = kind.key(req)
        with self.lock, torch.inference_mode():
            entry = self._entry(kind, key)
            try:
                return kind.run(entry["model"], req)
            finally:
                entry["last_used"] = time.monotonic()

    def run_packed(self, kind, req):
        return protocol.pack(protocol.to_cpu(kind.to_wire(self.run(kind, req), req)))

    def reload_recent(self):
        for name, key in read_recent():
            kind = KINDS[name]
            if (name, key) in self.entries:
                continue
            with self.lock:
                try:
                    # key() rejects entries whose files were removed since they were recorded.
                    kind.key(kind.key_to_json(key))
                    self._entry(kind, key)
                except Exception:
                    log.exception("could not reload %s %s", name, key)

    def cached_result(self, digest):
        hit = self.results.get(digest)
        if hit is not None:
            self.results.move_to_end(digest)
        return hit

    def store_result(self, digest, blob):
        if len(blob) > RESULT_CACHE_BYTES // 4:
            return
        self.results[digest] = blob
        self.results_bytes += len(blob)
        while self.results and (len(self.results) > RESULT_CACHE_ENTRIES or self.results_bytes > RESULT_CACHE_BYTES):
            _, old = self.results.popitem(last=False)
            self.results_bytes -= len(old)

    def unload(self, keys=None, wait=True):
        if not self.lock.acquire(blocking=wait):
            return 0
        try:
            keys = list(self.entries) if keys is None else keys
            for key in keys:
                entry = self.entries.pop(key, None)
                if entry is not None:
                    try:
                        mm.unload_model_and_clones(KINDS[key[0]].patcher(entry["model"]))
                    except Exception:
                        log.exception("unload failed for %s", key)
                    del entry
            if keys:
                gc.collect()
                mm.cleanup_models()
                mm.soft_empty_cache()
            return len(keys)
        finally:
            self.lock.release()

    def unload_idle(self):
        minutes = settings.idle_minutes()
        if minutes <= 0 or not self.entries:
            return
        now = time.monotonic()
        stale = [k for k, e in list(self.entries.items()) if now - e.get("last_used", now) >= minutes * 60]
        if stale and self.unload(stale, wait=False):
            log.info("unloaded %d idle model(s)", len(stale))

    def describe(self):
        now = time.monotonic()
        return [
            {"kind": name, **KINDS[name].key_to_json(key), "idle_seconds": round(now - e.get("last_used", now), 1)}
            for (name, key), e in list(self.entries.items())
        ]


pool = Pool()


def _error(status, message):
    return web.json_response({"error": message}, status=status)


DISABLED = "Model serving is disabled on this server. Enable 'Serve models to other machines' in its ComfyUI settings."


def _gate():
    if not settings.serving_enabled():
        return _error(403, DISABLED)
    return None


def _kind(request):
    kind = KINDS.get(request.match_info["kind"])
    if kind is None:
        return None, _error(404, f"Unknown model kind '{request.match_info['kind']}'. This server serves: {', '.join(KINDS)}.")
    return kind, None


def _housekeeping():
    reload_pending = True
    while True:
        time.sleep(15)
        if host_busy():
            continue
        try:
            if reload_pending and settings.serving_enabled():
                reload_pending = False
                if settings.reload_recent():
                    pool.reload_recent()
            pool.unload_idle()
        except Exception:
            log.exception("model housekeeping failed")


def _digest(kind, body):
    return hashlib.sha256(kind.name.encode() + b"\0" + body).digest()


def _parse(kind, body):
    req = protocol.unpack(body)
    if not isinstance(req, dict):
        raise ValueError("request is not a dict")
    kind.key(req)
    return req


def register_routes():
    from server import PromptServer

    routes = PromptServer.instance.routes

    @routes.get("/remote_models/status")
    async def status(request):
        identity = {"version": protocol.VERSION, "protocol": protocol.PROTOCOL, "instance": INSTANCE}
        if not settings.serving_enabled():
            return web.json_response({**identity, "enabled": False, "error": DISABLED})
        return web.json_response(
            {
                **identity,
                "enabled": True,
                "kinds": list(KINDS),
                "device": mm.get_torch_device_name(mm.get_torch_device()),
                "loaded": pool.describe(),
                "idle_minutes": settings.idle_minutes(),
                "reload_recent": settings.reload_recent(),
                "result_cache_entries": len(pool.results),
            }
        )

    @routes.get("/remote_models/{kind}/list")
    async def list_models(request):
        denied = _gate()
        if denied:
            return denied
        kind, missing = _kind(request)
        return missing or web.json_response(kind.options())

    @routes.post("/remote_models/unload")
    async def unload(request):
        denied = _gate()
        if denied:
            return denied
        if host_busy():
            return _error(503, "Server is busy with its own prompts; unload later.")
        count = await asyncio.get_running_loop().run_in_executor(None, pool.unload)
        return web.json_response({"unloaded": count})

    @routes.post("/remote_models/{kind}/run")
    async def run(request):
        denied = _gate()
        if denied:
            return denied
        kind, missing = _kind(request)
        if missing:
            return missing
        client_protocol = request.headers.get(protocol.HEADER)
        if client_protocol != str(protocol.PROTOCOL):
            return _error(409, f"Protocol mismatch: server {protocol.VERSION} speaks protocol {protocol.PROTOCOL}, "
                               f"client sent {client_protocol or 'none'}. Update the Remote Models pack on both machines.")
        loop = asyncio.get_running_loop()
        body = await request.read()
        digest = await loop.run_in_executor(None, _digest, kind, body)
        if kind.cache_results:
            hit = pool.cached_result(digest)
            if hit is not None:
                return web.Response(body=hit, content_type="application/octet-stream", headers={"X-Remote-Models-Cache": "hit"})
        try:
            req = await loop.run_in_executor(None, _parse, kind, body)
        except RequestError as e:
            return _error(e.status, str(e))
        except Exception as e:
            return _error(400, f"Malformed {kind.name} request: {type(e).__name__}: {e}")
        waited = 0.0
        while host_busy():
            if waited >= BUSY_WAIT_SECONDS:
                return _error(503, "Server is busy with its own prompts; try again later.")
            await asyncio.sleep(0.5)
            waited += 0.5
        try:
            blob = await loop.run_in_executor(None, pool.run_packed, kind, req)
        except RequestError as e:
            return _error(e.status, str(e))
        except Exception as e:
            log.exception("%s request failed", kind.name)
            return _error(500, f"{kind.name} failed on server: {type(e).__name__}: {e}")
        if kind.cache_results:
            pool.store_result(digest, blob)
        return web.Response(body=blob, content_type="application/octet-stream")

    threading.Thread(target=_housekeeping, name="remote_models_housekeeping", daemon=True).start()
