import asyncio
import http.client
import json
import logging
import urllib.error
import urllib.request

from aiohttp import web

from . import protocol, settings
from .host import INSTANCE, KINDS, pool

CATEGORY = "loaders/remote"
PLACEHOLDER = "(connect to list)"
LOCAL = "local"
MAX_RESPONSE_BYTES = 1024 * 1024 * 1024
URL_TOOLTIP = ("Base URL of the ComfyUI instance serving models, e.g. http://192.168.1.50:8188. "
               "Use 'local' to share this machine's served models with local workflows instead of loading a second copy.")

log = logging.getLogger("remote_models")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def is_local(url):
    return (url or "").strip().lower() == LOCAL


class AnyCombo(list):
    """Combo whose options come from the remote server, so any value passes validation."""

    def __contains__(self, item):
        return True


class RemoteError(RuntimeError):
    pass


def normalize_url(url):
    url = (url or "").strip().rstrip("/")
    if not url:
        url = str(settings.get(settings.DEFAULT_URL, "") or "").strip().rstrip("/")
    if not url:
        raise RemoteError("No server URL set. Fill in server_url or set 'Default server URL' in the Remote Models settings.")
    if "://" not in url:
        url = "http://" + url
    if not url.lower().startswith(("http://", "https://")) or "#" in url or "?" in url:
        raise RemoteError(f"Server URL must be a plain http:// or https:// address, got {url!r}.")
    return url


def call(url, path, payload=None, timeout=30, max_bytes=MAX_RESPONSE_BYTES):
    base = normalize_url(url)
    headers = {protocol.HEADER: str(protocol.PROTOCOL)}
    if payload is not None:
        headers["Content-Type"] = "application/octet-stream"
    req = urllib.request.Request(base + path, data=payload, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            body = resp.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise RemoteError(f"{base}: response larger than {max_bytes // 2**20} MB")
            return body
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error", "")
        except ValueError:
            detail = ""
        if e.code == 413:
            raise RemoteError(f"{base} refused a {len(payload or b'') // 2**20} MB request. Start its ComfyUI with a larger --max-upload-size.") from None
        if e.code == 404 and not detail:
            raise RemoteError(f"{base} does not have the Remote Models pack installed (404 on {path}).") from None
        raise RemoteError(f"{base}: {detail or 'HTTP %d' % e.code}") from None
    except urllib.error.URLError as e:
        raise RemoteError(f"Cannot reach {base}: {e.reason}") from None
    except TimeoutError:
        raise RemoteError(f"Timed out talking to {base}") from None
    except (http.client.HTTPException, OSError) as e:
        raise RemoteError(f"Lost the connection to {base}: {e}") from None


def fetch_json(url, path, timeout=30):
    return json.loads(call(url, path, timeout=timeout, max_bytes=4 * 1024 * 1024))


def check_version(status, base):
    if status.get("protocol") != protocol.PROTOCOL:
        raise RemoteError(f"{base} runs Remote Models {status.get('version', 'unknown')} (protocol {status.get('protocol', '?')}), "
                          f"this machine runs {protocol.VERSION} (protocol {protocol.PROTOCOL}). Update the pack on both machines.")
    if status.get("version") != protocol.VERSION:
        return f"server {status.get('version')} ≠ {protocol.VERSION}"
    return None


def list_models(url, kind, timeout=30):
    base = normalize_url(url)
    status = fetch_json(base, "/remote_models/status", timeout=timeout)
    note = check_version(status, base)
    if status.get("instance") == INSTANCE:
        raise RemoteError(f"{base} is this ComfyUI. Use 'local' as the server URL to share its models with local workflows.")
    if not status.get("enabled"):
        raise RemoteError(f"{base}: {status.get('error', 'model serving is disabled')}")
    if kind not in status.get("kinds", []):
        raise RemoteError(f"{base} does not serve {kind} models. Update the pack on the server.")
    return status, note, fetch_json(base, f"/remote_models/{kind}/list", timeout=timeout)


def prepare(url, kind, names):
    if is_local(url):
        KINDS[kind].require_files(names)
        return LOCAL
    try:
        _, note, options = list_models(url, kind)
    except RemoteError as e:
        raise RuntimeError(str(e)) from None
    if note:
        log.warning("Remote Models version differs: %s", note)
    missing = [n for n in names if n not in options["files"]]
    if missing:
        raise RuntimeError(f"Not found on {normalize_url(url)}: {', '.join(missing)}")
    return normalize_url(url)


def run(url, kind, req, timeout=900):
    if url == LOCAL:
        return pool.run(KINDS[kind], req)
    try:
        return protocol.unpack(call(url, f"/remote_models/{kind}/run", protocol.pack(req), timeout=timeout))
    except RemoteError as e:
        raise RuntimeError(str(e)) from None


def register_probe_route():
    from server import PromptServer

    @PromptServer.instance.routes.post("/remote_models/probe")
    async def probe(request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or body.get("kind") not in KINDS:
                raise ValueError
        except ValueError:
            return web.json_response({"ok": False, "error": f"Expected a JSON object with a url and a kind ({', '.join(KINDS)})."}, status=400)

        def probe_run():
            if is_local(body.get("url")):
                import comfy.model_management as mm

                return {"ok": True, "device": mm.get_torch_device_name(mm.get_torch_device()), "version_note": None,
                        **KINDS[body["kind"]].options()}
            status, note, options = list_models(body.get("url"), body["kind"], timeout=10)
            return {"ok": True, "device": status.get("device"), "version_note": note, **options}

        try:
            result = await asyncio.get_running_loop().run_in_executor(None, probe_run)
        except RemoteError as e:
            result = {"ok": False, "error": str(e)}
        return web.json_response(result)
