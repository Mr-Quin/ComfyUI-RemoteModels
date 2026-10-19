import json
import os

import folder_paths

SERVE = "RemoteModels.Serve"
IDLE_MINUTES = "RemoteModels.IdleMinutes"
RELOAD_RECENT = "RemoteModels.ReloadRecent"
DEFAULT_URL = "RemoteModels.DefaultUrl"

_cache = {"mtime": None, "data": {}}


def _path():
    return os.path.join(folder_paths.get_user_directory(), "default", "comfy.settings.json")


def _load():
    path = _path()
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return {}
    if mtime != _cache["mtime"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return _cache["data"]
        _cache["mtime"] = mtime
        _cache["data"] = data if isinstance(data, dict) else {}
    return _cache["data"]


def get(key, default=None):
    value = _load().get(key, default)
    return default if value is None else value


def serving_enabled():
    return bool(get(SERVE, False))


def reload_recent():
    return bool(get(RELOAD_RECENT, True))


def idle_minutes():
    try:
        return max(0.0, float(get(IDLE_MINUTES, 10)))
    except (TypeError, ValueError):
        return 10.0
