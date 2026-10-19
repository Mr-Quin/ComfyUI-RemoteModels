from .remote_models import client, host
from .remote_models import clip, vae

NODES = {**clip.NODES, **vae.NODES}
NODE_CLASS_MAPPINGS = {name: cls for name, (cls, _) in NODES.items()}
NODE_DISPLAY_NAME_MAPPINGS = {name: title for name, (_, title) in NODES.items()}

host.register_routes()
client.register_probe_route()

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
