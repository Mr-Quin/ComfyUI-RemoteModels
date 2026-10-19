import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ID = {
  serve: "RemoteModels.Serve",
  idle: "RemoteModels.IdleMinutes",
  url: "RemoteModels.DefaultUrl",
  reload: "RemoteModels.ReloadRecent",
};
const PLACEHOLDER = "(connect to list)";

// Each combo lists [widget, field from the probe response, values that are never "missing"].
const LOADERS = {
  RemoteCLIPLoader: {
    kind: "clip",
    noun: "encoders",
    combos: [
      ["clip_name", (r) => r.files, [PLACEHOLDER]],
      ["clip_name2", (r) => ["none", ...r.files], ["none"]],
      ["type", (r) => r.types, []],
    ],
  },
  RemoteVAELoader: {
    kind: "vae",
    noun: "VAEs",
    combos: [["vae_name", (r) => r.files, [PLACEHOLDER]]],
  },
};

function setting(id, fallback) {
  try {
    const v = app.extensionManager.setting.get(id);
    return v === undefined || v === null ? fallback : v;
  } catch {
    return fallback;
  }
}

function setupLoader(node, spec) {
  const w = (name) => node.widgets.find((x) => x.name === name);
  const urlW = w("server_url");
  if (!urlW.value) urlW.value = setting(ID.url, "");

  const button = node.addWidget("button", "Connect", null, () => connect());
  button.serialize = false;
  const show = (text) => {
    const label = text.length > 34 ? text.slice(0, 33) + "…" : text;
    button.name = label;
    button.label = label;
    node.setDirtyCanvas(true, true);
  };

  let seq = 0;
  async function connect() {
    const mine = ++seq;
    const url = urlW.value.trim();
    if (!url) {
      show("Enter a server URL");
      return;
    }
    show("Connecting…");
    let res;
    try {
      const r = await api.fetchApi("/remote_models/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, kind: spec.kind }),
      });
      res = await r.json();
    } catch (e) {
      res = { ok: false, error: String(e) };
    }
    if (mine !== seq) return;
    if (!res.ok) {
      console.warn("[RemoteModels]", res.error);
      const e = String(res.error);
      if (/is this ComfyUI/.test(e)) {
        show("This machine · use local");
        return;
      }
      const why = /cannot reach|timed out/i.test(e) ? "Unreachable"
        : /disabled/i.test(e) ? "Server not serving"
        : /not have the Remote Models|does not serve/i.test(e) ? "Pack missing or old on server"
        : /protocol/i.test(e) ? "Version mismatch, update both"
        : "Error";
      show(`${why} · click to retry`);
      return;
    }
    const missing = [];
    for (const [name, values, ignore] of spec.combos) {
      const widget = w(name);
      const list = values(res);
      const saved = widget.value;
      const absent = saved && !ignore.includes(saved) && !list.includes(saved);
      widget.options.values = absent ? [saved, ...list] : list;
      if (!widget.options.values.includes(widget.value)) widget.value = list[0] ?? "";
      if (absent) missing.push(saved);
    }
    if (missing.length) {
      show(`Not on server: ${missing.join(", ")}`);
      return;
    }
    if (res.version_note) {
      show(`Connected · ${res.version_note}`);
      return;
    }
    const count = `${res.files.length} ${spec.noun}`;
    show(url.toLowerCase() === "local" ? `Local · ${count}` : `Connected · ${count}`);
  }

  // The frontend reports a saved model name as missing when it is not in the
  // combo options at load time, and ours only arrive after connecting.
  const onConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const r = onConfigure?.apply(this, args);
    if (!urlW.value) urlW.value = setting(ID.url, "");
    for (const [name] of spec.combos) {
      const x = w(name);
      if (x?.value && !x.options.values.includes(x.value)) x.options.values = [x.value, ...x.options.values];
    }
    return r;
  };

  let timer;
  const original = urlW.callback;
  urlW.callback = function (...args) {
    original?.apply(this, args);
    clearTimeout(timer);
    timer = setTimeout(connect, 600);
  };
  setTimeout(connect, 400);
}

app.registerExtension({
  name: "RemoteModels",
  settings: [
    {
      id: ID.serve,
      name: "Serve models to other machines",
      type: "boolean",
      defaultValue: false,
      category: ["Remote Models", "Server", "Serve"],
      tooltip: "Lets other ComfyUI instances run text encoders and VAEs on this machine.",
    },
    {
      id: ID.idle,
      name: "Unload idle models after (minutes)",
      type: "number",
      defaultValue: 10,
      attrs: { min: 0, step: 1 },
      category: ["Remote Models", "Server", "Idle"],
      tooltip: "0 keeps loaded models forever.",
    },
    {
      id: ID.reload,
      name: "Reload recently used models on startup",
      type: "boolean",
      defaultValue: true,
      category: ["Remote Models", "Server", "Reload"],
      tooltip: "Loads the last models clients used again after ComfyUI restarts, so the first request is fast. They still unload after the idle time.",
    },
    {
      id: ID.url,
      name: "Default server URL",
      type: "text",
      defaultValue: "",
      category: ["Remote Models", "Client", "Url"],
      tooltip: "Prefilled into new Load Remote CLIP and Load Remote VAE nodes, e.g. http://192.168.1.50:8188",
    },
  ],
  nodeCreated(node) {
    const spec = LOADERS[node.comfyClass];
    if (spec) setupLoader(node, spec);
  },
});
