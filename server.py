"""CycleGAN style transfer: photo -> Monet / ukiyo-e, geometry preserved.

Thin HTTP shim over upstream's own generator. RFD 0036: call the upstream entry point rather
than reimplementing it -- the ResNet-9block generator and its checkpoint layout are upstream's,
and a reimplementation would be a second definition that drifts from the first.

  POST /predict   image + style -> stylized image + the checkpoint hash that made it

## Why the checkpoint hash rides with every result

Output from this model is *generated* synthetic under the workspace data rule, and condition 1
requires the generating model and checkpoint recorded WITH the data so the corpus can be
regenerated later. A hash written once into a server log is not with the data: logs rotate,
and a row in a parquet file six months from now has to answer for itself. So every response
carries `checkpoint_sha256`, and the caller is expected to store it beside the image rather
than trust that both were produced by the same run.

## Stub mode

`WEFTSPUN_STUB=1` serves the contract without torch or weights, so the HTTP shape can be
exercised on a machine with no GPU. The stub returns a deterministic 1x1 PNG and a sentinel
hash -- it is deliberately not a plausible-looking image, because a stub that returns
something that resembles output is a stub somebody eventually ships.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

STUB = os.environ.get("WEFTSPUN_STUB") == "1"
CKPT_DIR = Path(os.environ.get("CYCLEGAN_CKPT_DIR", "/models"))

#: Upstream's own pretrained one-directional photo->painting generators. The names are
#: upstream's file names, not ours; renaming them would break the mapping to the source.
STYLES = {"monet": "style_monet.pth", "ukiyoe": "style_ukiyoe.pth"}

app = FastAPI(title="cyclegan_style_transfer")
_G: dict[str, object] = {}


class Req(BaseModel):
    image: str = Field(..., description="path, URL, or base64 PNG/JPEG")
    style: str = Field("monet", description="monet | ukiyoe")
    load_size: int = Field(256, ge=64, le=1024)


def _read_image(spec: str) -> bytes:
    if spec.startswith(("http://", "https://")):
        return urllib.request.urlopen(spec, timeout=60).read()
    p = Path(spec)
    if p.exists():
        return p.read_bytes()
    try:
        return base64.b64decode(spec, validate=True)
    except Exception as exc:
        raise HTTPException(400, f"image is not a path, URL or base64: {exc}") from exc


def _checkpoint_path(style: str) -> Path:
    if style not in STYLES:
        raise HTTPException(400, f"style must be one of {sorted(STYLES)}, got {style!r}")
    return CKPT_DIR / STYLES[style]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generator(style: str):
    """Upstream's ResNet-9block generator, loaded once per style and kept.

    Loading is deferred rather than done at import so the contract stage never needs torch,
    and so a request for a style whose checkpoint is missing fails loudly at that request
    instead of preventing the service from starting at all.
    """
    if style in _G:
        return _G[style]

    import torch
    from models.networks import define_G  # upstream

    ckpt = _checkpoint_path(style)
    if not ckpt.exists():
        raise HTTPException(503, f"checkpoint absent: {ckpt}. This is a FAILURE, not a "
                                 f"fallback -- serving an unstyled image would look like a "
                                 f"pass and silently poison the corpus.")
    net = define_G(3, 3, 64, "resnet_9blocks", norm="instance", use_dropout=False,
                   init_type="normal", init_gain=0.02, gpu_ids=[])
    state = torch.load(ckpt, map_location="cpu")
    # Upstream strips this key when saving from a DataParallel wrapper; tolerate both.
    state.pop("_metadata", None)
    net.load_state_dict(state)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    _G[style] = (net, _sha256(ckpt))
    return _G[style]


@app.get("/health")
def health():
    return {"ok": True, "stub": STUB, "styles": sorted(STYLES)}


@app.post("/predict")
def predict(req: Req):
    raw = _read_image(req.image)

    if STUB:
        # A 1x1 transparent PNG. Deliberately not image-shaped: a stub that returns something
        # plausible is a stub that eventually reaches production unnoticed.
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        return {"image": base64.b64encode(png).decode(), "style": req.style,
                "checkpoint_sha256": "STUB-NOT-A-REAL-CHECKPOINT", "stub": True}

    import numpy as np
    import torch
    from PIL import Image

    net, ckpt_hash = _generator(req.style)
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(
        (req.load_size, req.load_size), Image.BICUBIC)

    # Upstream's normalisation: [0,1] -> [-1,1]. Getting this wrong does not error, it
    # produces washed-out output that looks like a weak style rather than a bug.
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = (x.permute(2, 0, 1)[None] - 0.5) / 0.5
    if next(net.parameters()).is_cuda:
        x = x.cuda()

    with torch.no_grad():
        y = net(x)

    y = ((y[0].cpu().permute(1, 2, 0) * 0.5) + 0.5).clamp(0, 1).numpy()
    out = Image.fromarray((y * 255).round().astype("uint8"))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode(),
            "style": req.style, "checkpoint_sha256": ckpt_hash}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
