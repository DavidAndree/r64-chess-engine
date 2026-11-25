"""Download the pretrained human-like base model, so you can play without training first."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

from r64 import config

RELEASE_TAG = "model-base-v1"
URL = f"https://github.com/DavidAndree/r64-chess-engine/releases/download/{RELEASE_TAG}/base.pt"
SHA256 = "860fab9937fedd0350fd2e74690969a9325a21776fd6908c14b3fe4ca36b64df"


def download_base(dest: Path = config.BOT_CHECKPOINT, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"{dest} already exists (use --force to replace it)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    with (
        tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp,
        urllib.request.urlopen(URL) as resp,
    ):
        shutil.copyfileobj(resp, tmp)
    tmp_path = Path(tmp.name)
    digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    if digest != SHA256:
        tmp_path.unlink()
        raise SystemExit(f"checksum mismatch: expected {SHA256}, got {digest}")
    tmp_path.replace(dest)
    print(f"saved {dest}")
    return dest
