"""Download helpers.

- TDBRAIN via Kaggle: only a PRIVATE copy the user uploaded from their own licensed download.
  Credentials come from KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json and are never printed.
- Mumtaz via the figshare API: invoked only from src/eval/external.py (phase 8).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger(__name__)
CONFIRM_GB = 5.0


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def kaggle_download_tdbrain(slug: str, dest: Path, private_confirmed: bool, files: list[str] | None = None) -> None:
    if not slug or "/" not in slug:
        raise ValueError("tdbrain.kaggle_dataset must be '<owner>/<dataset-slug>'")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # reads env vars or ~/.kaggle/kaggle.json; never echo credentials
    meta = api.dataset_metadata if hasattr(api, "dataset_metadata") else None  # noqa: F841
    owner = slug.split("/")[0]
    me = api.get_config_value("username")
    if owner != me or not private_confirmed:
        raise PermissionError(
            f"Kaggle dataset '{slug}' is not confirmed as your own private upload (owner={owner}). "
            "A public third-party copy of TDBRAIN may breach the TDBRAIN data-use agreement. STOP: "
            "upload your own licensed copy privately and set tdbrain.kaggle_private_confirmed: true."
        )
    log.info("Free disk at %s: %.1f GB", dest, free_gb(dest))
    if files:
        for f in files:
            api.dataset_download_file(slug, f, path=str(dest))
    else:
        api.dataset_download_files(slug, path=str(dest), unzip=True)


def figshare_download(article_id: int, dest: Path, confirm_large: bool = False) -> list[Path]:
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    files, page = [], 1
    while True:  # the files endpoint is paginated (10 per page by default)
        r = requests.get(f"https://api.figshare.com/v2/articles/{article_id}/files",
                         params={"page": page, "page_size": 100}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        files += batch
        if len(batch) < 100:
            break
        page += 1
    total_gb = sum(f["size"] for f in files) / 1e9
    log.info("figshare %s: %d files, %.2f GB; free %.1f GB", article_id, len(files), total_gb, free_gb(dest))
    if total_gb > CONFIRM_GB and not confirm_large:
        raise RuntimeError(f"Download is {total_gb:.1f} GB > {CONFIRM_GB} GB; confirm disk space and path first.")
    out = []
    for f in files:
        target = dest / f["name"]
        if target.exists() and target.stat().st_size == f["size"]:
            out.append(target)
            continue
        with requests.get(f["download_url"], stream=True, timeout=300) as resp:
            if resp.status_code == 404:  # some files are listed but not served; logged, not fatal
                log.warning("figshare file not available (404): %s", f["name"])
                continue
            resp.raise_for_status()
            tmp = target.with_suffix(target.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.rename(target)
        out.append(target)
    return out


def main(argv=None) -> None:
    """`make data` step 1: in kaggle mode, download the user's PRIVATE TDBRAIN copy; in local mode, do nothing."""
    from src.data.tdbrain import resolve_root
    from src.utils.config import load_config

    cfg = load_config("data")["tdbrain"]
    if cfg["mode"] != "kaggle":
        log.info("tdbrain.mode=%s: nothing to download (data root %s)", cfg["mode"], resolve_root({"tdbrain": cfg}))
        return
    dest = resolve_root({"tdbrain": cfg})
    if free_gb(dest) < 20:
        raise SystemExit(f"Only {free_gb(dest):.1f} GB free at {dest}; free space or choose another path first.")
    kaggle_download_tdbrain(cfg["kaggle_dataset"], dest, bool(cfg.get("kaggle_private_confirmed")))


if __name__ == "__main__":
    main()
