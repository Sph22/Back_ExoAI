import hashlib
import io
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import requests

# Carpeta donde guardaremos datasets y el manifest
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Vistas de tabla (páginas) que compartiste
VIEW_URLS = {
    "cumulative": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative",
    "TOI":        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI",
    "k2pandc":    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc",
}

def _candidate_csv_url(view_url: str) -> str:
    joiner = "&" if "?" in view_url else "?"
    return f"{view_url}{joiner}output=csv"

def _sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()

def _save_bytes(name: str, data: bytes) -> Tuple[str, str]:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(DATA_DIR, f"{name}_{ts}.csv")
    with open(path, "wb") as f:
        f.write(data)
    # copiado de “latest”
    latest = os.path.join(DATA_DIR, f"{name}.csv")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
    except FileNotFoundError:
        pass
    with open(latest, "wb") as f:
        f.write(data)
    return path, latest

def _load_manifest() -> Dict:
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"datasets": {}}

def _save_manifest(manifest: Dict) -> None:
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def _download_csv(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def _fallback_html_to_csv(url: str) -> bytes:
    # Si no hay CSV directo, intentamos parsear la tabla HTML principal
    tables = pd.read_html(url)
    if not tables:
        raise RuntimeError("No se encontraron tablas HTML en la página.")
    df = tables[0]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def fetch_all(auto_save: bool = True) -> Dict:
    """
    Descarga los 3 datasets. Intenta CSV directo; si falla, usa parseo HTML.
    Guarda:
      - data/<name>_YYYYMMDDThhmmssZ.csv
      - data/<name>.csv              (último)
      - data/manifest.json           (hash y metadata)
    """
    manifest = _load_manifest()
    result = {"datasets": {}, "change_detected": False}

    for key, view_url in VIEW_URLS.items():
        try:
            csv_url = _candidate_csv_url(view_url)
            data = _download_csv(csv_url)
        except Exception:
            data = _fallback_html_to_csv(view_url)

        sha = _sha256_bytes(data)
        prev_sha = manifest["datasets"].get(key, {}).get("sha256")

        saved_path, latest_path = _save_bytes(key, data)

        df = pd.read_csv(io.BytesIO(data), comment="#")
        meta = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns[:50]),
            "saved_path": saved_path,
            "latest_path": latest_path,
            "sha256": sha,
            "source_view": view_url,
            "csv_attempt": _candidate_csv_url(view_url),
            "updated_at_utc": datetime.utcnow().isoformat() + "Z",
            "changed": (prev_sha is None) or (prev_sha != sha),
        }
        result["datasets"][key] = meta

        manifest["datasets"][key] = {"sha256": sha, "latest_path": latest_path}

        if meta["changed"]:
            result["change_detected"] = True

    if auto_save:
        _save_manifest(manifest)

    return result

def preview(n: int = 5) -> Dict:
    """Devuelve head(n) de cada dataset y métricas básicas usando los archivos “latest”."""
    out = {"datasets": {}}
    for key in VIEW_URLS.keys():
        latest_path = os.path.join(DATA_DIR, f"{key}.csv")
        if not os.path.exists(latest_path):
            out["datasets"][key] = {"error": "No existe archivo latest. Ejecuta /datasets primero."}
            continue
        df = pd.read_csv(latest_path, comment="#")
        out["datasets"][key] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns[:50]),
            "head": df.head(n).to_dict(orient="records"),
            "latest_path": latest_path,
        }
    return out