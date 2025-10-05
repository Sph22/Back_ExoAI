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
    # Copia como “latest”
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

def _download(url: str) -> Tuple[bytes, str]:
    """Descarga y regresa (contenido, content_type)."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").lower()
    return r.content, ctype

def _looks_like_csv(data: bytes, content_type: str) -> bool:
    """Heurística para decidir si los bytes representan CSV."""
    if "text/csv" in content_type or "application/csv" in content_type:
        return True
    # Si empieza con '<' probablemente es HTML
    head = data[:200].lstrip()
    if head.startswith(b"<") or b"<html" in head.lower():
        return False
    # Si tiene comas/line breaks típicos
    # (muy laxa, solo como apoyo)
    return b"," in head and b"\n" in data[:1000]

def _read_as_csv_bytes(data: bytes) -> pd.DataFrame:
    """Lee CSV de forma tolerante."""
    bio = io.BytesIO(data)
    # comment='#' por headers descriptivos de NASA; engine='python' + on_bad_lines para robustez
    return pd.read_csv(bio, comment="#", engine="python", on_bad_lines="skip")

def _read_as_html(url: str) -> pd.DataFrame:
    """Parsea la primera tabla HTML encontrada."""
    tables = pd.read_html(url)  # requiere lxml
    if not tables:
        raise RuntimeError("No se encontraron tablas HTML en la página.")
    return tables[0]

def fetch_all(auto_save: bool = True) -> Dict:
    """
    Descarga los 3 datasets. Intenta CSV directo (output=csv);
    si la respuesta no es CSV o falla el parser, hace fallback a parseo HTML.
    Guarda:
      - data/<name>_YYYYMMDDThhmmssZ.csv
      - data/<name>.csv              (último)
      - data/manifest.json           (hash y metadata)
    Devuelve metadata con filas/columnas y parse_method usado.
    """
    manifest = _load_manifest()
    result = {"datasets": {}, "change_detected": False}

    for key, view_url in VIEW_URLS.items():
        csv_url = _candidate_csv_url(view_url)

        # 1) Descargar intentando CSV
        try:
            data, ctype = _download(csv_url)
        except Exception:
            # Si la descarga falla, forzamos parseo HTML
            data, ctype = b"", ""

        parse_method = "csv"
        df = None

        # 2) Verificar si realmente es CSV; si no, intentar HTML
        if not data or not _looks_like_csv(data, ctype):
            parse_method = "html"
            df = _read_as_html(view_url)
            # Convertimos df a bytes CSV para guardar artefactos “latest”
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            data = buf.getvalue().encode("utf-8")
        else:
            # Sí parece CSV → intentar leerlo con tolerancia; si falla, fallback a HTML
            try:
                df = _read_as_csv_bytes(data)
            except Exception:
                parse_method = "html"
                df = _read_as_html(view_url)
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                data = buf.getvalue().encode("utf-8")

        # 3) Guardar archivos y actualizar manifest
        sha = _sha256_bytes(data)
        prev_sha = manifest["datasets"].get(key, {}).get("sha256")

        saved_path, latest_path = _save_bytes(key, data)

        meta = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns[:50]),
            "saved_path": saved_path,
            "latest_path": latest_path,
            "sha256": sha,
            "source_view": view_url,
            "csv_attempt": csv_url,
            "content_type": ctype,
            "parse_method": parse_method,  # <-- importante para depurar
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
        df = pd.read_csv(latest_path, comment="#", engine="python", on_bad_lines="skip")
        out["datasets"][key] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns[:50]),
            "head": df.head(n).to_dict(orient="records"),
            "latest_path": latest_path,
        }
    return out