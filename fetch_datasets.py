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

def _candidate_csv_urls(view_url: str) -> list[str]:
    # Probamos varias variantes conocidas
    base_has_q = "?" in view_url
    joiner = "&" if base_has_q else "?"
    return [
        f"{view_url}{joiner}output=csv",
        f"{view_url}{joiner}format=csv",
    ]

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
    head = data[:200].lstrip()
    if head.startswith(b"<") or b"<html" in head.lower():
        return False
    return b"," in head and b"\n" in data[:1000]

def _read_as_csv_bytes(data: bytes) -> pd.DataFrame:
    """Lee CSV de forma tolerante."""
    bio = io.BytesIO(data)
    return pd.read_csv(bio, comment="#", engine="python", on_bad_lines="skip")

def _best_html_table(url: str) -> tuple[pd.DataFrame, dict]:
    """
    Lee TODAS las tablas HTML y elige la mejor:
    - mayor score = filas * columnas
    - descarta tablas con <= 2 columnas
    Devuelve (df, info) con métricas de selección.
    """
    # Intento 1: asumir encabezados en la primera fila
    tables = pd.read_html(url, header=0)  # requiere lxml
    if not tables:
        # Intento 2: sin header (pandas decide)
        tables = pd.read_html(url)
        if not tables:
            raise RuntimeError("No se encontraron tablas HTML en la página.")

    scored = []
    for idx, t in enumerate(tables):
        # Normaliza nombres a str
        t.columns = [str(c) for c in t.columns]
        n_rows, n_cols = t.shape
        # descartamos tablas triviales
        if n_cols <= 2:
            continue
        score = int(n_rows) * int(n_cols)
        scored.append((score, idx, n_rows, n_cols, t))

    if not scored:
        # Si todas eran triviales, como último recurso usa la primera
        t0 = tables[0]
        t0.columns = [str(c) for c in t0.columns]
        return t0, {"selected_index": 0, "selected_shape": t0.shape, "tables_found": len(tables)}

    scored.sort(reverse=True)  # mejor score primero
    best = scored[0]
    _, idx, n_rows, n_cols, df = best
    return df, {
        "selected_index": idx,
        "selected_shape": (int(n_rows), int(n_cols)),
        "tables_found": len(tables),
    }

def fetch_all(auto_save: bool = True) -> Dict:
    """
    Descarga los 3 datasets. Intenta CSV directo con varias variantes;
    si la respuesta no es CSV o falla el parser, hace fallback a parseo HTML
    escogiendo la tabla más grande (filas×columnas).
    Guarda:
      - data/<name>_YYYYMMDDThhmmssZ.csv
      - data/<name>.csv              (último)
      - data/manifest.json           (hash y metadata)
    Devuelve metadata con filas/columnas, parse_method y tabla elegida.
    """
    manifest = _load_manifest()
    result = {"datasets": {}, "change_detected": False}

    for key, view_url in VIEW_URLS.items():
        tried_csv_urls = _candidate_csv_urls(view_url)

        data = b""
        ctype = ""
        chosen_csv_url = None
        df = None
        parse_method = "csv"

        # 1) Intentar CSV con varias rutas
        for u in tried_csv_urls:
            try:
                _data, _ctype = _download(u)
                if _looks_like_csv(_data, _ctype):
                    # Validar que lo podamos leer
                    tmp_df = _read_as_csv_bytes(_data)
                    if tmp_df.shape[1] > 2 and tmp_df.shape[0] > 1:
                        data, ctype, df, chosen_csv_url = _data, _ctype, tmp_df, u
                        parse_method = "csv"
                        break
            except Exception:
                continue  # probamos la siguiente

        # 2) Fallback a HTML si no hubo CSV válido
        html_select_info = {}
        if df is None:
            parse_method = "html"
            df, html_select_info = _best_html_table(view_url)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            data = buf.getvalue().encode("utf-8")

        # 3) Guardar artefactos
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
            "csv_attempts": tried_csv_urls,
            "csv_used": chosen_csv_url,
            "content_type": ctype,
            "parse_method": parse_method,              # csv | html
            "html_selection": html_select_info,        # índice/shape elegidos
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