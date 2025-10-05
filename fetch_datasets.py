import io
import json
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import pandas as pd
import requests

# === Config ===
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
TIMEOUT = 90

# Vistas web (solo referencia)
VIEW_URLS = {
    "cumulative": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative",
    "TOI":        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI",
    "k2pandc":    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc",
}

# 1) API clásica (CSV directo)
NSTED_API = {
    "cumulative": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=*&format=csv",
    "TOI":        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&select=*&format=csv",
    "k2pandc":    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2pandc&select=*&format=csv",
}

# 2) TAP (ADQL)
TAP_API = {
    "cumulative": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv",
    "TOI":        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv",
    "k2pandc":    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv",
}

# Columnas esperadas (para validar que sea la tabla correcta)
EXPECTED_COLS: Dict[str, List[str]] = {
    "cumulative": ["koi_disposition", "koi_period", "koi_prad"],
    "TOI":        ["tfopwg_disp", "pl_orbper", "pl_rade"],
    "k2pandc":    ["disposition", "pl_orbper", "pl_rade"],
}

# ---------------- utilidades ----------------

def _sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()

def _save_bytes(name: str, data: bytes) -> Tuple[str, str]:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(DATA_DIR, f"{name}_{ts}.csv")
    with open(path, "wb") as f:
        f.write(data)
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
    headers = {
        "User-Agent": "ExoAI/1.0 (+https://render.com) Python-requests",
        "Accept": "text/csv, text/plain, */*",
        "Connection": "close",
    }
    r = requests.get(url, timeout=TIMEOUT, headers=headers, allow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    data = r.content
    if not data:
        raise ValueError("Respuesta vacía")
    return data, ctype

def _read_csv_bytes_loose(data: bytes) -> pd.DataFrame:
    """Lector tolerante (mismo que usamos tras descargar); sin low_memory, engine python, saltando líneas malas."""
    bio = io.BytesIO(data)
    return pd.read_csv(
        bio,
        comment="#",
        engine="python",
        on_bad_lines="skip",
    )

def _valid_shape(df: pd.DataFrame) -> bool:
    return (df.shape[0] >= 5) and (df.shape[1] >= 5)

def _has_expected_cols(df: pd.DataFrame, keys: List[str]) -> bool:
    cols = set(map(str, df.columns))
    return any(k in cols for k in keys)

# ---------------- descarga ----------------

def _fetch_one(key: str, view_url: str) -> Dict:
    attempts = []
    parse_method = None
    df: Optional[pd.DataFrame] = None
    data = b""
    content_type = ""

    # 1) nstedAPI (TOI a veces alterna nombre)
    nsted_candidates = (
        [
            "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&select=*&format=csv",
            "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=poi_toi&select=*&format=csv",
        ]
        if key == "TOI"
        else [NSTED_API[key]]
    )
    for api_url in nsted_candidates:
        attempts.append(api_url)
        try:
            data, content_type = _download(api_url)
            tmp = _read_csv_bytes_loose(data)
            if _valid_shape(tmp) and _has_expected_cols(tmp, EXPECTED_COLS[key]):
                df = tmp
                parse_method = "nstedAPI"
                break
        except Exception:
            continue

    # 2) TAP
    if df is None:
        api_url = TAP_API[key]
        attempts.append(api_url)
        try:
            data, content_type = _download(api_url)
            tmp = _read_csv_bytes_loose(data)
            if _valid_shape(tmp) and _has_expected_cols(tmp, EXPECTED_COLS[key]):
                df = tmp
                parse_method = "tap"
        except Exception:
            df = None

    # 3) HTML (último recurso)
    if df is None:
        attempts.append(view_url)
        try:
            tables = pd.read_html(view_url, header=0)
            if not tables:
                tables = pd.read_html(view_url)
            best_df = None
            best_score = -1
            for t in tables:
                t.columns = [str(c) for c in t.columns]
                score = int(t.shape[0]) * int(t.shape[1])
                if t.shape[1] <= 2:
                    continue
                if score > best_score:
                    best_score = score
                    best_df = t
            if best_df is not None and _valid_shape(best_df):
                df = best_df
                buf = io.StringIO()
                best_df.to_csv(buf, index=False)
                data = buf.getvalue().encode("utf-8")
                parse_method = "html"
        except Exception:
            pass

    if df is None:
        return {
            "df": pd.DataFrame(),
            "data": b"",
            "content_type": "",
            "parse_method": "error",
            "attempts": attempts + ["❌ No se obtuvo CSV válido"],
            "saved": False,
        }

    return {
        "df": df,
        "data": data,
        "content_type": content_type,
        "parse_method": parse_method,
        "attempts": attempts,
        "saved": True,
    }

def fetch_all(auto_save: bool = True) -> Dict:
    manifest = _load_manifest()
    result = {"datasets": {}, "change_detected": False}

    for key, view_url in VIEW_URLS.items():
        got = _fetch_one(key, view_url)
        df: pd.DataFrame = got["df"]
        data: bytes = got["data"]
        ctype: str = got["content_type"]
        parse_method: str = got["parse_method"]
        attempts: List[str] = got["attempts"]
        saved = got["saved"]

        meta = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(map(str, df.columns[:50])),
            "saved_path": None,
            "latest_path": None,
            "sha256": None,
            "source_view": view_url,
            "attempts": attempts,
            "content_type": ctype,
            "parse_method": parse_method,
            "updated_at_utc": datetime.utcnow().isoformat() + "Z",
            "changed": False,
        }

        if saved:
            sha = _sha256_bytes(data)
            prev_sha = manifest["datasets"].get(key, {}).get("sha256")
            saved_path, latest_path = _save_bytes(key, data)
            meta.update({
                "saved_path": saved_path,
                "latest_path": latest_path,
                "sha256": sha,
                "changed": (prev_sha is None) or (prev_sha != sha),
            })
            manifest["datasets"][key] = {"sha256": sha, "latest_path": latest_path}
            if meta["changed"]:
                result["change_detected"] = True

        result["datasets"][key] = meta

    if auto_save:
        _save_manifest(manifest)

    return result

# --------------- PREVIEW ---------------

def _read_csv_from_path_like_download(path: str) -> Optional[pd.DataFrame]:
    """
    Reproduce la lectura que funcionó al descargar (bytes -> _read_csv_bytes_loose).
    Si falla, intenta: sep=None (autodetección), dtype=str, varios encodings.
    Nunca lanza excepción hacia arriba.
    """
    # 1) Igual que la lectura desde descarga (bytes)
    try:
        with open(path, "rb") as f:
            data = f.read()
        df = _read_csv_bytes_loose(data)
        if df is not None and df.shape[1] > 1:
            return df
    except Exception:
        pass

    # 2) Fallbacks (autodetección de separador y encoding)
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(
                path,
                comment="#",
                engine="python",
                sep=None,       # autodetecta delimitador
                dtype=str,      # evita problemas de tipos
                on_bad_lines="skip",
                encoding=enc,
            )
            if df is not None and df.shape[1] > 1:
                return df
        except Exception:
            continue

    return None

def preview(n: int = 5) -> Dict:
    """
    Devuelve head(n) por dataset leyendo desde disco con estrategias robustas.
    Nunca lanza excepción; en caso de fallo devuelve {"error": "..."} por dataset.
    """
    out = {"datasets": {}}
    for key in VIEW_URLS.keys():
        latest_path = os.path.join(DATA_DIR, f"{key}.csv")
        if not os.path.exists(latest_path):
            out["datasets"][key] = {"error": "No existe archivo latest. Ejecuta /datasets primero."}
            continue

        df = _read_csv_from_path_like_download(latest_path)
        if df is None:
            out["datasets"][key] = {
                "error": f"No se pudo leer {latest_path} con los métodos de fallback."
            }
            continue

        # Serialización segura (evita NaN/np types no JSON-serializables)
        head_data = []
        for _, row in df.head(n).iterrows():
            safe_row = {}
            for col, val in row.items():
                try:
                    if isinstance(val, (float, int, str)) or val is None:
                        safe_row[col] = val
                    else:
                        safe_row[col] = str(val)
                except Exception:
                    safe_row[col] = None
            head_data.append(safe_row)

        out["datasets"][key] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(map(str, df.columns[:50])),
            "head": head_data,
            "latest_path": latest_path,
        }

    return out