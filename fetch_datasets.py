import io
import json
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from functools import lru_cache

import math
import numpy as np
import pandas as pd
import requests

# === Config ===
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
TIMEOUT = 30  # Reducido de 90 a 30 segundos

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

# Columnas "señuelo" para validar
EXPECTED_COLS: Dict[str, List[str]] = {
    "cumulative": ["koi_disposition", "koi_period", "koi_prad"],
    "TOI":        ["tfopwg_disp", "pl_orbper", "pl_rade"],
    "k2pandc":    ["disposition", "pl_orbper", "pl_rade"],
}


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
    # Usar sesión para reutilizar conexiones
    with requests.Session() as session:
        r = session.get(url, timeout=TIMEOUT, headers=headers, allow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        data = r.content
        if not data:
            raise ValueError("Respuesta vacía")
        return data, ctype


def _read_csv_bytes_loose(data: bytes) -> pd.DataFrame:
    """Lector tolerante (engine='python', sin low_memory, saltando líneas malas)."""
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


def _fetch_one(key: str, view_url: str) -> Dict:
    attempts = []
    parse_method = None
    df: Optional[pd.DataFrame] = None
    data = b""
    content_type = ""

    # 1) nstedAPI (TOI puede alternar nombre)
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

    # 3) HTML (último recurso) - REMOVIDO para optimizar tiempo

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


# ---------- helpers JSON-safe ----------
def _to_json_safe(v):
    """Convierte cualquier escalar a un valor JSON-seguro (NaN/Inf -> None, numpy -> python)."""
    try:
        if v is None:
            return None
        if isinstance(v, (np.generic,)):
            v = v.item()
        if isinstance(v, float):
            if not math.isfinite(v):
                return None
            return float(v)
        if isinstance(v, (str, int, bool)):
            return v
        if pd.isna(v):
            return None
    except Exception:
        return None
    return v if isinstance(v, (str, int, bool, float)) else str(v)


@lru_cache(maxsize=3)
def _cached_read_csv(path: str, mtime: float) -> Optional[pd.DataFrame]:
    """
    Lee CSV con caché basado en tiempo de modificación.
    El parámetro mtime fuerza invalidación de caché cuando el archivo cambia.
    """
    return _read_csv_from_path_like_download(path)


def _read_csv_from_path_like_download(path: str) -> Optional[pd.DataFrame]:
    """Reproduce lectura desde bytes; si falla, autodetecta separador/encoding."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        df = _read_csv_bytes_loose(data)
        if df is not None and df.shape[1] > 1:
            return df
    except Exception:
        pass

    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(
                path,
                comment="#",
                engine="python",
                sep=None,
                dtype=str,
                on_bad_lines="skip",
                encoding=enc,
            )
            if df is not None and df.shape[1] > 1:
                return df
        except Exception:
            continue
    return None


def preview(n: int = 5) -> Dict:
    """Devuelve head(n) por dataset leyendo desde disco con caché (sin NaN/Inf en el JSON)."""
    out = {"datasets": {}}
    for key in VIEW_URLS.keys():
        latest_path = os.path.join(DATA_DIR, f"{key}.csv")
        if not os.path.exists(latest_path):
            out["datasets"][key] = {"error": "No existe archivo latest. Ejecuta /datasets primero."}
            continue

        # Usar caché basado en tiempo de modificación
        try:
            mtime = os.path.getmtime(latest_path)
            df = _cached_read_csv(latest_path, mtime)
        except Exception:
            df = None

        if df is None:
            out["datasets"][key] = {
                "error": f"No se pudo leer {latest_path} con los métodos de fallback."
            }
            continue

        # Construir head JSON-seguro
        head_rows: List[Dict] = []
        for _, row in df.head(n).iterrows():
            safe_row = {}
            for col, val in row.items():
                safe_row[str(col)] = _to_json_safe(val)
            head_rows.append(safe_row)

        out["datasets"][key] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": [str(c) for c in df.columns[:50]],
            "head": head_rows,
            "latest_path": latest_path,
        }
    return out


def get_first_n_with_names(n: int = 3) -> Dict:
    """
    Obtiene los primeros N registros de cumulative con nombre del exoplaneta.
    Devuelve datos completos + nombre si está disponible.
    """
    cumulative_path = os.path.join(DATA_DIR, "cumulative.csv")
    if not os.path.exists(cumulative_path):
        return {"error": "No existe cumulative.csv. Ejecuta /datasets primero."}
    
    try:
        mtime = os.path.getmtime(cumulative_path)
        df = _cached_read_csv(cumulative_path, mtime)
    except Exception:
        df = None
    
    if df is None:
        return {"error": "No se pudo leer cumulative.csv"}
    
    # Tomar primeros N registros
    subset = df.head(n)
    
    # Columnas de interés para features + nombre
    feature_cols = ["koi_period", "koi_duration", "koi_depth", "koi_prad", 
                    "koi_insol", "koi_steff", "koi_srad", "koi_disposition"]
    
    # Buscar columna de nombre (puede ser 'kepler_name', 'kepoi_name', etc.)
    name_col = None
    for possible_name in ["kepler_name", "kepoi_name", "koi_name", "pl_name"]:
        if possible_name in df.columns:
            name_col = possible_name
            break
    
    results = []
    for idx, row in subset.iterrows():
        record = {}
        
        # Agregar features
        for col in feature_cols:
            if col in df.columns:
                record[col] = _to_json_safe(row[col])
        
        # Agregar nombre si existe
        if name_col:
            record["exoplanet_name"] = _to_json_safe(row[name_col])
        else:
            record["exoplanet_name"] = None
        
        # Agregar ID del registro
        record["row_index"] = int(idx)
        
        results.append(record)
    
    return {
        "count": len(results),
        "records": results,
        "name_column_used": name_col,
        "dataset": "cumulative"
    }