import os
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

import requests

# Configuración
SPACE_DEVS_API_BASE = "https://ll.thespacedevs.com/2.0.0"
CACHE_DIR = os.getenv("DATA_DIR", "data")
CACHE_FILE = os.path.join(CACHE_DIR, "launches_cache.json")
CACHE_DURATION_MINUTES = 60  # Aumentado a 60 minutos para evitar rate limits

# Keywords para identificar misiones de exoplanetas
EXOPLANET_KEYWORDS = [
    'exoplanet', 'tess', 'jwst', 'james webb', 'kepler', 
    'cheops', 'plato', 'ariel', 'habitable', 'planet hunting', 'transiting'
]

# Rate limiting
_last_api_call = None
_min_seconds_between_calls = 3  # Mínimo 3 segundos entre llamadas

def _wait_for_rate_limit():
    """Espera el tiempo necesario para respetar rate limit"""
    global _last_api_call
    
    if _last_api_call is not None:
        elapsed = time.time() - _last_api_call
        if elapsed < _min_seconds_between_calls:
            wait_time = _min_seconds_between_calls - elapsed
            print(f"Esperando {wait_time:.1f}s para respetar rate limit...")
            time.sleep(wait_time)
    
    _last_api_call = time.time()

def _is_cache_valid() -> bool:
    """Verifica si el caché es válido (no expiró)"""
    if not os.path.exists(CACHE_FILE):
        return False
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cached_time = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
            expiry_time = cached_time + timedelta(minutes=CACHE_DURATION_MINUTES)
            is_valid = datetime.utcnow() < expiry_time
            
            if is_valid:
                remaining = (expiry_time - datetime.utcnow()).total_seconds() / 60
                print(f"Cache valido ({remaining:.1f} min restantes)")
            
            return is_valid
    except Exception:
        return False

def _load_from_cache() -> Optional[Dict]:
    """Carga datos desde el caché si es válido"""
    if not _is_cache_valid():
        return None
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Lanzamientos cargados desde cache")
            return data
    except Exception as e:
        print(f"Error leyendo cache: {e}")
        return None

def _save_to_cache(data: Dict) -> None:
    """Guarda datos en caché con timestamp"""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_data = {
            'cached_at': datetime.utcnow().isoformat(),
            'data': data
        }
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Lanzamientos guardados en cache (valido por {CACHE_DURATION_MINUTES} min)")
    except Exception as e:
        print(f"Error guardando cache: {e}")

def _is_exoplanet_mission(launch: Dict) -> bool:
    """Determina si una misión está relacionada con exoplanetas"""
    searchable_text = " ".join([
        launch.get('name', ''),
        launch.get('mission', {}).get('name', ''),
        launch.get('mission', {}).get('description', ''),
        launch.get('mission', {}).get('type', '')
    ]).lower()
    
    return any(keyword in searchable_text for keyword in EXOPLANET_KEYWORDS)

def _calculate_countdown(net_date: str) -> Dict[str, int]:
    """Calcula cuenta regresiva desde fecha ISO"""
    try:
        launch_date = datetime.fromisoformat(net_date.replace('Z', '+00:00'))
        now = datetime.now(launch_date.tzinfo)
        delta = launch_date - now
        
        if delta.total_seconds() < 0:
            return {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0, 'launched': True}
        
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        seconds = delta.seconds % 60
        
        return {
            'days': days,
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds,
            'launched': False,
            'total_seconds': int(delta.total_seconds())
        }
    except Exception:
        return {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0, 'launched': False}

def _process_launch(launch: Dict) -> Dict:
    """Procesa y enriquece un lanzamiento con metadata adicional"""
    mission = launch.get('mission', {}) or {}
    status = launch.get('status', {}) or {}
    provider = launch.get('launch_service_provider', {}) or {}
    pad = launch.get('pad', {}) or {}
    location = pad.get('location', {}) or {}
    
    # Status IDs: 1=Go, 2=TBD, 3=Success, 4=Failure, 6=In Flight, 7=Partial Failure
    status_id = status.get('id', 0)
    
    is_exoplanet = _is_exoplanet_mission(launch)
    net_date = launch.get('net', '')
    countdown = _calculate_countdown(net_date) if net_date else {}
    
    return {
        'id': launch.get('id'),
        'name': launch.get('name', 'Unknown Launch'),
        'net': net_date,
        'window_start': launch.get('window_start'),
        'window_end': launch.get('window_end'),
        
        # Status
        'status': {
            'id': status_id,
            'name': status.get('name', 'Unknown'),
            'abbrev': status.get('abbrev', ''),
            'is_go': status_id == 1,
            'is_success': status_id == 3,
            'is_failure': status_id in [4, 7],
            'in_flight': status_id == 6,
            'completed': status_id in [3, 4, 7]
        },
        
        # Mission details
        'mission': {
            'name': mission.get('name', ''),
            'description': mission.get('description', ''),
            'type': mission.get('type', 'N/A'),
            'orbit': mission.get('orbit', {}).get('name', 'N/A') if mission.get('orbit') else 'N/A'
        },
        
        # Provider
        'provider': {
            'name': provider.get('name', 'N/A'),
            'type': provider.get('type', ''),
            'country_code': provider.get('country_code', '')
        },
        
        # Location
        'location': {
            'name': location.get('name', 'N/A'),
            'country_code': location.get('country_code', ''),
            'pad_name': pad.get('name', 'N/A')
        },
        
        # Rocket
        'rocket': {
            'name': launch.get('rocket', {}).get('configuration', {}).get('name', 'N/A'),
            'family': launch.get('rocket', {}).get('configuration', {}).get('family', ''),
            'variant': launch.get('rocket', {}).get('configuration', {}).get('variant', '')
        },
        
        # Metadata
        'probability': launch.get('probability', -1),
        'hold_reason': launch.get('holdreason', ''),
        'fail_reason': launch.get('failreason', ''),
        'hashtag': launch.get('hashtag', ''),
        'image': launch.get('image', ''),
        'webcast_live': launch.get('webcast_live', False),
        
        # Exoplanet flag
        'is_exoplanet_mission': is_exoplanet,
        
        # Countdown (si está disponible)
        'countdown': countdown,
        
        # Links útiles
        'info_urls': [
            url.get('url') for url in launch.get('infoURLs', [])
        ] if launch.get('infoURLs') else [],
        'vid_urls': [
            url.get('url') for url in launch.get('vidURLs', [])
        ] if launch.get('vidURLs') else []
    }

def fetch_upcoming_launches(
    limit: int = 20,
    force_refresh: bool = False,
    prioritize_exoplanets: bool = True
) -> Dict:
    """
    Obtiene próximos lanzamientos de The Space Devs API con manejo de rate limits
    
    Args:
        limit: Número máximo de lanzamientos a obtener
        force_refresh: Ignorar caché y forzar nueva petición
        prioritize_exoplanets: Poner misiones de exoplanetas primero
    
    Returns:
        {
            "launches": [...],
            "exoplanet_count": 5,
            "total_count": 20,
            "cached": false,
            "cached_until": "2025-01-10T15:30:00Z"
        }
    """
    # Intentar cargar desde caché
    if not force_refresh:
        cached = _load_from_cache()
        if cached:
            data = cached['data']
            data['cached'] = True
            cache_age = (datetime.utcnow() - datetime.fromisoformat(cached['cached_at'])).total_seconds() / 60
            data['cache_age_minutes'] = round(cache_age, 1)
            return data
    
    # Fetch desde API con rate limiting
    try:
        print(f"Obteniendo lanzamientos desde Space Devs API (limit={limit})...")
        _wait_for_rate_limit()
        
        url = f"{SPACE_DEVS_API_BASE}/launch/upcoming/"
        params = {
            'limit': min(limit, 100),  # API tiene límite de 100
            'mode': 'detailed'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        print(f"Obtenidos {len(results)} lanzamientos")
        
        # Procesar cada lanzamiento
        processed = [_process_launch(launch) for launch in results]
        
        # Separar exoplanetas y otros
        if prioritize_exoplanets:
            exoplanet_launches = [l for l in processed if l['is_exoplanet_mission']]
            other_launches = [l for l in processed if not l['is_exoplanet_mission']]
            final_launches = exoplanet_launches + other_launches
        else:
            final_launches = processed
        
        # Construir respuesta
        result = {
            'launches': final_launches[:limit],
            'exoplanet_count': sum(1 for l in final_launches if l['is_exoplanet_mission']),
            'total_count': len(final_launches),
            'cached': False,
            'cache_age_minutes': 0,
            'cached_until': (datetime.utcnow() + timedelta(minutes=CACHE_DURATION_MINUTES)).isoformat() + 'Z',
            'fetched_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Guardar en caché
        _save_to_cache(result)
        
        return result
        
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            # Rate limit hit - intentar cargar del caché aunque esté expirado
            if os.path.exists(CACHE_FILE):
                print("Rate limit alcanzado, usando cache expirado...")
                try:
                    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        data = cached['data']
                        data['cached'] = True
                        data['cache_expired'] = True
                        data['note'] = "Datos del cache (expirados) debido a rate limit"
                        return data
                except Exception:
                    pass
            raise Exception(f"Rate limit alcanzado y no hay cache disponible. Espera {CACHE_DURATION_MINUTES} minutos.")
        raise Exception(f"Error HTTP {e.response.status_code}: {str(e)}")
    except requests.Timeout:
        raise Exception("Timeout al conectar con Space Devs API")
    except requests.RequestException as e:
        raise Exception(f"Error conectando con Space Devs API: {str(e)}")
    except Exception as e:
        raise Exception(f"Error procesando lanzamientos: {str(e)}")

def fetch_launch_by_id(launch_id: str) -> Optional[Dict]:
    """Obtiene detalles de un lanzamiento específico por ID"""
    try:
        _wait_for_rate_limit()
        url = f"{SPACE_DEVS_API_BASE}/launch/{launch_id}/"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        launch = response.json()
        return _process_launch(launch)
        
    except Exception as e:
        print(f"Error obteniendo lanzamiento {launch_id}: {e}")
        return None

def search_launches(
    search_query: str,
    limit: int = 10
) -> List[Dict]:
    """
    Busca lanzamientos por nombre/misión
    
    Args:
        search_query: Término de búsqueda
        limit: Número máximo de resultados
    """
    try:
        _wait_for_rate_limit()
        url = f"{SPACE_DEVS_API_BASE}/launch/"
        params = {
            'search': search_query,
            'limit': limit,
            'mode': 'detailed'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        return [_process_launch(launch) for launch in results]
        
    except Exception as e:
        print(f"Error buscando lanzamientos: {e}")
        return []

def get_exoplanet_missions_only(limit: int = 10) -> Dict:
    """Obtiene solo misiones relacionadas con exoplanetas"""
    all_launches = fetch_upcoming_launches(limit=50, prioritize_exoplanets=True)
    
    exoplanet_launches = [
        l for l in all_launches['launches'] 
        if l['is_exoplanet_mission']
    ][:limit]
    
    return {
        'launches': exoplanet_launches,
        'count': len(exoplanet_launches),
        'cached': all_launches.get('cached', False),
        'fetched_at': datetime.utcnow().isoformat() + 'Z'
    }

def clear_cache() -> bool:
    """Limpia el caché de lanzamientos"""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print("Cache de lanzamientos limpiado")
            return True
        return False
    except Exception as e:
        print(f"Error limpiando cache: {e}")
        return False

# Estadísticas útiles
def get_launch_statistics() -> Dict:
    """Obtiene estadísticas de los próximos lanzamientos"""
    try:
        data = fetch_upcoming_launches(limit=100)
        launches = data['launches']
        
        # Contar por status
        status_counts = {}
        for launch in launches:
            status = launch['status']['name']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Contar por proveedor
        provider_counts = {}
        for launch in launches:
            provider = launch['provider']['name']
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        # Top proveedores
        top_providers = sorted(
            provider_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'total_launches': len(launches),
            'exoplanet_missions': data['exoplanet_count'],
            'status_breakdown': status_counts,
            'top_providers': dict(top_providers),
            'cached': data['cached']
        }
        
    except Exception as e:
        return {'error': str(e)}