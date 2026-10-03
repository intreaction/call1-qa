"""Curated local packs and authenticated Pro1 discovery; no automatic downloads."""
import json
import os
from pathlib import Path
import sys
import urllib.request

LOCAL_PACKS = {
    'gemma4-e4b': {'name': 'Call1 Optimized · 4B', 'model': 'gemma-4-e4b-it-4bit',
                   'directory': 'gemma-4-e4b-it', 'memory': 'Approximately 5–6 GB QA memory', 'supported': True},
    'gemma4-12b': {'name': 'Call1 Optimized · 12B', 'model': 'gemma-4-12B-it-4bit',
                   'directory': 'gemma-4-12b-it', 'memory': 'Higher memory budget; qualification pending', 'supported': False},
}


def local_pack_path(pack_id):
    pack = LOCAL_PACKS[pack_id]
    default = (Path.home() / 'Library/Application Support/Call1/models' if getattr(sys, 'frozen', False)
               else Path('data/models'))
    return Path(os.getenv('CALL1_OPTIONAL_MODELS_DIR', str(default))) / pack['directory']


def local_catalog():
    result = []
    for pack_id, pack in LOCAL_PACKS.items():
        path = local_pack_path(pack_id)
        installed = (path / 'config.json').is_file() and any(path.glob('*.safetensors'))
        result.append(dict(pack, id=pack_id, installed=installed,
                           available=installed and pack['supported']))
    return result


def resolve_local_pack(pack_id):
    pack = next((p for p in local_catalog() if p['id'] == pack_id), None)
    if not pack or not pack['available']:
        raise RuntimeError('This local model pack is not installed or supported by this app version')
    return str(local_pack_path(pack_id).resolve())


def pro1_catalog(settings):
    from call1.models.schemas import QuestionModel
    from call1.question_models import _OPENER
    existing = next((m for m in settings.question_models.models if m.source == 'pro1'), None)
    endpoint = os.getenv('CALL1_PRO1_ENDPOINT') or (existing.endpoint if existing else None) or 'https://call1.cc'
    key_env = (existing.api_key_env if existing else None) or 'CALL1_MODEL_KEY_PRO1'
    key = os.getenv(key_env)
    base = dict(models=[], endpoint=endpoint, api_key_env=key_env, key_loaded=bool(key))
    if not key:
        return dict(base, status='needs_key', message='Load your Pro1 API key on this appliance to browse premium models.')
    if not endpoint:
        return dict(base, status='needs_endpoint', message='Configure the Pro1 API connection to load its model catalog.')
    # Reuse the same endpoint restrictions as generation. No arbitrary redirects.
    model = QuestionModel(id='pro1-catalog', name='Pro1', source='pro1', model='catalog', endpoint=endpoint, api_key_env=key_env)
    req = urllib.request.Request(model.endpoint + '/models', headers={'Authorization': 'Bearer ' + key})
    try:
        with _OPENER.open(req, timeout=10) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError('Catalog too large')
        data = json.loads(payload)
        ids = sorted({item['id'] for item in data['data']
                      if isinstance(item, dict) and isinstance(item.get('id'), str) and 0 < len(item['id']) <= 200})
        return dict(base, status='ready', models=ids, message='' if ids else 'No models were returned by Pro1.')
    except Exception:
        return dict(base, status='error', message='Could not load Pro1 models. Check the connection and API key.')
