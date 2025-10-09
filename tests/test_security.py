from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

from app.main import app, _reset_failed_downloads, RATE_LIMIT_MAX_FAILURES


def _client():
    _reset_failed_downloads()
    return TestClient(app)


def test_download_path_traversal_rejected(temp_env):
    with _client() as client:
        resp = client.get('/models/../../etc/passwd/download')
    assert resp.status_code in {400, 404}


def test_download_symlink_escape(temp_env):
    target_root = temp_env['data'].parent / 'outside'
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / 'model.pt').write_bytes(b'fake model')

    link_dir = temp_env['models'] / 'linkescape'
    try:
        os.symlink(target_root, link_dir, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip('symlinks not supported on this platform')

    try:
        with _client() as client:
            resp = client.get('/models/linkescape/download')
        assert resp.status_code == 404
    finally:
        if link_dir.exists() or link_dir.is_symlink():
            os.unlink(link_dir)


def test_download_rate_limit(temp_env):
    with _client() as client:
        for _ in range(RATE_LIMIT_MAX_FAILURES):
            resp = client.get('/models/nonexistent/download')
            assert resp.status_code == 404
        resp = client.get('/models/nonexistent/download')
    assert resp.status_code == 429


def test_download_prefers_safetensors(temp_env):
    model_dir = temp_env['models'] / 'model-alpha'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'model.safetensors').write_bytes(b'safe')
    (model_dir / 'model.pt').write_bytes(b'legacy')

    with _client() as client:
        resp = client.get('/models/model-alpha/download')
    assert resp.status_code == 200
    assert 'filename="model.safetensors"' in resp.headers.get('content-disposition', '')


def test_project_download_imports_fallback(temp_env):
    project_dir = temp_env['projects'] / 'proj-sf'
    models_dir = project_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    imports_dir = project_dir / 'imports' / 'model-b'
    imports_dir.mkdir(parents=True, exist_ok=True)
    (imports_dir / 'model.safetensors').write_bytes(b'safe-import')

    with _client() as client:
        resp = client.get('/projects/proj-sf/models/model-b/download')
    assert resp.status_code == 200
    assert 'filename="model.safetensors"' in resp.headers.get('content-disposition', '')
