import json
import mimetypes
import os
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib import error, parse, request


def _env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _clean_url(url: str) -> str:
    return url.rstrip("/")


def storage_config() -> Dict[str, Optional[str]]:
    base_url = _clean_url(
        _env("SUPABASE_URL", "BATTINGIQ_SUPABASE_URL", "VITE_SUPABASE_URL")
    )
    service_key = _env(
        "SUPABASE_SERVICE_ROLE_KEY",
        "BATTINGIQ_SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_ANON_KEY",
        "VITE_SUPABASE_PUBLISHABLE_KEY",
    )
    bucket = _env("SUPABASE_STORAGE_BUCKET", "BATTINGIQ_SUPABASE_STORAGE_BUCKET") or "battingiq-uploads"
    prefix = _env("SUPABASE_STORAGE_PREFIX", "BATTINGIQ_SUPABASE_STORAGE_PREFIX") or "analysis-media"
    return {
        "base_url": base_url or None,
        "service_key": service_key or None,
        "bucket": bucket,
        "prefix": prefix.strip("/"),
        "enabled": bool(base_url and service_key),
    }


def storage_enabled() -> bool:
    return bool(storage_config()["enabled"])


def _storage_headers(content_type: Optional[str] = None) -> Dict[str, str]:
    cfg = storage_config()
    key = cfg["service_key"] or ""
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def storage_object_path(relative_path: str) -> str:
    cfg = storage_config()
    rel = relative_path.strip("/")
    prefix = cfg["prefix"] or ""
    return f"{prefix}/{rel}" if prefix else rel


def _object_url(relative_path: str) -> str:
    cfg = storage_config()
    object_path = parse.quote(storage_object_path(relative_path), safe="/")
    return f"{cfg['base_url']}/storage/v1/object/{cfg['bucket']}/{object_path}"


def upload_file(local_path: Path, relative_path: str) -> Dict[str, Optional[str]]:
    cfg = storage_config()
    if not cfg["enabled"]:
        return {"status": "disabled", "object_path": storage_object_path(relative_path), "error": None}
    if not local_path.exists() or not local_path.is_file():
        return {"status": "skipped", "object_path": storage_object_path(relative_path), "error": "missing_local_file"}

    content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    req = request.Request(
        _object_url(relative_path),
        data=local_path.read_bytes(),
        method="POST",
        headers={
            **_storage_headers(content_type),
            "x-upsert": "true",
            "cache-control": "3600",
        },
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return {
                "status": "ok",
                "object_path": storage_object_path(relative_path),
                "http_status": str(getattr(resp, "status", 200)),
                "response": body[:500] or None,
                "error": None,
            }
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return {
            "status": "failed",
            "object_path": storage_object_path(relative_path),
            "http_status": str(exc.code),
            "response": detail[:500] or None,
            "error": f"http_{exc.code}",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "object_path": storage_object_path(relative_path),
            "http_status": None,
            "response": None,
            "error": str(exc),
        }


def upload_tree(root_dir: Path, relative_to: Path) -> Dict[str, object]:
    cfg = storage_config()
    summary = {
        "enabled": bool(cfg["enabled"]),
        "bucket": cfg["bucket"],
        "prefix": cfg["prefix"],
        "uploaded": [],
        "failed": [],
        "skipped": [],
    }
    if not root_dir.exists():
        summary["error"] = f"missing_root:{root_dir}"
        return summary
    if not cfg["enabled"]:
        summary["status"] = "disabled"
        return summary

    for file_path in sorted(p for p in root_dir.rglob("*") if p.is_file()):
        rel = file_path.relative_to(relative_to).as_posix()
        result = upload_file(file_path, rel)
        status = result.get("status")
        if status == "ok":
            summary["uploaded"].append(result)
        elif status in {"disabled", "skipped"}:
            summary["skipped"].append(result)
        else:
            summary["failed"].append(result)

    summary["status"] = "ok" if not summary["failed"] else "partial_failure"
    summary["uploaded_count"] = len(summary["uploaded"])
    summary["failed_count"] = len(summary["failed"])
    summary["skipped_count"] = len(summary["skipped"])
    return summary


def get_signed_result_url(relative_path: str, expires_in: int = 3600) -> Optional[str]:
    cfg = storage_config()
    if not cfg["enabled"]:
        return None

    object_path = parse.quote(storage_object_path(relative_path), safe="/")
    url = f"{cfg['base_url']}/storage/v1/object/sign/{cfg['bucket']}/{object_path}"
    payload = json.dumps({"expiresIn": expires_in}).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        method="POST",
        headers=_storage_headers("application/json"),
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    signed = (
        data.get("signedURL")
        or data.get("signedUrl")
        or data.get("url")
    )
    if not signed:
        return None
    if signed.startswith("http://") or signed.startswith("https://"):
        return signed
    if signed.startswith("/storage/"):
        return f"{cfg['base_url']}{signed}"
    if signed.startswith("/object/"):
        return f"{cfg['base_url']}/storage/v1{signed}"
    return f"{cfg['base_url']}/storage/v1/{signed.lstrip('/')}"


def result_redirect_url(job_id: str, file_path: str, expires_in: int = 3600) -> Optional[str]:
    relative_path = Path(job_id) / file_path
    return get_signed_result_url(relative_path.as_posix(), expires_in=expires_in)
