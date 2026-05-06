from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


PASSWORD_PREFIX_SALT = "EsP!rR"
PASSWORD_SUFFIX_SALT = "J$%aKkP[]"
DEFAULT_PASSWORD_HASH_PATH = Path(".streamlit/password.hash")


def hash_password(password: str) -> str:
    salted_password = f"{PASSWORD_PREFIX_SALT}{password}{PASSWORD_SUFFIX_SALT}"
    return hashlib.sha256(salted_password.encode("utf-8")).hexdigest()


def read_password_hash(path: str | Path = DEFAULT_PASSWORD_HASH_PATH) -> str | None:
    hash_path = Path(path)
    if not hash_path.exists():
        return None

    stored_hash = hash_path.read_text(encoding="utf-8").strip()
    return stored_hash or None


def write_password_hash(password: str, path: str | Path = DEFAULT_PASSWORD_HASH_PATH) -> Path:
    hash_path = Path(path)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(f"{hash_password(password)}\n", encoding="utf-8")
    return hash_path


def verify_password_from_file(password: str, path: str | Path = DEFAULT_PASSWORD_HASH_PATH) -> bool:
    stored_hash = read_password_hash(path)
    if not stored_hash:
        return False

    return hmac.compare_digest(hash_password(password), stored_hash)
