"""
Simple API key authentication.

Real production auth (OAuth2, JWT with user accounts, etc.) is overkill for a
portfolio project and would mostly demonstrate copy-pasting a tutorial. An API
key is what most internal or B2B financial APIs actually use for
service-to-service or analyst-tool access — so this matches a real pattern
without pretending to be a full identity system it isn't.
"""
import os
import secrets
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# In production, set API_KEY as a real secret via your deployment platform's
# environment variable / secrets manager — never commit a real key to git.
API_KEY = os.getenv("API_KEY", "dev-only-key-change-me")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(provided_key: str = Security(api_key_header)) -> str:
    if provided_key is None or not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Send it as the 'X-API-Key' header.",
        )
    return provided_key
