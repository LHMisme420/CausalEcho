import hashlib
import json

def seal_forensic_record(data: dict, prev_hash: str = "") -> str:
    payload = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(payload + prev_hash.encode()).hexdigest()
