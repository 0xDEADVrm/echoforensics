"""Minimal SHA-256 hash-chain demonstration for record-tamper detection."""
import hashlib
import json

def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def build_chain(records):
    previous = "0" * 64
    chain = []
    for index, record in enumerate(records):
        digest = hashlib.sha256(previous.encode("ascii") + _canonical(record)).hexdigest()
        chain.append({"index": index, "record": record, "previous_hash": previous, "hash": digest})
        previous = digest
    return chain

def verify_chain(chain):
    previous = "0" * 64
    for expected_index, item in enumerate(chain):
        if item["index"] != expected_index or item["previous_hash"] != previous:
            return False, expected_index
        expected = hashlib.sha256(previous.encode("ascii") + _canonical(item["record"])).hexdigest()
        if expected != item["hash"]:
            return False, expected_index
        previous = item["hash"]
    return True, None

def run_tamper_test():
    records = [
        {"event_id": 1, "label": "Empty", "confidence": 0.98},
        {"event_id": 2, "label": "Benign", "confidence": 0.95},
        {"event_id": 3, "label": "Intrusion", "confidence": 0.97},
        {"event_id": 4, "label": "Replay/Spoof", "confidence": 0.91},
        {"event_id": 5, "label": "Empty", "confidence": 0.99},
    ]
    original = build_chain(records)
    valid_before, index_before = verify_chain(original)
    tampered = json.loads(json.dumps(original))
    tampered[2]["record"]["confidence"] = 0.10
    valid_after, detected_index = verify_chain(tampered)
    return {
        "records": len(records),
        "valid_chain_passes": valid_before,
        "valid_chain_error_index": index_before,
        "tampered_chain_rejected": not valid_after,
        "first_detected_index_zero_based": detected_index,
        "note": "Demonstration only; no external timestamping or key management."
    }
