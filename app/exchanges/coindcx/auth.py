import hmac
import hashlib
import json
import time


class CoinDCXAuth:
    """CoinDCX HMAC-SHA256 authentication handler."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def sign(self, body: dict) -> str:
        """Generate HMAC-SHA256 signature for a request body."""
        secret_bytes = bytes(self.api_secret, encoding='utf-8')
        json_body = json.dumps(body, separators=(',', ':'))
        return hmac.new(
            secret_bytes, json_body.encode(), hashlib.sha256
        ).hexdigest()
    
    def get_auth_headers(self, body: dict) -> dict:
        """Get authenticated headers with signature."""
        return {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': self.sign(body),
        }
    
    @staticmethod
    def get_timestamp(in_seconds: bool = False) -> int:
        """Current timestamp in milliseconds (default) or seconds."""
        if in_seconds:
            return int(time.time())
        return int(round(time.time() * 1000))
    
    def prepare_body(self, body: dict, in_seconds: bool = False) -> tuple[str, dict]:
        """Add timestamp to body if not present, return (json_string, headers)."""
        if 'timestamp' not in body:
            body['timestamp'] = self.get_timestamp(in_seconds=in_seconds)
        json_body = json.dumps(body, separators=(',', ':'))
        headers = self.get_auth_headers(body)
        return json_body, headers

