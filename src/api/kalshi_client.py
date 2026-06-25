"""
Kalshi API Client

Handles authentication and communication with the Kalshi trading API.
Uses RSA-PSS signing for request authentication.
"""

import base64
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    """Client for interacting with the Kalshi API."""

    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        base_url: str = "https://trading-api.kalshi.com/trade-api/v2",
    ):
        """
        Initialize the Kalshi API client.

        Args:
            api_key: The API Key ID from Kalshi
            private_key_path: Path to the RSA private key file
            base_url: Base URL for the Kalshi API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.private_key = self._load_private_key(private_key_path)
        self.session = requests.Session()

    def _load_private_key(self, key_path: str):
        """Load RSA private key from file."""
        path = Path(key_path).expanduser()
        with open(path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
            )
        return private_key

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)

    def _sign_request(self, timestamp: int, method: str, path: str) -> str:
        """
        Sign a request using RSA-PSS with SHA256.

        Args:
            timestamp: Request timestamp in milliseconds
            method: HTTP method (GET, POST, etc.)
            path: Request path without query parameters

        Returns:
            Base64-encoded signature
        """
        # Create message to sign: timestamp + method + path
        message = f"{timestamp}{method}{path}".encode("utf-8")

        # Sign using RSA-PSS
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        return base64.b64encode(signature).decode("utf-8")

    def _get_headers(self, method: str, path: str) -> dict:
        """
        Generate authenticated headers for a request.

        Args:
            method: HTTP method
            path: Request path (without query parameters)

        Returns:
            Dictionary of headers
        """
        timestamp = self._get_timestamp()
        signature = self._sign_request(timestamp, method, path)

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
    ) -> dict:
        """
        Make an authenticated request to the Kalshi API.

        Args:
            method: HTTP method
            endpoint: API endpoint (e.g., "/exchange/status")
            params: Query parameters
            json_data: JSON body for POST/PUT requests

        Returns:
            Response JSON data
        """
        url = f"{self.base_url}{endpoint}"

        # Parse full path for signing (without query params)
        parsed = urlparse(url)
        path = parsed.path.split("?")[0]  # Remove any query params

        headers = self._get_headers(method, path)

        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
        )

        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make a GET request."""
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, json_data: Optional[dict] = None) -> dict:
        """Make a POST request."""
        return self._request("POST", endpoint, json_data=json_data)

    # -------------------------------------------------------------------------
    # API Methods
    # -------------------------------------------------------------------------

    def get_exchange_status(self) -> dict:
        """Get the current exchange status."""
        return self.get("/exchange/status")

    def get_balance(self) -> dict:
        """Get account balance."""
        return self.get("/portfolio/balance")

    def get_positions(self) -> dict:
        """Get current positions."""
        return self.get("/portfolio/positions")

    def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """
        Get available markets.

        Args:
            limit: Maximum number of markets to return
            cursor: Pagination cursor
            event_ticker: Filter by event ticker
            series_ticker: Filter by series ticker
            status: Filter by status (e.g., "open", "closed")

        Returns:
            Dictionary with markets and pagination info
        """
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status

        return self.get("/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """Get details for a specific market."""
        return self.get(f"/markets/{ticker}")

    def get_events(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
    ) -> dict:
        """Get available events."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker

        return self.get("/events", params=params)

    def get_event(self, event_ticker: str) -> dict:
        """Get details for a specific event."""
        return self.get(f"/events/{event_ticker}")


def load_client_from_config(config_path: str = "config/config.yaml") -> KalshiClient:
    """
    Load a KalshiClient from a config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Configured KalshiClient instance
    """
    import yaml

    path = Path(config_path).expanduser()
    with open(path) as f:
        config = yaml.safe_load(f)

    kalshi_config = config["kalshi"]
    return KalshiClient(
        api_key=kalshi_config["api_key"],
        private_key_path=kalshi_config["api_secret"],
        base_url=kalshi_config["base_url"],
    )
