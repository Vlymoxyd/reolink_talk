from __future__ import annotations

import asyncio
import dataclasses
import logging
import time

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.storage import Store

from .talk import TalkAbility, TalkProtocolHints, parse_talk_ability

_LOGGER = logging.getLogger(__name__)

_PROBE_TIMEOUT: float = 0.5
_IDLE_RECONNECT_AFTER: float = 60.0  # camera idle timeout threshold (seconds)
_STORE_VERSION = 1
_STORE_KEY = "reolink_talk_cache"


class TalkCache:
    """Single HA Store shared across all camera sessions.

    The JSON file contains one section per camera keyed by "{entry_id}_{channel}".
    Concurrent writes are serialised by an asyncio lock so sessions don't
    overwrite each other's sections.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        self._store: Store = Store(hass, _STORE_VERSION, _STORE_KEY)
        self._data: dict = {}
        self._load_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if not self._loaded:
                self._data = await self._store.async_load() or {}
                self._loaded = True

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    async def set(self, key: str, value: dict) -> None:
        async with self._write_lock:
            self._data[key] = value
            await self._store.async_save(dict(self._data))


class BaichuanSession:
    """Persistent authenticated Baichuan TCP session for a single camera channel.

    Subsequent play_media calls reuse the open connection.  Liveness is checked
    by an async probe (cmd 10) sent in parallel with audio preparation, so the
    check adds zero latency in the common case and also resets the camera's idle
    timer on every call.

    TalkAbility and TalkProtocolHints are persisted via TalkCache so even the
    very first call after an HA restart skips both network fetches and
    trial-and-error protocol negotiation.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        host: str,
        port: int,
        http_port: int | None,
        use_https: bool | None,
        username: str,
        password: str,
        channel: int,
        cache: TalkCache,
        cache_key: str,
    ) -> None:
        self._hass = hass
        self._host_addr = host
        self._port = port
        self._http_port = http_port
        self._use_https = use_https
        self._username = username
        self._password = password
        self._channel = channel

        self._reolink_host = None
        self._bc = None
        self._ability: TalkAbility | None = None
        self._hints: TalkProtocolHints | None = None
        self._last_success: float = 0.0
        self._connect_lock = asyncio.Lock()
        self._cache = cache
        self._cache_key = cache_key

    @property
    def bc(self):
        return self._bc

    @property
    def cached_ability(self) -> TalkAbility | None:
        """Return cached TalkAbility without touching the network, or None."""
        return self._ability

    @property
    def cached_hints(self) -> TalkProtocolHints | None:
        """Return cached protocol hints, or None if not yet discovered."""
        return self._hints

    async def load_ability_cache(self) -> None:
        """Load TalkAbility and TalkProtocolHints from the shared cache file."""
        if self._ability is not None:
            return
        await self._cache._ensure_loaded()
        entry = self._cache.get(self._cache_key)
        if not entry:
            return
        # Support both current format {"ability": {...}, "hints": {...}}
        # and the legacy per-file flat format {"duplex": ..., ...}.
        if "ability" in entry:
            ability_data = entry["ability"]
            hints_data = entry.get("hints")
        else:
            ability_data = entry
            hints_data = None
        try:
            self._ability = TalkAbility(**ability_data)
            _LOGGER.debug(
                "BaichuanSession: loaded cached ability for %s ch%s",
                self._host_addr,
                self._channel,
            )
        except Exception:
            _LOGGER.debug(
                "BaichuanSession: ability cache invalid for %s, will re-fetch",
                self._host_addr,
            )
            return
        if hints_data:
            try:
                self._hints = TalkProtocolHints(**hints_data)
            except Exception:
                pass  # hints corrupt; will re-discover on first call

    async def _save_cache(self) -> None:
        """Persist ability and hints for this camera into the shared cache."""
        data: dict = {}
        if self._ability is not None:
            data["ability"] = dataclasses.asdict(self._ability)
        if self._hints is not None:
            data["hints"] = dataclasses.asdict(self._hints)
        if data:
            await self._cache.set(self._cache_key, data)

    def touch(self) -> None:
        """Record that the connection was just used successfully."""
        self._last_success = time.monotonic()

    async def save_hints(self, hints: TalkProtocolHints) -> None:
        """Persist protocol hints if they changed."""
        if hints != self._hints:
            self._hints = hints
            await self._save_cache()

    async def probe(self) -> bool:
        """Send cmd 10 to verify the connection is alive and reset the idle timer.

        Also updates the ability cache from the live response if it changed.
        Returns True if the connection is confirmed alive, False if dead.
        Checks the local transport state first (no network) so a cleanly-closed
        TCP connection is detected instantly without waiting for the timeout.
        """
        if self._bc is None:
            return False
        # Staleness check: if we haven't had a successful exchange in over 60s,
        # the camera's idle timeout has certainly fired — skip the probe entirely.
        if time.monotonic() - self._last_success > _IDLE_RECONNECT_AFTER:
            self._bc = None  # _reolink_host kept so _login() can logout cleanly
            return False
        # Local check: if the asyncio transport already knows it's closed (the
        # camera sent TCP FIN/RST), bail immediately without a network round-trip.
        conn = getattr(self._bc, "_connection", None)
        if conn is not None and not getattr(conn, "connection_open", True):
            self._bc = None
            return False
        try:
            async with asyncio.timeout(_PROBE_TIMEOUT):
                response = await self._bc.send(cmd_id=10, channel=self._channel)
            self._last_success = time.monotonic()
            try:
                new_ability = parse_talk_ability(response)
                if new_ability != self._ability:
                    self._ability = new_ability
                    await self._save_cache()
            except Exception:
                pass  # Keep existing cached ability if parse fails
            return True
        except Exception:
            self._bc = None  # mark dead; ensure_connected() will reconnect
            return False

    async def ensure_connected(self):
        """Return an authenticated Baichuan object, connecting if not already open."""
        async with self._connect_lock:
            if self._bc is None:
                await self._login()
        return self._bc

    async def _login(self) -> None:
        from reolink_aio.api import Host

        if self._reolink_host is not None:
            try:
                async with asyncio.timeout(1.0):
                    await self._reolink_host.logout()
            except Exception:
                pass

        self._reolink_host = Host(
            host=self._host_addr,
            username=self._username,
            password=self._password,
            port=self._http_port,
            use_https=self._use_https,
            bc_port=self._port,
            aiohttp_get_session_callback=lambda: async_get_clientsession(self._hass),
        )
        self._bc = self._reolink_host.baichuan
        await self._bc.login()
        self._last_success = time.monotonic()
        _LOGGER.debug("BaichuanSession: (re)connected to %s", self._host_addr)

    async def get_ability(self) -> TalkAbility:
        """Return TalkAbility, from cache or fetched from camera."""
        if self._ability is None:
            bc = await self.ensure_connected()
            response = await bc.send(cmd_id=10, channel=self._channel)
            self._ability = parse_talk_ability(response)
            await self._save_cache()
        return self._ability

    async def close(self) -> None:
        """Close the Baichuan connection cleanly (call on entity removal)."""
        async with self._connect_lock:
            if self._reolink_host is not None:
                try:
                    await self._reolink_host.logout()
                except Exception:
                    pass
            self._reolink_host = None
            self._bc = None
