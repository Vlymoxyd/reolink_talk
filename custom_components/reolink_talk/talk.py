from __future__ import annotations

import asyncio
import logging
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.network import get_url

_LOGGER = logging.getLogger(__name__)


BC_MESSAGE_CLASS_1464: Final[bytes] = bytes.fromhex("00001464")  # status_code=0, class=1464


@dataclass(frozen=True)
class TalkAbility:
    duplex: str
    audio_stream_mode: str
    audio_type: str
    priority: int | None
    sample_rate: int
    sample_precision: int
    length_per_encoder: int
    sound_track: str


@dataclass(frozen=True)
class TalkProtocolHints:
    enc_type: str        # "AES" or "BC" — name of bc_util.EncType enum member
    talk_config_variant: int  # index into build_talk_config_variants()


def _first_text(root: ET.Element, path: str) -> str | None:
    el = root.find(path)
    if el is None or el.text is None:
        return None
    return el.text.strip()

def _all_texts(root: ET.Element, path: str) -> list[str]:
    out: list[str] = []
    for el in root.findall(path):
        if el is None or el.text is None:
            continue
        t = el.text.strip()
        if t:
            out.append(t)
    return out


def parse_talk_ability(xml: str) -> TalkAbility:
    root = ET.fromstring(xml)
    ta = root.find(".//TalkAbility")
    if ta is None:
        raise ValueError("TalkAbility not found in response")

    # Prefer "best" settings when lists are present:
    # - FDX: full duplex is typically what we want for talkback
    # - mixAudioStream: avoids dependency on the live video stream audio mode
    duplex_list = _all_texts(ta, ".//duplexList/duplex")
    stream_mode_list = _all_texts(ta, ".//audioStreamModeList/audioStreamMode")

    duplex = _first_text(ta, ".//duplex") or ""
    if "FDX" in duplex_list:
        duplex = "FDX"
    if not duplex:
        duplex = duplex_list[0] if duplex_list else "FDX"

    audio_stream_mode = _first_text(ta, ".//audioStreamMode") or ""
    if "mixAudioStream" in stream_mode_list:
        audio_stream_mode = "mixAudioStream"
    if not audio_stream_mode:
        audio_stream_mode = stream_mode_list[0] if stream_mode_list else "followVideoStream"

    ac = ta.find(".//audioConfig")
    if ac is None:
        raise ValueError("audioConfig not found in TalkAbility")

    audio_type = _first_text(ac, ".//audioType") or "adpcm"
    prio_txt = _first_text(ac, ".//priority")
    priority = int(prio_txt) if prio_txt and prio_txt.isdigit() else None
    sample_rate = int(_first_text(ac, ".//sampleRate") or "16000")
    sample_precision = int(_first_text(ac, ".//samplePrecision") or "16")
    length_per_encoder = int(_first_text(ac, ".//lengthPerEncoder") or "1024")
    sound_track = _first_text(ac, ".//soundTrack") or "mono"

    return TalkAbility(
        duplex=duplex,
        audio_stream_mode=audio_stream_mode,
        audio_type=audio_type,
        priority=priority,
        sample_rate=sample_rate,
        sample_precision=sample_precision,
        length_per_encoder=length_per_encoder,
        sound_track=sound_track,
    )


def build_talk_config_xml(channel: int, ability: TalkAbility) -> str:
    # Match the XML shapes documented by neolink.
    prio = f"<priority>{ability.priority}</priority>\n" if ability.priority is not None else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        "<body>\n"
        '<TalkConfig version="1.1">\n'
        f"<channelId>{channel}</channelId>\n"
        f"<duplex>{ability.duplex}</duplex>\n"
        f"<audioStreamMode>{ability.audio_stream_mode}</audioStreamMode>\n"
        "<audioConfig>\n"
        + prio
        + f"<audioType>{ability.audio_type}</audioType>\n"
        + f"<sampleRate>{ability.sample_rate}</sampleRate>\n"
        + f"<samplePrecision>{ability.sample_precision}</samplePrecision>\n"
        + f"<lengthPerEncoder>{ability.length_per_encoder}</lengthPerEncoder>\n"
        + f"<soundTrack>{ability.sound_track}</soundTrack>\n"
        + "</audioConfig>\n"
        + "</TalkConfig>\n"
        + "</body>\n"
    )


def build_talk_config_variants(channel: int, ability: TalkAbility) -> list[str]:
    """Return a small set of TalkConfig XML variants for firmware quirks."""
    full = build_talk_config_xml(channel, ability)
    variants: list[str] = [full]

    # Some firmwares appear picky about the XML header.
    if full.lstrip().startswith("<?xml"):
        try:
            _, rest = full.split("\n", 1)
            variants.append(rest)
        except ValueError:
            pass

    # Some firmwares expect just the TalkConfig element (no <body> wrapper).
    start = full.find("<TalkConfig")
    end = full.rfind("</TalkConfig>")
    if start != -1 and end != -1:
        tc = full[start : end + len("</TalkConfig>")] + "\n"
        if tc not in variants:
            variants.append(tc)

    return variants


def bcmedia_adpcm_packet(block: bytes) -> bytes:
    # Port of neolink bcmedia_adpcm() + padding rules.
    # block must be: 4 bytes predictor state + N bytes adpcm payload.
    if len(block) < 5:
        raise ValueError("ADPCM block too small")
    payload_len = len(block) + 4  # + magic u16 + blocksize u16
    # Neolink format: "block size without header, halved" (DVI-4 payload bytes / 2).
    block_size = ((len(block) - 4) // 2)
    header = struct.pack(
        "<IHHHH",
        0x62773130,  # MAGIC_HEADER_BCMEDIA_ADPCM
        payload_len,
        payload_len,
        0x0100,  # MAGIC_HEADER_BCMEDIA_ADPCM_DATA
        block_size,
    )
    pad_len = (-len(block)) % 8
    return header + block + (b"\x00" * pad_len)


def talk_binary_payload(adpcm_bytes: bytes, full_block_size: int, blocks_per_payload: int = 4) -> list[tuple[bytes, int]]:
    # Returns list of (binary_payload, blocks_in_payload).
    out: list[tuple[bytes, int]] = []
    blocks = [adpcm_bytes[i : i + full_block_size] for i in range(0, len(adpcm_bytes), full_block_size)]
    # Drop incomplete trailing block (if any)
    if blocks and len(blocks[-1]) != full_block_size:
        blocks = blocks[:-1]
    for i in range(0, len(blocks), blocks_per_payload):
        group = blocks[i : i + blocks_per_payload]
        payload = b"".join(bcmedia_adpcm_packet(b) for b in group)
        out.append((payload, len(group)))
    return out


async def fetch_bytes(hass: HomeAssistant, url: str) -> bytes:
    session = async_get_clientsession(hass)
    fetch_url = url
    # If it's a Home Assistant local URL (TTS/media proxy), we need to sign it,
    # because we are fetching server-side (no browser cookies / auth headers).
    try:
        parsed = urlparse(url)
        if url.startswith("/"):
            path_q = url
        elif parsed.scheme in ("http", "https"):
            path_q = parsed.path + (("?" + parsed.query) if parsed.query else "")
            base = get_url(hass, allow_internal=True)
            base_netloc = urlparse(base).netloc
            if parsed.netloc != base_netloc:
                path_q = ""
        else:
            path_q = ""

        if path_q:
            from homeassistant.components.http.auth import async_sign_path

            base = get_url(hass, allow_internal=True)
            signed = async_sign_path(hass, path_q)
            fetch_url = f"{base}{signed}"
    except Exception:
        # Best-effort; fall back to raw URL.
        pass

    async with session.get(fetch_url, allow_redirects=True) as resp:
        resp.raise_for_status()
        return await resp.read()


async def ffmpeg_to_pcm_s16le(
    input_bytes: bytes,
    *,
    sample_rate: int,
    volume: float = 1.0,
) -> bytes:
    """Decode arbitrary audio to mono 16-bit PCM (little-endian) at sample_rate."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-af",
        f"volume={max(0.0, float(volume))}",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    out, err = await proc.communicate(input_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode('utf-8', 'ignore')}")
    return out


# IMA/DVI ADPCM encoder tables (standard IMA ADPCM).
_IMA_INDEX_TABLE: Final[list[int]] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]
_IMA_STEP_TABLE: Final[list[int]] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41,
    45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190,
    209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499,
    2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845,
    8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385,
    24623, 27086, 29794, 32767,
]


def _ima_encode_nibble(sample: int, predictor: int, step_index: int) -> tuple[int, int, int]:
    step = _IMA_STEP_TABLE[step_index]
    diff = sample - predictor
    sign = 0
    if diff < 0:
        sign = 8
        diff = -diff
    delta = 0
    vpdiff = step >> 3
    if diff >= step:
        delta |= 4
        diff -= step
        vpdiff += step
    if diff >= (step >> 1):
        delta |= 2
        diff -= step >> 1
        vpdiff += step >> 1
    if diff >= (step >> 2):
        delta |= 1
        vpdiff += step >> 2
    predictor = max(-32768, min(32767, predictor + (vpdiff if not sign else -vpdiff)))
    step_index = max(0, min(88, step_index + _IMA_INDEX_TABLE[delta | sign]))
    return (delta | sign) & 0xF, predictor, step_index


def ima_adpcm_encode_dvi_blocks(pcm_s16le: bytes, *, full_block_size: int) -> bytes:
    """Encode PCM s16le into DVI-4 ADPCM blocks (4-byte header + nibble payload)."""
    if full_block_size < 8 or len(pcm_s16le) % 2 != 0:
        raise ValueError("invalid full_block_size or PCM length")

    payload_bytes = full_block_size - 4
    payload_samples = payload_bytes * 2
    sample_count = len(pcm_s16le) // 2
    samples = struct.unpack("<" + "h" * sample_count, pcm_s16le) if sample_count else ()
    if not samples:
        return b""

    predictor = int(samples[0])
    step_index = 0
    pos = 1
    out = bytearray()
    while pos <= len(samples):
        block = bytearray()
        block += struct.pack("<hBB", predictor, step_index, 0)
        nibble_acc = None
        for _ in range(payload_samples):
            s = int(samples[pos]) if pos < len(samples) else 0
            pos += 1
            nib, predictor, step_index = _ima_encode_nibble(s, predictor, step_index)
            if nibble_acc is None:
                nibble_acc = nib
            else:
                block.append(((nibble_acc & 0xF) << 4) | (nib & 0xF))
                nibble_acc = None
        if nibble_acc is not None:
            block.append(nibble_acc & 0xF)
        if len(block) < full_block_size:
            block.extend(b"\x00" * (full_block_size - len(block)))
        out += block[:full_block_size]
        if pos >= len(samples):
            break
    return bytes(out)


async def send_talk_binary(
    bc,  # reolink_aio.baichuan.Baichuan
    channel: int,
    binary_payload: bytes,
    *,
    mess_id: int | None = None,
    enc_type=None,
) -> None:
    # Like reolink_aio Baichuan.send(), but:
    # - doesn't wait for a response
    # - encrypts only the Extension XML
    # - appends the BcMedia binary payload unencrypted
    from reolink_aio.baichuan import util as bc_util
    from reolink_aio.baichuan import xmls

    if not getattr(bc, "_logged_in", False):
        await bc.login()

    # reolink_aio internal fields have changed between versions. We only need a
    # monotonically increasing 24-bit message id for the Baichuan header.
    if not hasattr(bc, "_mess_id"):
        setattr(bc, "_mess_id", 0)

    # Map channel -> ch_id like reolink_aio does (1.. for channels)
    ch_id = channel + 1

    ext = (
        xmls.XML_HEADER
        + '<Extension version="1.1">\n'
        + "<binaryData>1</binaryData>\n"
        + f"<channelId>{channel}</channelId>\n"
        + "</Extension>\n"
    )

    if mess_id is None:
        bc._mess_id = (bc._mess_id + 1) % 16777216
    else:
        bc._mess_id = mess_id

    # IMPORTANT: Baichuan TCP parsing uses:
    # - len_body (rec_len_body) to know how many bytes to consume after the header
    # - payload_offset to split "message" vs "payload" bytes
    #
    # For normal AES messages, reolink_aio sets both to the *plaintext* length
    # and sends only encrypted body bytes (no extra payload bytes).
    #
    # For talk/audio (cmd 202), we send:
    # - encrypted Extension XML ("message" bytes)
    # - raw, unencrypted BcMedia ADPCM bytes ("payload" bytes)
    #
    # Therefore, the header must use ON-WIRE byte lengths:
    # - payload_offset = len(enc_ext)
    # - mess_len = len(enc_ext) + len(binary_payload)
    if enc_type is None:
        enc_type = bc_util.EncType.AES

    if enc_type == bc_util.EncType.BC:
        enc_ext = bc_util.encrypt_baichuan(ext, ch_id)  # enc_offset = ch_id
    else:
        enc_ext = bc._aes_encrypt(ext if isinstance(ext, bytes) else ext.encode())
    payload_offset = len(enc_ext)
    mess_len = payload_offset + len(binary_payload)

    cmd_id = 202
    header = (
        bytes.fromhex(bc_util.HEADER_MAGIC)
        + int(cmd_id).to_bytes(4, "little")
        + int(mess_len).to_bytes(4, "little")
        + int(ch_id).to_bytes(1, "little")
        + int(bc._mess_id).to_bytes(3, "little")
        + BC_MESSAGE_CLASS_1464
        + int(payload_offset).to_bytes(4, "little")
    )

    packet = header + enc_ext + binary_payload
    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug(
            "TalkBinary %s: ch=%s mess_id=%s enc=%s enc_ext=%s payload=%s mess_len=%s payload_offset=%s",
            getattr(bc, "_host", "?"),
            channel,
            getattr(bc, "_mess_id", "?"),
            getattr(enc_type, "value", str(enc_type)),
            len(enc_ext),
            len(binary_payload),
            mess_len,
            payload_offset,
        )

    # reolink_aio >= 0.21 no longer exposes bc._transport / bc._protocol directly:
    # the TCP/UDP transport is now wrapped in bc._connection, which serialises
    # writes under its own mutex and handles timeouts/connection errors.
    #
    # Talk audio frames (cmd 202) are streamed fire-and-forget (the camera does
    # not ACK each frame); playback timing is handled by the pacing sleeps in
    # talk_playback(), so send_without_wait() is exactly the right primitive.
    await bc._connect_if_needed()
    conn = bc._connection
    if conn is None:
        raise RuntimeError(f"Baichuan host {getattr(bc, '_host', '?')}: no connection available for talk playback")
    await conn.send_without_wait(packet, cmd_id=cmd_id)


async def talk_playback(
    bc,  # reolink_aio.baichuan.Baichuan
    channel: int,
    adpcm_bytes: bytes,
    ability: TalkAbility,
    *,
    block_align: int | None = None,
    hints: TalkProtocolHints | None = None,
) -> TalkProtocolHints:
    from reolink_aio.exceptions import ApiError
    from reolink_aio.baichuan import util as bc_util

    async def _send_with_fallback(cmd_id: int, *, body: str = "") -> bc_util.EncType:
        # Some firmwares expect BC encryption for talk commands (otherwise 400).
        for enc in (bc_util.EncType.AES, bc_util.EncType.BC):
            try:
                await bc.send(cmd_id=cmd_id, channel=channel, body=body, enc_type=enc)
                return enc
            except ApiError as err:
                # Retry with BC only for "bad request" style errors.
                if getattr(err, "rspCode", None) == 400 and enc == bc_util.EncType.AES:
                    continue
                raise
        return bc_util.EncType.BC

    async def _stop_talk_best_effort() -> None:
        try:
            await bc.send(cmd_id=11, channel=channel, enc_type=enc_used)
            await asyncio.sleep(0.1)
        except Exception:
            pass

    variants = build_talk_config_variants(channel, ability)
    # Seed enc_used from hints so StopTalk uses the known-good enc_type immediately.
    _hints_enc: bc_util.EncType | None = None
    if hints is not None:
        try:
            _hints_enc = bc_util.EncType[hints.enc_type]
        except KeyError:
            pass
    enc_used: bc_util.EncType = _hints_enc if _hints_enc is not None else bc_util.EncType.AES
    variant_idx = 0

    # Try cached (variant, enc) first — skips trial-and-error on known-good cameras.
    hints_succeeded = False
    if hints is not None:
        h_idx = hints.talk_config_variant
        try:
            h_enc = bc_util.EncType[hints.enc_type]
        except KeyError:
            h_enc = None
        if h_enc is not None and 0 <= h_idx < len(variants):
            try:
                await bc.send(cmd_id=201, channel=channel, body=variants[h_idx], enc_type=h_enc)
                enc_used = h_enc
                variant_idx = h_idx
                hints_succeeded = True
            except Exception:
                pass  # hints stale (firmware update?); fall through to full search

    if not hints_succeeded:
        # Send TalkConfig first (cmd 201). If we get 422, stop talk and retry.
        try:
            last_err: Exception | None = None
            for v_idx, talk_cfg in enumerate(variants):
                try:
                    enc_used = await _send_with_fallback(201, body=talk_cfg)
                    variant_idx = v_idx
                    last_err = None
                    break
                except ApiError as err:
                    last_err = err
                    rsp = getattr(err, "rspCode", None)
                    if rsp in (400, 421, 422):
                        # Many firmwares require stopping an existing talk session first.
                        await _stop_talk_best_effort()
                        continue
                    raise
            if last_err is not None:
                _LOGGER.error(
                    "TalkConfig rejected for ch=%s sample_rate=%s length_per_encoder=%s audio_type=%s (rspCode=%s)",
                    channel,
                    ability.sample_rate,
                    ability.length_per_encoder,
                    ability.audio_type,
                    getattr(last_err, "rspCode", None),
                )
                raise last_err
        except ApiError as err:
            if getattr(err, "rspCode", None) in (421, 422):
                # Stop talk and retry. Best-effort: 421 on StopTalk = camera rejects
                # it (wrong owner / no session) — swallow and retry TalkConfig anyway.
                await _stop_talk_best_effort()
                last_err = None
                for v_idx, talk_cfg in enumerate(variants):
                    try:
                        enc_used = await _send_with_fallback(201, body=talk_cfg)
                        variant_idx = v_idx
                        last_err = None
                        break
                    except ApiError as err2:
                        last_err = err2
                        rsp = getattr(err2, "rspCode", None)
                        if rsp in (400, 421, 422):
                            await _stop_talk_best_effort()
                            continue
                        raise
                if last_err is not None:
                    _LOGGER.error(
                        "TalkConfig rejected after stop/retry for ch=%s sample_rate=%s length_per_encoder=%s audio_type=%s (rspCode=%s)",
                        channel,
                        ability.sample_rate,
                        ability.length_per_encoder,
                        ability.audio_type,
                        getattr(last_err, "rspCode", None),
                    )
                    raise last_err
            else:
                raise

    full_block_size = int(block_align or ability.length_per_encoder)
    payloads = talk_binary_payload(adpcm_bytes, full_block_size, blocks_per_payload=4)

    try:
        # Deadline-based pacing: each payload is sent at an absolute target time
        # derived from t0. asyncio.sleep drift in one iteration is corrected by
        # the shorter sleep in the next, so the camera never starves of data.
        t0 = asyncio.get_event_loop().time()
        cumulative_duration = 0.0
        for payload, blocks_in_payload in payloads:
            await send_talk_binary(bc, channel, payload, enc_type=enc_used)
            adpcm_len = full_block_size * blocks_in_payload
            samples_sent = (adpcm_len - 4 * blocks_in_payload) * 2 + blocks_in_payload
            cumulative_duration += samples_sent / float(ability.sample_rate)
            remaining = t0 + cumulative_duration - asyncio.get_event_loop().time()
            if remaining > 0:
                await asyncio.sleep(remaining)
        # Wait until playback should be complete, plus 50ms network grace.
        # If the last deadline sleep already drifted past this point, skip.
        t_done = t0 + cumulative_duration + 0.05
        remaining = t_done - asyncio.get_event_loop().time()
        if remaining > 0:
            await asyncio.sleep(remaining)
    finally:
        try:
            await bc.send(cmd_id=11, channel=channel, enc_type=enc_used)
        except Exception:
            # Best-effort stop; do not mask the primary exception.
            pass

    return TalkProtocolHints(enc_type=enc_used.name, talk_config_variant=variant_idx)
