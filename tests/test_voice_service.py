"""Tests for VoiceService and speech-to-text Whisper transcription."""

import io
import numpy as np
import pytest
from scipy.io import wavfile
from fastapi.testclient import TestClient

from app.api.main import app
from app.services.voice_service import VoiceService, get_voice_service


def _generate_test_wav_bytes(sr: int = 16000, duration: float = 0.5, freq: float = 440.0) -> bytes:
    """Helper to generate a clean synthetic sine-wave WAV in memory."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate 440Hz sine wave scaled to int16 range
    data = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    return buf.getvalue()


def test_voice_service_singleton():
    """Verify get_voice_service returns a consistent VoiceService singleton."""
    service1 = get_voice_service()
    service2 = get_voice_service()
    assert service1 is service2
    assert service1.model_name is not None


def test_audio_bytes_to_waveform_empty():
    """Verify empty audio bytes produces an empty float32 array."""
    service = VoiceService(model_name="tiny")
    arr = service.audio_bytes_to_waveform(b"")
    assert isinstance(arr, np.ndarray)
    assert len(arr) == 0
    assert arr.dtype == np.float32


def test_audio_bytes_to_waveform_resampling():
    """Verify conversion and resampling from 44.1kHz stereo to 16kHz mono float32."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    right = (0.3 * np.sin(2 * np.pi * 400 * t) * 32767).astype(np.int16)
    stereo = np.stack([left, right], axis=-1)

    buf = io.BytesIO()
    wavfile.write(buf, sr, stereo)
    wav_bytes = buf.getvalue()

    service = VoiceService(model_name="tiny")
    waveform = service.audio_bytes_to_waveform(wav_bytes, target_sr=16000)

    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert len(waveform.shape) == 1  # mono
    # Expected ~16000 samples for 1.0 second
    assert 15900 <= len(waveform) <= 16100
    # Values normalized between -1.0 and 1.0
    assert np.max(waveform) <= 1.05
    assert np.min(waveform) >= -1.05


def test_transcribe_empty_signal():
    """Verify empty or zero-byte input returns status='empty'."""
    service = VoiceService(model_name="tiny")
    res = service.transcribe(b"")
    assert res["status"] == "empty"
    assert res["text"] == ""
    assert res["duration_seconds"] == 0.0


def test_transcribe_silent_audio():
    """Verify silent waveform returns status='empty' with warning."""
    service = VoiceService(model_name="tiny")
    silence = np.zeros(16000, dtype=np.float32)
    res = service.transcribe(silence)
    assert res["status"] == "empty"
    assert "warning" in res


def test_transcribe_nonexistent_file():
    """Verify non-existent file returns status='error'."""
    service = VoiceService(model_name="tiny")
    res = service.transcribe("non_existent_file_path_12345.wav")
    assert res["status"] == "error"
    assert "not found" in res["error"].lower()


def test_transcribe_with_model():
    """Verify transcription runs with loaded model or fallback gracefully."""
    service = VoiceService(model_name="tiny")
    wav_bytes = _generate_test_wav_bytes(sr=16000, duration=0.5)

    res = service.transcribe(wav_bytes)
    assert res["status"] in ("success", "fallback")
    assert isinstance(res["text"], str)
    assert res["duration_seconds"] > 0
    assert "model" in res


def test_transcribe_fallback_mode(monkeypatch):
    """Verify fallback mode triggers cleanly when Whisper model fails to load."""
    service = VoiceService(model_name="nonexistent_fake_model")
    monkeypatch.setattr(service, "_get_model", lambda: None)

    wav_bytes = _generate_test_wav_bytes(sr=16000, duration=0.8)
    res = service.transcribe(wav_bytes)

    assert res["status"] == "fallback"
    assert "guidelines" in res["text"].lower() or len(res["text"]) > 0
    assert res["model"] == "mock-whisper-fallback"


def test_api_transcribe_endpoint():
    """Verify FastAPI POST /api/transcribe endpoint returns expected response."""
    client = TestClient(app)
    wav_bytes = _generate_test_wav_bytes(sr=16000, duration=0.5)
    files = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}

    response = client.post("/api/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "duration_seconds" in data
    assert data["status"] in ("success", "fallback")


def test_api_transcribe_empty_file():
    """Verify FastAPI POST /api/transcribe rejects empty payload."""
    client = TestClient(app)
    files = {"audio_file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/api/transcribe", files=files)
    assert response.status_code == 400

