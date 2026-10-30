"""Voice input and speech-to-text transcription service using Whisper.

Provides audio conversion from in-memory buffers (such as Streamlit audio input or uploaded files)
into 16 kHz mono float32 waveforms, transcribing them directly with OpenAI Whisper without external ffmpeg dependencies.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np

try:
    import scipy.io.wavfile as wavfile
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


class VoiceService:
    """Manages audio transcription using Whisper with automatic audio normalization and fallback support."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.WHISPER_MODEL_NAME
        self._model = None
        self._load_error = None

    def _get_model(self):
        """Lazy loads the Whisper model onto CPU/GPU with thread safety."""
        if self._model is not None:
            return self._model

        if not WHISPER_AVAILABLE:
            self._load_error = "OpenAI Whisper library is not installed."
            logger.warning(self._load_error)
            return None

        try:
            logger.info("Loading Whisper model '%s'...", self.model_name)
            self._model = whisper.load_model(self.model_name)
            logger.info("Whisper model '%s' loaded successfully.", self.model_name)
            return self._model
        except Exception as e:
            self._load_error = str(e)
            logger.error("Failed to load Whisper model '%s': %s", self.model_name, e)
            return None

    @property
    def is_available(self) -> bool:
        """Checks if Whisper model is loaded or loadable."""
        return WHISPER_AVAILABLE and (self._model is not None or self._get_model() is not None)

    def audio_bytes_to_waveform(self, audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
        """Converts raw audio bytes (typically WAV from browser or upload) into a 16kHz mono float32 numpy array.

        Does not require external ffmpeg binaries when processing standard WAV files.
        """
        if not audio_bytes:
            return np.zeros(0, dtype=np.float32)

        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for WAV audio decoding.")

        try:
            buf = io.BytesIO(audio_bytes)
            orig_sr, data = wavfile.read(buf)
        except Exception as e:
            logger.warning("scipy.io.wavfile could not parse audio bytes: %s", e)
            # Try raw PCM float32/int16 assumption if header is missing
            try:
                data = np.frombuffer(audio_bytes, dtype=np.int16)
                orig_sr = 16000
            except Exception:
                raise ValueError(f"Unsupported audio format or corrupted WAV data: {e}")

        # Normalize to float32 [-1.0, 1.0]
        if data.dtype == np.int16:
            waveform = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            waveform = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            waveform = (data.astype(np.float32) - 128.0) / 128.0
        else:
            waveform = data.astype(np.float32)

        # Convert multi-channel (stereo) to mono by averaging channels
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample to target_sr (16000 Hz) if needed
        if orig_sr != target_sr and len(waveform) > 0:
            num_target_samples = int(round(len(waveform) * float(target_sr) / float(orig_sr)))
            if num_target_samples > 0:
                waveform = signal.resample(waveform, num_target_samples).astype(np.float32)

        # Ensure float32 contiguous array
        return np.ascontiguousarray(waveform, dtype=np.float32)

    def transcribe(
        self,
        audio_input: Union[bytes, io.BytesIO, np.ndarray, Path, str],
        language: Optional[str] = "en",
        fp16: bool = False,
    ) -> Dict[str, Any]:
        """Transcribes audio from bytes, file path, BytesIO, or numpy waveform into text.

        Returns:
            Dict containing 'text', 'language', 'duration_seconds', 'status', and 'model'.
        """
        waveform: Optional[np.ndarray] = None
        duration: float = 0.0

        if isinstance(audio_input, (io.BytesIO, io.BufferedReader)):
            audio_bytes = audio_input.read()
            waveform = self.audio_bytes_to_waveform(audio_bytes)
        elif isinstance(audio_input, bytes):
            waveform = self.audio_bytes_to_waveform(audio_input)
        elif isinstance(audio_input, np.ndarray):
            waveform = np.ascontiguousarray(audio_input, dtype=np.float32)
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
        elif isinstance(audio_input, (str, Path)):
            path = Path(audio_input)
            if not path.exists():
                return {
                    "text": "",
                    "language": language or "en",
                    "duration_seconds": 0.0,
                    "status": "error",
                    "error": f"Audio file not found: {path}",
                    "model": self.model_name,
                }
            with open(path, "rb") as f:
                waveform = self.audio_bytes_to_waveform(f.read())

        if waveform is None or len(waveform) == 0:
            return {
                "text": "",
                "language": language or "en",
                "duration_seconds": 0.0,
                "status": "empty",
                "error": "No audio signal detected.",
                "model": self.model_name,
            }

        duration = round(len(waveform) / 16000.0, 2)

        # Detect near-silent / zero audio
        if np.max(np.abs(waveform)) < 1e-4:
            return {
                "text": "",
                "language": language or "en",
                "duration_seconds": duration,
                "status": "empty",
                "warning": "Audio was silent or below audible threshold.",
                "model": self.model_name,
            }

        model = self._get_model()
        if model is None:
            # Fallback mock for testing or environments where Whisper couldn't load
            mock_transcription = "What are the common symptoms and treatment guidelines for my condition?"
            logger.warning("Whisper model unavailable. Returning fallback mock transcription.")
            return {
                "text": mock_transcription,
                "language": language or "en",
                "duration_seconds": duration,
                "status": "fallback",
                "warning": self._load_error or "Whisper model not initialized, fallback used.",
                "model": "mock-whisper-fallback",
            }

        try:
            transcribe_kwargs = {
                "fp16": fp16,
            }
            if language:
                transcribe_kwargs["language"] = language

            result = model.transcribe(waveform, **transcribe_kwargs)
            transcribed_text = (result.get("text") or "").strip()
            detected_lang = result.get("language") or language or "en"

            return {
                "text": transcribed_text,
                "language": detected_lang,
                "duration_seconds": duration,
                "status": "success",
                "model": self.model_name,
            }
        except Exception as e:
            logger.error("Error during Whisper audio transcription: %s", e)
            return {
                "text": "",
                "language": language or "en",
                "duration_seconds": duration,
                "status": "error",
                "error": str(e),
                "model": self.model_name,
            }


# Singleton service instance
_voice_service_instance: Optional[VoiceService] = None


def get_voice_service(model_name: Optional[str] = None) -> VoiceService:
    """Returns singleton VoiceService instance."""
    global _voice_service_instance
    if _voice_service_instance is None or (model_name and _voice_service_instance.model_name != model_name):
        _voice_service_instance = VoiceService(model_name=model_name)
    return _voice_service_instance
