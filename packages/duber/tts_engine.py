import os
import logging
import torch
import hashlib
import shutil
from typing import Optional
from openvoice import se_extractor  # Optional for post-cloning polish
from openvoice.api import ToneColorConverter
import librosa
from pydub import AudioSegment

logger = logging.getLogger(__name__)
logger.info("TTS Engine: XTTS-v2 (Coqui fork) for PT-BR naturalness, MeloTTS for others")

# Suppress warnings for cleaner logs
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="TTS")


class TextToSpeechEngine:
    def __init__(self, language: str = 'en'):
        self.language = language
        self.lang_map = {
            'en': 'EN', 'es': 'ES', 'fr': 'FR', 'zh': 'ZH', 'ja': 'JP', 'ko': 'KR', 'pt': 'pt'  # Fixed for XTTS
        }
        try:
            logger.info("Loading OpenVoice v2 (optional cloning polish)...")
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            ckpt_converter = 'checkpoints_v2/converter'
            self.tone_color_converter = ToneColorConverter(
                f'{ckpt_converter}/config.json', device=self.device
            )
            self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
            self.xtts_model = None
            self.melo_models = {}
            self.output_dir = 'outputs_v2'
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"OpenVoice v2 loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load OpenVoice: {e}")
            self.tone_color_converter = None  # Graceful degradation

    def _load_xtts(self):
        if self.xtts_model:
            return
        try:
            from TTS.api import TTS
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.tts.configs.shared_configs import BaseDatasetConfig
            import torch.serialization

            # Full allowlist for XTTS classes (PyTorch 2.6+ fix)
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

            logger.info("Loading XTTS-v2 (Coqui fork) for PT-BR...")
            self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=self.device != "cpu")
            logger.info("✅ XTTS-v2 loaded (native PT-BR, zero-shot cloning)")
        except Exception as e:
            logger.error(f"XTTS load failed: {e}. Run: uv add coqui-tts")
            self.xtts_model = None

    def _get_melo_model(self, language: str):
        melo_lang = self.lang_map.get(language, 'EN')
        if melo_lang not in self.melo_models:
            try:
                from melo.api import TTS
                logger.info(f"Loading MeloTTS {melo_lang}...")
                self.melo_models[melo_lang] = TTS(language=melo_lang, device=self.device)
            except Exception as e:
                logger.error(f"MeloTTS failed: {e}")
                raise
        return self.melo_models[melo_lang]

    def _adjust_speed_for_duration(self, audio_path: str, target_duration: float, max_retries: int = 1) -> str:
        for retry in range(max_retries + 1):
            if not os.path.exists(audio_path):
                return audio_path
            current_dur = librosa.get_duration(filename=audio_path)
            if abs(current_dur - target_duration) < 0.2:
                return audio_path
            if current_dur > 0:
                new_speed = max(0.5, min(2.0, target_duration / current_dur))
                audio = AudioSegment.from_wav(audio_path)
                adjusted = audio.speedup(playback_speed=new_speed, chunk_size=150, crossfade=25)
                adjusted_path = audio_path.replace('.wav', f'_adj{retry}.wav')
                adjusted.export(adjusted_path, format='wav')
                logger.debug(f"Speed adjusted to {new_speed:.2f}x")
                return adjusted_path
        return audio_path

    def synthesize(
            self, text: str, language: str = 'en', reference_speaker: Optional[str] = None,
            speaker_wav: Optional[str] = None, speed: float = 1.0, output_path: Optional[str] = None,
            target_duration: Optional[float] = None, enable_clone: bool = True, enable_polish: bool = True
    ) -> str:
        ref_speaker = speaker_wav or reference_speaker
        if not output_path:
            hash_name = hashlib.md5(text.encode()).hexdigest()[:8]
            output_path = f'{self.output_dir}/tmp_{hash_name}_{language}.wav'
        base_path = output_path.replace('.wav', '_base.wav')
        try:
            if language == 'pt':
                self._load_xtts()
                if self.xtts_model:
                    # XTTS-v2: Native PT-BR synthesis + built-in cloning
                    xtts_lang = self.lang_map.get(language, 'pt')
                    self.xtts_model.tts_to_file(
                        text=text, speaker_wav=ref_speaker if enable_clone else None,
                        language=xtts_lang, file_path=base_path, speed=speed,
                        temperature=0.65  # Lower for stability, less noise
                    )
                    logger.info(f"XTTS-v2 PT-BR synthesized: '{text[:50]}...' (cloned if ref)")
                else:
                    raise ValueError("XTTS unavailable; install coqui-tts")
            else:
                # Non-PT: MeloTTS
                melo_model = self._get_melo_model(language)
                speaker_ids = melo_model.hps.data.spk2id
                speaker_key = list(speaker_ids.keys())[0]
                speaker_id = speaker_ids[speaker_key]
                melo_model.tts_to_file(text, speaker_id, base_path, speed=speed)
                logger.info(f"MeloTTS synthesized: '{text[:50]}...'")

            if target_duration:
                base_path = self._adjust_speed_for_duration(base_path, target_duration)

            # Optional OpenVoice polish (for extra timbre fidelity)
            if enable_polish and self.tone_color_converter and ref_speaker and os.path.exists(
                    ref_speaker) and os.path.getsize(base_path) > 16000:
                try:
                    target_se, _ = se_extractor.get_se(ref_speaker, self.tone_color_converter, target_dir='processed',
                                                       vad=False)
                    source_se_path = f'checkpoints_v2/base_speakers/ses/por-default.pth' if language == 'pt' else f'checkpoints_v2/base_speakers/ses/en-default.pth'
                    if not os.path.exists(source_se_path):
                        source_se_path = f'checkpoints_v2/base_speakers/ses/en-default.pth'
                    source_se = torch.load(source_se_path, map_location=self.device)
                    self.tone_color_converter.convert(
                        audio_src_path=base_path, src_se=source_se, tgt_se=target_se,
                        output_path=output_path, message="@MyShell"
                    )
                    logger.info("OpenVoice polish applied")
                except Exception as e:
                    logger.warning(f"OpenVoice skipped: {e}")
                    shutil.move(base_path, output_path)
            else:
                shutil.move(base_path, output_path)

            if base_path != output_path and os.path.exists(base_path):
                os.remove(base_path)
            return output_path
        except Exception as e:
            logger.error(f"Synthesis failed for '{text[:50]}...': {e}")
            if target_duration:
                silence = AudioSegment.silent(duration=int(target_duration * 1000))
                silence.export(output_path, format="wav")
                logger.warning("Silence fallback used")
            raise