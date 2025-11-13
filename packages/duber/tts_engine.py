import os
import logging
import torch
import hashlib
import shutil
from typing import Optional
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
logger = logging.getLogger(__name__)
# NLTK monkey-patch for g2p_en (before any Melo import)
try:
    import nltk
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('cmudict', quiet=True)
    nltk.download('punkt', quiet=True)
    # Patch g2p_en's tagger load (override the faulty _load_tagger)
    from g2p_en import g2p
    original_load = g2p._load_tagger # Assume internal method; adjust if needed
    def patched_load_tagger(self):
        try:
            # Use NLTK's standard tagger
            tagger = nltk.data.load('taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
            logger.info("Patched g2p_en with NLTK standard tagger")
            return tagger # Return the classifier
        except Exception as e:
            logger.warning(f"Tagger patch failed: {e}; falling back to original")
            return original_load(self)
    g2p._load_tagger = patched_load_tagger # Bind to class
    logger.info("✅ NLTK g2p_en patch applied")
except Exception as e:
    logger.warning(f"NLTK patch setup failed: {e}")
class TextToSpeechEngine:
    """Text-to-speech using OpenVoice v2 + MeloTTS (with NLTK-patched G2P)"""
    def __init__(self, language: str = 'en'):
        """Initialize text-to-speech engine with OpenVoice v2 + MeloTTS"""
        self.language = language # Store for potential use; synthesis overrides per-call
        try:
            logger.info("Loading OpenVoice v2 models...")
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # OpenVoice V2 structure
            ckpt_converter = 'checkpoints_v2/converter'
            # Load tone color converter
            from openvoice.api import ToneColorConverter
            self.tone_color_converter = ToneColorConverter(
                f'{ckpt_converter}/config.json',
                device=self.device
            )
            self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
            # MeloTTS will be loaded on demand per language
            self.melo_models = {}
            # Language mapping for MeloTTS
            self.lang_map = {
                'en': 'EN',
                'es': 'ES',
                'fr': 'FR',
                'zh': 'ZH',
                'ja': 'JP',
                'ko': 'KR',
                'pt': 'EN', # Use English model for Portuguese (best available; note: may cause G2P issues on PT text)
            }
            self.output_dir = 'outputs_v2'
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"OpenVoice v2 loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load OpenVoice models: {e}")
            logger.error("Make sure checkpoints_v2 directory exists with proper model files")
            raise
    def _get_melo_model(self, language: str):
        """Load MeloTTS model for a specific language"""
        melo_lang = self.lang_map.get(language, 'EN')
        if melo_lang not in self.melo_models:
            try:
                from melo.api import TTS
                logger.info(f"Loading MeloTTS model for {melo_lang}...")
                # Note: For non-JP langs, MeCab should not be required; if error persists post-install, consider custom phonemizer
                self.melo_models[melo_lang] = TTS(language=melo_lang, device=self.device)
            except Exception as mecab_err:
                if 'MeCab' in str(mecab_err):
                    logger.error(f"MeCab error during MeloTTS load for {melo_lang}: {mecab_err}")
                    logger.info("Ensure 'sudo apt install mecab libmecab-dev' and 'uv add unidic-lite' run")
                    raise
                else:
                    raise
        return self.melo_models[melo_lang]
    def synthesize(
            self,
            text: str,
            language: str = 'en',
            reference_speaker: Optional[str] = None,
            speaker_wav: Optional[str] = None,
            # New alias for backward compatibility (e.g., from OpenVoice Gradio examples)
            speed: float = 1.0,
            output_path: Optional[str] = None
    ) -> str:
        """
        Synthesize speech from text using OpenVoice v2
        Args:
            text: Text to synthesize
            language: Target language code
            reference_speaker: Path to reference audio for voice cloning (optional)
            speaker_wav: Alternative name for reference_speaker (maps to it)
            speed: Speech speed multiplier
            output_path: Optional custom path for output (defaults to unique temp file)
        Returns:
            Path to generated audio file
        """
        src_path = None # Declare early to avoid scope error in cleanup
        try:
            # Use speaker_wav if provided, else reference_speaker
            ref_speaker = speaker_wav or reference_speaker
            if speaker_wav and not reference_speaker:
                logger.info("Using 'speaker_wav' as reference_speaker for cloning")
            if not output_path:
                # Generate unique filename to avoid overwrites
                hash_name = hashlib.md5(text.encode()).hexdigest()[:8]
                output_path = f'{self.output_dir}/tmp_{hash_name}_{language}.wav'
            # Get MeloTTS model for this language
            melo_model = self._get_melo_model(language)
            # Generate base speech with MeloTTS
            base_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            src_path = f'{self.output_dir}/tmp_base_{base_hash}.wav'
            # Get speaker ID (use default speaker for the language)
            speaker_ids = melo_model.hps.data.spk2id
            speaker_key = list(speaker_ids.keys())[0] # Use first available speaker
            speaker_id = speaker_ids[speaker_key]
            # Generate speech
            melo_model.tts_to_file(text, speaker_id, src_path, speed=speed)
            # If reference speaker provided, apply voice cloning
            if ref_speaker and os.path.exists(ref_speaker):
                # Extract target speaker embedding
                target_se, _ = se_extractor.get_se(
                    ref_speaker,
                    self.tone_color_converter,
                    target_dir='processed',
                    vad=False  # Skip VAD/splitting for short references to avoid "too short" error
                )
                # Load source speaker embedding
                speaker_key_normalized = speaker_key.lower().replace('_', '-')
                source_se_path = f'checkpoints_v2/base_speakers/ses/{speaker_key_normalized}.pth'
                if not os.path.exists(source_se_path):
                    # Fallback to default
                    source_se_path = f'checkpoints_v2/base_speakers/ses/en-default.pth'
                    logger.warning(f"Using fallback source_se: {source_se_path}")
                source_se = torch.load(source_se_path, map_location=self.device)
                # Convert tone color
                encode_message = "@MyShell"
                self.tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=output_path,
                    message=encode_message
                )
                # Cleanup temp src
                if src_path and os.path.exists(src_path) and src_path != output_path:
                    os.remove(src_path)
                logger.info(f"Cloned voice to {output_path}")
                return output_path
            else:
                # No voice cloning, move/rename base to output_path
                if src_path and src_path != output_path:
                    shutil.move(src_path, output_path)
                logger.info(f"Generated base speech: {output_path}")
                return output_path
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            # Cleanup partial files (safe check for None/existence)
            for path in [src_path, output_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass # Ignore cleanup failures
            raise