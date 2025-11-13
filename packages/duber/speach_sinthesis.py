from tts_engine import TextToSpeechEngine
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict
import click
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
from pydub import AudioSegment
from tqdm import tqdm

logger = logging.getLogger(__name__)
from typing import Optional, Union, List, Dict


class SpeechSynthesizer:
    """Synthesizes speech in target language with voice matching"""

    def __init__(self, tts_engine: TextToSpeechEngine):
        self.tts = tts_engine

    def synthesize_segment(
            self,
            translated_segment: Dict,
            reference_audio: Optional[str] = None,
            target_language: str = 'pt'
    ) -> str:
        """
        Synthesize speech for a translated segment
        Args:
            translated_segment: Dict with 'text', 'start', 'end' keys
            reference_audio: Path to original audio for voice matching
            target_language: Target language code
        Returns:
            Path to synthesized audio file
        """
        text = translated_segment['text']
        duration = translated_segment['end'] - translated_segment['start']
        # Calculate speed to match original duration (basic estimate: ~150 words/min = 0.15s per word)
        estimated_duration = len(text.split()) * 0.15  # Rough estimate
        speed = max(0.5, min(2.0, duration / estimated_duration)) if estimated_duration > 0 else 1.0
        logger.info(f"Synthesizing: '{text[:50]}...' (duration: {duration:.2f}s, speed: {speed:.2f})")

        output_path = None
        try:
            output_path = self.tts.synthesize(
                text=text,
                language=target_language,
                reference_speaker=reference_audio,  # Use correct param name here
                speed=speed
            )
        except TypeError as e:
            error_msg = str(e)
            if any(kwarg in error_msg for kwarg in ["'output_path'", "'speaker_wav'"]):
                logger.warning(f"Retrying segment with fixed kwargs: {e}")
                # Retry with only supported params (no extras)
                output_path = self.tts.synthesize(
                    text=text,
                    language=target_language,
                    reference_speaker=reference_audio,
                    speed=speed
                )
            else:
                raise
        except Exception as e:
            logger.error(f"Synthesis failed for segment: {e}")
            # Fallback: Generate silence matching duration
            silence = AudioSegment.silent(duration=int(duration * 1000))
            output_path = f"dubbing_output/silence_{hash(text) % 10000}.wav"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            silence.export(output_path, format="wav")
            logger.info(f"Generated silence fallback: {output_path}")

        return output_path

    def synthesize_batch(self, segments: List[Dict], reference_audio: str, target_language: str = 'pt') -> List[str]:
        """Batch synthesize multiple segments"""
        paths = []
        for seg in tqdm(segments, desc="Synthesizing segments"):
            paths.append(self.synthesize_segment(seg, reference_audio, target_language))
        return paths