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

    def synthesize_segment(self, translated_segment: Dict, reference_audio: Optional[str] = None,
                           target_language: str = 'pt') -> str:
        text = translated_segment['text']
        duration = translated_segment['end'] - translated_segment['start']
        estimated_duration = len(text) / 6.0  # Chars/sec for PT-BR
        speed = max(0.5, min(2.0, duration / estimated_duration)) if estimated_duration > 0 else 1.0

        try:
            return self.tts.synthesize(
                text=text, language=target_language, reference_speaker=reference_audio,
                speed=speed, target_duration=duration
            )
        except Exception as e:
            logger.warning(f"Primary synthesis failed: {e}; retrying without cloning")
            try:
                return self.tts.synthesize(
                    text=text, language=target_language, speed=speed, target_duration=duration
                )
            except Exception as retry_e:
                logger.error(f"Retry failed: {retry_e}; silence fallback")
                silence = AudioSegment.silent(duration=int(duration * 1000))
                output_path = f"dubbing_output/silence_{hash(text) % 10000}.wav"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                silence.export(output_path, format="wav")
                return output_path

    def synthesize_batch(self, segments: List[Dict], reference_audio: str, target_language: str = 'pt') -> List[str]:
        """Batch synthesize multiple segments"""
        paths = []
        for seg in tqdm(segments, desc="Synthesizing segments"):
            paths.append(self.synthesize_segment(seg, reference_audio, target_language))
        return paths