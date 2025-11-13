#!/usr/bin/env python3
"""
Duber - Video Dubbing Pipeline
Converts video speech from one language to another with synchronized audio
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union, List

import click
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
from pydub import AudioSegment
from tqdm import tqdm

from translation_augmentation import TranslationAugmentor
from tts_engine import TextToSpeechEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_ffmpeg():
    """Check if ffmpeg is installed"""
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        logger.error("❌ FFmpeg not found! Please install it:")
        logger.error("   Ubuntu/Debian: sudo apt-get install ffmpeg")
        logger.error("   macOS: brew install ffmpeg")
        logger.error("   Or visit: https://ffmpeg.org/download.html")
        sys.exit(1)
    logger.info(f"✅ FFmpeg found at: {ffmpeg_path}")
    return ffmpeg_path


class SpeechToTextEngine:
    """Handles speech-to-text transcription using Faster Whisper"""

    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        """
        Initialize the speech-to-text engine

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run on (cuda, cpu, or auto)
        """
        logger.info(f"Loading Whisper model '{model_size}'...")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )
        logger.info(f"Whisper model loaded on {device}")

    def transcribe(self, audio_path: str, language: str = "en") -> list:
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file
            language: Source language code

        Returns:
            List of transcription segments with timing info
        """
        logger.info(f"Transcribing {audio_path}...")
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True  # Enable word-level timestamps for better alignment
        )

        results = []
        for segment in segments:
            results.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })

        logger.info(
            f"Transcription complete. Detected language: {info.language} (probability: {info.language_probability:.2f})")
        logger.info(f"Found {len(results)} segments")
        return results


class TranslationEngine:
    """Handles text translation using Helsinki-NLP models"""

    def __init__(self, source_lang: str = "en", target_lang: str = "pt"):
        """
        Initialize translation engine

        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        # Map language codes to model names
        model_map = {
            ('en', 'pt'): ('Helsinki-NLP/opus-mt-tc-big-en-pt', '>>por<<'),
            ('en', 'es'): ('Helsinki-NLP/opus-mt-en-es', '>>es<<'),
            ('en', 'fr'): ('Helsinki-NLP/opus-mt-en-fr', '>>fr<<'),
            ('en', 'de'): ('Helsinki-NLP/opus-mt-en-de', '>>de<<'),
            ('es', 'en'): ('Helsinki-NLP/opus-mt-es-en', '>>en<<'),
            ('fr', 'en'): ('Helsinki-NLP/opus-mt-fr-en', '>>en<<'),
            ('pt', 'en'): ('Helsinki-NLP/opus-mt-tc-big-pt-en', '>>eng<<'),
        }

        lang_pair = (source_lang, target_lang)
        if lang_pair in model_map:
            model_name, self.lang_token = model_map[lang_pair]
        else:
            # Fallback to generic model
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            self.lang_token = f">>{target_lang}<<"
            logger.warning(f"Using fallback model: {model_name}")

        logger.info(f"Loading translation model: {model_name}...")

        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)

            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            logger.info(f"Translation model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def translate(self, text: str) -> str:
        """
        Translate text

        Args:
            text: Text to translate

        Returns:
            Translated text
        """
        if not text.strip():
            return ""

        # Add language token prefix for multilingual models
        input_text = f"{self.lang_token} {text}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Translate
        with torch.no_grad():
            translated = self.model.generate(**inputs, max_length=512, num_beams=4)

        # Decode
        result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return result.strip()

    def translate_batch(self, texts: list) -> list:
        """
        Translate multiple texts efficiently

        Args:
            texts: List of texts to translate

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        # Filter empty texts
        non_empty_indices = [i for i, text in enumerate(texts) if text.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return [""] * len(texts)

        # Add language token prefix to all texts
        input_texts = [f"{self.lang_token} {text}" for text in non_empty_texts]

        # Tokenize batch
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Translate
        with torch.no_grad():
            translated = self.model.generate(**inputs, max_length=512, num_beams=4)

        # Decode all
        results = [
            self.tokenizer.decode(t, skip_special_tokens=True).strip()
            for t in translated
        ]

        # Reconstruct full list with empty strings
        full_results = [""] * len(texts)
        for idx, result in zip(non_empty_indices, results):
            full_results[idx] = result

        return full_results


class DubbingPipeline:
    """Complete dubbing pipeline orchestrator"""

    def __init__(
            self,
            output_dir: str = "dubbing_output",
            whisper_model: str = "large-v3",
            source_lang: str = "en",
            target_lang: str = "pt",
            speaker_wav: Optional[str] = None
    ):
        """
        Initialize the dubbing pipeline

        Args:
            output_dir: Directory for output files
            whisper_model: Whisper model size
            source_lang: Source language code
            target_lang: Target language code
            speaker_wav: Path to custom reference audio for voice cloning
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "segments").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)

        # Initialize engines
        self.stt = SpeechToTextEngine(model_size=whisper_model)
        self.translator = TranslationEngine(source_lang=source_lang, target_lang=target_lang)
        self.tts = TextToSpeechEngine(language=target_lang)  # Pass language for PT model

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.speaker_wav = speaker_wav

    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file using ffmpeg

        Args:
            video_path: Path to video file

        Returns:
            Path to extracted audio file
        """
        import subprocess

        audio_path = self.output_dir / "extracted_audio.wav"

        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            str(audio_path)
        ]

        try:
            logger.info("Extracting audio from video...")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✅ Audio extracted to {audio_path}")
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FFmpeg failed: {e.stderr}")
            raise RuntimeError("Failed to extract audio from video")

    def extract_reference_clips(self, audio_path: str, segments: list) -> List[str]:
        """
        Extract reference audio clips for voice cloning

        Args:
            audio_path: Path to source audio
            segments: List of transcription segments

        Returns:
            List of paths to reference clips
        """
        if self.speaker_wav:
            logger.info(f"Using custom speaker reference: {self.speaker_wav}")
            return [self.speaker_wav]

        logger.info("Extracting reference clips for voice cloning...")
        reference_paths = []

        # Load full audio
        try:
            full_audio = AudioSegment.from_wav(audio_path)
        except Exception as e:
            logger.warning(f"Could not load audio for reference extraction: {e}")
            return []

        # Find suitable segments (3-10 seconds of clean speech)
        suitable_segments = [
            seg for seg in segments
            if 3 <= (seg['end'] - seg['start']) <= 10
        ]

        # Sort by duration (longest first) and take top 5
        suitable_segments = sorted(
            suitable_segments,
            key=lambda s: s['end'] - s['start'],
            reverse=True
        )[:5]

        if not suitable_segments:
            logger.warning("⚠️  No suitable reference segments found (need 3-10s clean speech)")
            logger.warning("Voice cloning will use default voice")
            return []

        # Extract clips
        for i, seg in enumerate(suitable_segments):
            ref_path = self.output_dir / "processed" / f"reference_clip_{i:02d}.wav"
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)

            clip = full_audio[start_ms:end_ms]
            clip = clip.set_channels(1).set_frame_rate(22050)
            clip.export(str(ref_path), format="wav")

            reference_paths.append(str(ref_path))
            logger.info(f"  ✓ Reference {i + 1}: {seg['end'] - seg['start']:.2f}s")

        return reference_paths

    def process_video(self, video_path: str) -> dict:
        """
        Process entire video through the dubbing pipeline

        Args:
            video_path: Path to input video

        Returns:
            Dictionary with results and output paths
        """
        logger.info("=" * 70)
        logger.info("🎬 Starting Dubbing Pipeline")
        logger.info("=" * 70)

        # Step 1: Extract audio
        logger.info("\n📼 Step 1/5: Extracting audio from video...")
        audio_path = self.extract_audio(video_path)

        # Step 2: Transcribe
        logger.info("\n🎤 Step 2/5: Transcribing audio...")
        segments = self.stt.transcribe(audio_path, language=self.source_lang)

        if not segments:
            raise RuntimeError("No speech detected in audio")

        # Save transcription
        transcript_path = self.output_dir / "transcript.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}\n")
        logger.info(f"✅ Transcription saved: {transcript_path}")

        # Step 2.5: Extract reference clips
        logger.info("\n🎵 Step 2.5/5: Extracting voice reference clips...")
        reference_paths = self.extract_reference_clips(audio_path, segments)
        if reference_paths:
            logger.info(f"✅ Extracted {len(reference_paths)} reference clips")
        else:
            logger.info("Using default voice (no suitable references found)")

        # Step 3: Translate
        logger.info(f"\n🌐 Step 3/5: Translating {len(segments)} segments...")
        texts_to_translate = [seg['text'] for seg in segments]
        translated_texts = self.translator.translate_batch(texts_to_translate)

        # Update segments with translations
        for seg, translated in zip(segments, translated_texts):
            seg['translated'] = translated

        # Save translations
        translation_path = self.output_dir / "translations.txt"
        with open(translation_path, 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]\n")
                f.write(f"  Original:   {seg['text']}\n")
                f.write(f"  Translated: {seg['translated']}\n\n")
        logger.info(f"✅ Translations saved: {translation_path}")

        # Step 3.5: Augment translations with LLM
        logger.info("🎯 Step 3.5/5: Augmenting translations with LLM...")
        # Prepare translated_segments for augmentor (extract translated text with timings)
        translated_segments = [
            {'text': seg['translated'], 'start': seg['start'], 'end': seg['end']}
            for seg in segments
        ]
        try:
            augmentor = TranslationAugmentor()  # No arg until fixed
            augmented_segments = augmentor.augment_transcript(
                translated_segments, segments, self.target_lang
            )
        except Exception as e:
            logger.warning(f"Augmentation failed ({e}); using raw translations")
            augmented_segments = translated_segments  # Fallback

        # Update original segments with augmented text
        for i, aug_seg in enumerate(augmented_segments):
            segments[i]['translated'] = aug_seg['text']
        logger.info("✅ Augmented cohesive translations")

        # Save augmented translations (overwrite with augmented)
        with open(translation_path, 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]\n")
                f.write(f"  Original:   {seg['text']}\n")
                f.write(f"  Augmented:  {seg['translated']}\n\n")
        logger.info(f"✅ Augmented translations saved: {translation_path}")

        # Step 4: Synthesize speech
        logger.info("\n🎙️  Step 4/5: Synthesizing dubbed audio...")
        dubbed_segments = []
        segments_dir = self.output_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        # Pick first reference if available (tts expects str or None)
        ref_audio = reference_paths[0] if reference_paths else None

        for i, seg in enumerate(tqdm(segments, desc="Generating speech")):
            audio_file = segments_dir / f"segment_{i:04d}.wav"

            try:
                self.tts.synthesize(
                    text=seg['translated'],
                    output_path=str(audio_file),
                    language=self.target_lang,
                    speaker_wav=ref_audio  # Single reference path
                )
            except Exception as e:
                logger.error(f"Failed to synthesize segment {i}: {e}")
                # Create silent segment as fallback
                silence_duration = int((seg['end'] - seg['start']) * 1000)
                silence = AudioSegment.silent(duration=silence_duration)
                silence.export(str(audio_file), format="wav")
                logger.warning(f"  → Used silence fallback for segment {i}")

            dubbed_segments.append({
                'audio_path': str(audio_file),
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['end'] - seg['start']
            })

        logger.info(f"✅ Generated {len(dubbed_segments)} audio segments")

        # Step 5: Combine audio segments
        logger.info("\n🎼 Step 5/5: Combining audio segments...")
        final_audio = self.combine_audio_segments(dubbed_segments)
        final_audio_path = self.output_dir / "dubbed_audio.wav"
        final_audio.export(str(final_audio_path), format="wav")
        logger.info(f"✅ Final dubbed audio: {final_audio_path}")

        # Cleanup temporary reference clips
        for ref_path in reference_paths:
            if "reference_clip" in str(ref_path) and os.path.exists(ref_path):
                try:
                    os.remove(ref_path)
                except:
                    pass

        logger.info("\n" + "=" * 70)
        logger.info("✅ Dubbing Pipeline Complete!")
        logger.info("=" * 70)

        return {
            'transcript': str(transcript_path),
            'translation': str(translation_path),
            'dubbed_audio': str(final_audio_path),
            'segments': segments,
            'segment_count': len(segments)
        }

    def combine_audio_segments(self, segments: list) -> AudioSegment:
        """
        Combine individual audio segments with proper timing and alignment

        Args:
            segments: List of segment dicts with timing and audio paths

        Returns:
            Combined AudioSegment
        """
        if not segments:
            return AudioSegment.silent(duration=1000)

        # Create silent audio for the full duration
        last_end = max(seg['end'] for seg in segments)
        combined = AudioSegment.silent(duration=int(last_end * 1000) + 1000)  # +1s padding

        sorted_segments = sorted(segments, key=lambda s: s['start'])

        for i, seg in enumerate(sorted_segments):
            try:
                audio = AudioSegment.from_wav(seg['audio_path'])
            except Exception as e:
                logger.error(f"Failed to load segment {i}: {e}")
                continue

            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            original_duration_ms = end_ms - start_ms
            current_duration_ms = len(audio)

            # Time-stretch if synthesized audio duration differs significantly
            duration_diff_ms = abs(current_duration_ms - original_duration_ms)
            if duration_diff_ms > 1000:  # Increased tolerance for PT (non-English timing variances)
                target_duration = original_duration_ms / 1000.0
                stretch_ratio = current_duration_ms / original_duration_ms
                logger.debug(
                    f"Segment {i}: Stretching from {current_duration_ms / 1000:.2f}s "
                    f"to {target_duration:.2f}s (ratio: {stretch_ratio:.2f})"
                )

                try:
                    audio = self.stretch_audio_with_ffmpeg(audio, target_duration)
                    current_duration_ms = len(audio)
                except Exception as e:
                    logger.warning(f"Time stretching failed for segment {i}: {e}")

            # Check for overlap with next segment
            if i < len(sorted_segments) - 1:
                next_start_ms = int(sorted_segments[i + 1]['start'] * 1000)
                segment_end_ms = start_ms + current_duration_ms
                overlap_ms = segment_end_ms - next_start_ms

                if overlap_ms > 100:  # Significant overlap
                    logger.debug(f"Segment {i} overlaps with next by {overlap_ms}ms - applying fade")
                    crossfade_duration = min(overlap_ms, 300)  # Max 300ms crossfade
                    audio = audio.fade_out(duration=crossfade_duration)

            # Apply subtle fade in/out for smooth transitions
            fade_duration = min(50, len(audio) // 4)  # Max 50ms or 25% of audio
            audio = audio.fade_in(duration=fade_duration).fade_out(duration=fade_duration)

            # Overlay at correct position
            combined = combined.overlay(audio, position=start_ms)

            logger.debug(f"Segment {i}: Placed at {start_ms}ms (duration: {len(audio)}ms)")

        return combined

    def stretch_audio_with_ffmpeg(self, audio: AudioSegment, target_duration: float) -> AudioSegment:
        """
        Stretch audio duration using FFmpeg's atempo filter (pitch-preserving)
        Handles extreme tempo changes by chaining multiple atempo filters

        Args:
            audio: Input AudioSegment
            target_duration: Target duration in seconds

        Returns:
            Stretched AudioSegment
        """
        import subprocess
        import tempfile

        current_duration = len(audio) / 1000.0
        tempo = current_duration / target_duration

        # FFmpeg atempo is limited to 0.5-2.0 range
        # For extreme changes, chain multiple filters
        atempo_filters = []
        remaining_tempo = tempo

        # Handle tempo > 2.0 (speed up significantly)
        while remaining_tempo > 2.0:
            atempo_filters.append('atempo=2.0')
            remaining_tempo /= 2.0

        # Handle tempo < 0.5 (slow down significantly)
        while remaining_tempo < 0.5:
            atempo_filters.append('atempo=0.5')
            remaining_tempo /= 0.5

        # Add final adjustment
        if abs(remaining_tempo - 1.0) > 0.01:  # Only if not essentially 1.0
            atempo_filters.append(f'atempo={remaining_tempo:.4f}')

        if not atempo_filters:
            return audio  # No stretching needed

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file, \
                tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:

            try:
                # Export input audio
                audio.export(input_file.name, format='wav')

                # Build filter chain
                filter_string = ','.join(atempo_filters)

                cmd = [
                    'ffmpeg',
                    '-i', input_file.name,
                    '-filter:a', filter_string,
                    '-y',
                    output_file.name
                ]

                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Load stretched audio
                stretched_audio = AudioSegment.from_wav(output_file.name)
                return stretched_audio

            finally:
                # Cleanup temp files
                try:
                    os.unlink(input_file.name)
                except:
                    pass
                try:
                    os.unlink(output_file.name)
                except:
                    pass


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option(
    '--output-dir',
    '-o',
    default='dubbing_output',
    help='Output directory for dubbed content'
)
@click.option(
    '--whisper-model',
    '-w',
    default='large-v3',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large-v3']),
    help='Whisper model size (larger = more accurate but slower)'
)
@click.option(
    '--source-lang',
    '-s',
    default='en',
    help='Source language code (e.g., en, es, fr, pt)'
)
@click.option(
    '--target-lang',
    '-t',
    default='pt',
    help='Target language code (e.g., pt, es, fr, de)'
)
@click.option(
    '--speaker-wav',
    default=None,
    type=click.Path(exists=True),
    help='Path to custom reference audio for voice cloning (overrides auto-extraction)'
)
def main(video_path, output_dir, whisper_model, source_lang, target_lang, speaker_wav):
    """
    Dub a video from one language to another with voice cloning.

    VIDEO_PATH: Path to the input video file

    Examples:
        \b
        # Basic usage (English to Portuguese)
        duber my_video.mp4

        \b
        # English to Spanish with custom output directory
        duber my_video.mp4 --target-lang es --output-dir spanish_dub

        \b
        # With custom voice reference and smaller model
        duber my_video.mp4 -w medium -s en -t pt --speaker-wav voice_sample.wav

        \b
        # French to English
        duber french_video.mp4 -s fr -t en
    """
    try:
        # Check FFmpeg availability
        check_ffmpeg()

        logger.info("")
        logger.info("=" * 70)
        logger.info("                    DUBER - Video Dubbing Pipeline")
        logger.info("=" * 70)
        logger.info(f"  Video:             {video_path}")
        logger.info(f"  Source Language:   {source_lang}")
        logger.info(f"  Target Language:   {target_lang}")
        logger.info(f"  Whisper Model:     {whisper_model}")
        logger.info(f"  Output Directory:  {output_dir}")
        if speaker_wav:
            logger.info(f"  Custom Voice:      {speaker_wav}")
        logger.info("=" * 70)
        logger.info("")

        # Initialize pipeline
        pipeline = DubbingPipeline(
            output_dir=output_dir,
            whisper_model=whisper_model,
            source_lang=source_lang,
            target_lang=target_lang,
            speaker_wav=speaker_wav
        )

        # Process video
        results = pipeline.process_video(video_path)

        # Print results summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  📝 Transcript:       {results['transcript']}")
        logger.info(f"  🌐 Translations:     {results['translation']}")
        logger.info(f"  🎵 Dubbed Audio:     {results['dubbed_audio']}")
        logger.info(f"  📊 Segments:         {results['segment_count']}")
        logger.info("=" * 70)
        logger.info("")
        logger.info("✅ SUCCESS! Your dubbed audio is ready.")
        logger.info(f"   Next step: Use video editing software to replace the original audio")
        logger.info(f"   with: {results['dubbed_audio']}")
        logger.info("")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()