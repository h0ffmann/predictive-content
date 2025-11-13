"""
Translation Augmentor: Uses LLM to make translations more cohesive/natural.
"""

import os
import logging
from typing import List, Dict, Any
import tiktoken  # For token counting
from openai import OpenAI  # OpenRouter compatible

logger = logging.getLogger(__name__)

class TranslationAugmentor:
    def __init__(self, max_chunk_tokens: int = 8000):
        self.max_chunk_tokens = max_chunk_tokens
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),  # Optional for free
        )
        self.model = "meta-llama/llama-3.3-8b-instruct:free"  # Reliable free model
        self.original_segments = []  # Store for fallback

    def augment_transcript(self, translated_segments: List[Dict], original_segments: List[Dict], target_lang: str) -> List[Dict]:
        """
        Augment translated segments for cohesion.
        """
        self.original_segments = original_segments  # For fallback
        if os.getenv('SKIP_AUGMENT', 'false').lower() == 'true':
            logger.info("Skipping augmentation")
            return translated_segments

        # Chunk by tokens/duration
        chunks = self._chunk_transcript(translated_segments)
        augmented_chunks = []
        for chunk in chunks:
            try:
                augmented = self._augment_chunk(chunk, target_lang)
                augmented_chunks.append(augmented)
            except Exception as e:
                logger.error(f"Augmentation failed for chunk: {e}")
                augmented_chunks.append([])  # Empty for fallback

        # Stitch with overlap handling
        return self._stitch_chunks(augmented_chunks)

    def _chunk_transcript(self, segments: List[Dict]) -> List[List[Dict]]:
        """Chunk segments by max tokens and ~300s duration."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_duration = 0
        max_duration_per_chunk = 300  # Seconds

        encoding = tiktoken.get_encoding("cl100k_base")

        for seg in segments:
            seg_tokens = len(encoding.encode(seg['text']))
            seg_dur = seg['end'] - seg['start']

            if current_duration + seg_dur > max_duration_per_chunk or current_tokens + seg_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [seg]
                current_tokens = seg_tokens
                current_duration = seg_dur
            else:
                current_chunk.append(seg)
                current_tokens += seg_tokens
                current_duration += seg_dur

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _augment_chunk(self, chunk: List[Dict], target_lang: str) -> List[Dict]:
        """Augment a chunk with LLM."""
        # Build prompt for cohesion
        original_text = " ".join(seg['text'] for seg in chunk)
        prompt = f"""
        Rewrite the following {target_lang} translation chunk to make it more natural, cohesive, and conversational while preserving the original meaning, timing, and structure. 
        Keep it as a continuous script. Do not add/remove content.

        Original chunk: {original_text}

        Augmented version:
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3,
        )

        augmented_text = response.choices[0].message.content.strip()

        # Split back into segments (simple: approximate by original lengths)
        # For production, use better alignment (e.g., via difflib)
        total_orig_dur = sum(seg['end'] - seg['start'] for seg in chunk)
        augmented_segments = []
        for i, seg in enumerate(chunk):
            ratio = (seg['end'] - seg['start']) / total_orig_dur
            seg_text = augmented_text[:int(len(augmented_text) * ratio)]  # Rough split
            augmented_text = augmented_text[int(len(augmented_text) * ratio):]
            augmented_segments.append({'text': seg_text.strip(), 'start': seg['start'], 'end': seg['end']})

        logger.info(f"Augmented chunk: {len(chunk)} segments")
        return augmented_segments

    def _stitch_chunks(self, augmented_chunks: List[List[Dict]]) -> List[Dict]:
        """Stitch augmented chunks, handling overlaps/failures."""
        if not augmented_chunks or all(len(c) == 0 for c in augmented_chunks):
            logger.warning("No augmented chunks; returning originals")
            return [{'text': seg['translated'], 'start': seg['start'], 'end': seg['end']} for seg in self.original_segments]

        stitched = []
        for i, chunk in enumerate(augmented_chunks):
            if len(chunk) == 0:  # Failed chunk
                # Insert originals for this chunk's range (approximate)
                continue
            # Overlap logic: Trim if short (your original <10 words check)
            if i > 0 and len(chunk) > 0 and len(chunk[0]['text'].split()) < 10:
                # Overlap trim: Remove last 5-10% of prev, add here
                if stitched:
                    prev_text = stitched[-1]['text']
                    stitched[-1]['text'] = prev_text[:-len(prev_text)//10]
            stitched.extend(chunk)

        # Ensure full length
        if len(stitched) < len(self.original_segments):
            # Pad with originals
            for seg in self.original_segments[len(stitched):]:
                stitched.append({'text': seg['translated'], 'start': seg['start'], 'end': seg['end']})

        logger.info(f"Stitched {len(stitched)} augmented segments")
        return stitched