#!/usr/bin/env python3
"""
translation.py

Simplified translation script using the DeepL Python client library.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import deepl 

OCR_JSON_PATH = "ocr_results_image_0_7.json"
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY")

if not DEEPL_API_KEY:
    raise RuntimeError("Please set the DEEPL_API_KEY environment variable")

# Instantiate the DeepL client
deepl_client = deepl.DeepLClient(DEEPL_API_KEY)


@dataclass
class Segment:
    text: str
    score: float
    box: Tuple[int, int, int, int]


def load_segments(path: str = OCR_JSON_PATH) -> List[Segment]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = []
    for item in data:
        segments.append(
            Segment(
                text=item.get("text", "").strip(),
                score=float(item.get("score", 0.0)),
                box=tuple(item.get("box", [0, 0, 0, 0]))
            )
        )
    return segments


def sort_segments(segments: List[Segment]) -> List[Segment]:
    return sorted(segments, key=lambda s: (s.box[1], s.box[0]))


def group_segments(segments: List[Segment], strategy: str = "lines") -> List[str]:
    sorted_segs = sort_segments(segments)
    if strategy == "page":
        combined = "\n".join(seg.text for seg in sorted_segs if seg.text)
        return [combined] if combined else []
    # default: every segment as one chunk
    return [seg.text for seg in sorted_segs if seg.text]


def translate_chunks(chunks: List[str], target_lang: str = "KO") -> List[str]:
    if not chunks:
        return []
    results = deepl_client.translate_text(chunks, target_lang=target_lang)
    # results might be list or single object depending on input
    translated = [res.text for res in results]
    return translated


def main():
    print("=== DeepL Translation Test ===")
    print(f"Loading OCR segments from {OCR_JSON_PATH}")
    segments = load_segments(OCR_JSON_PATH)
    print(f"Loaded {len(segments)} segments.")
    strategy = "lines"  # or "page"
    chunks = group_segments(segments, strategy=strategy)
    print(f"Translating {len(chunks)} chunks (strategy={strategy}) to KO …")
    translations = translate_chunks(chunks, target_lang="KO")
    for i, (src, tgt) in enumerate(zip(chunks, translations), start=1):
        print(f"--- CHUNK {i}/{len(chunks)} ---")
        print("SOURCE:", src)
        print("TRANSLATED:", tgt)
        print()

if __name__ == "__main__":
    main()
