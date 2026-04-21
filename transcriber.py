# transcriber.py
import os
from openai import OpenAI

def transcribe_with_timestamps(audio_path: str) -> dict:
    """
    Uses OpenAI Whisper API (whisper-1).
    Returns:
    {
        "full_text": "...",
        "segments": [{"start": 0.0, "end": 2.1, "text": "..."}, ...],
        "second_by_second": [{"second": 0, "text": "..."}, ...]
    }
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = result.segments or []

    # Build second-by-second map (assign text only to the starting second)
    second_map = {}
    for seg in segments:
        start = int(seg.start)
        text  = seg.text.strip()
        if start not in second_map:
            second_map[start] = text
        else:
            second_map[start] += " " + text

    second_by_second = [
        {"second": sec, "text": second_map[sec]}
        for sec in sorted(second_map.keys())
    ]

    return {
        "full_text": result.text or "",
        "segments": segments,
        "second_by_second": second_by_second
    }
