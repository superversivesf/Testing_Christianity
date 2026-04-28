#!/usr/bin/env python3
"""Transcribe all .mp4 files in this directory with faster-whisper.

Outputs .txt (plain) and .srt (timestamped) into transcripts/.
Skips files that already have both outputs.
"""
import sys
from pathlib import Path
from faster_whisper import WhisperModel

HERE = Path(__file__).resolve().parent
TRANSCRIPTS = HERE / "transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_model():
    # Try large-v3 with int8_float16 first; fall back to medium on OOM.
    for model_size, compute in [("large-v3", "int8_float16"), ("medium", "int8_float16")]:
        try:
            print(f"[loader] trying {model_size} on cuda ({compute})", flush=True)
            return WhisperModel(model_size, device="cuda", compute_type=compute), model_size
        except Exception as e:
            print(f"[loader] {model_size} failed: {e}", flush=True)
    raise RuntimeError("could not load any whisper model")


def transcribe_one(model, mp4: Path):
    stem = mp4.stem
    txt_path = TRANSCRIPTS / f"{stem}.txt"
    srt_path = TRANSCRIPTS / f"{stem}.srt"
    if txt_path.exists() and srt_path.exists() and txt_path.stat().st_size > 0:
        print(f"[skip] {stem} (already transcribed)", flush=True)
        return

    print(f"[transcribe] {stem}", flush=True)
    segments, info = model.transcribe(
        str(mp4),
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    txt_lines = []
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        txt_lines.append(text)
        srt_lines.append(
            f"{i}\n{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}\n{text}\n"
        )
        if i % 50 == 0:
            print(f"  ... {i} segments, t={seg.end:.0f}s", flush=True)

    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    print(f"[done] {stem} ({len(txt_lines)} segments)", flush=True)


def main():
    mp4s = sorted(HERE.glob("*.mp4"))
    if not mp4s:
        print("no .mp4 files yet", flush=True)
        return 1

    model, size = load_model()
    print(f"[loader] loaded {size}", flush=True)

    for mp4 in mp4s:
        try:
            transcribe_one(model, mp4)
        except Exception as e:
            print(f"[error] {mp4.name}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
