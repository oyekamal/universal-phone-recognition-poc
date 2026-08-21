"""Stage 1 (S2P): shared Allosaurus / PhoneticXEUS inference functions.
Used by run_report.py (scored against IPAPack++ ground truth) and
recognize_any_audio.py (no ground truth — your own audio)."""
import subprocess
import sys

import torch
import torchaudio
from transformers import AutoModel


def allosaurus_ipa(wav_path):
    out = subprocess.run(
        [sys.executable, "-m", "allosaurus.run", "-i", str(wav_path)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().replace(" ", "")


_XEUS_MODEL = None


def phoneticxeus_ipa(wav_path):
    global _XEUS_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _XEUS_MODEL is None:
        _XEUS_MODEL = AutoModel.from_pretrained(
            "changelinglab/PhoneticXeus", trust_remote_code=True
        ).eval().to(device)
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.to(device)
    with torch.no_grad():
        return _XEUS_MODEL.transcribe(waveform, sampling_rate=16000)[0]["processed_transcript"]
