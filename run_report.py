"""Fetch a real sample for a FLEURS language, run both S2P models
(Allosaurus, PhoneticXEUS), and print a PER comparison against ground
truth. This is the one script to run per language.

    ./venv/bin/python run_report.py sd_in
    ./venv/bin/python run_report.py ur_pk
    ./venv/bin/python run_report.py hi_in
"""
import subprocess
import sys

import torch
import torchaudio
from transformers import AutoModel

from eval import levenshtein
from fetch_sample import fetch


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


def per(pred, truth):
    truth_compact = truth.replace(" ", "")
    dist = levenshtein(pred, truth_compact)
    return dist, len(truth_compact), dist / max(len(truth_compact), 1)


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <fleurs_lang_code> [split] [index]")
    lang_code = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    wav_path, meta = fetch(lang_code, split, index)
    truth = meta["ground_truth_ipa"]

    print(f"=== {lang_code} ({meta['utt_id']}) ===")
    print("orthographic:", meta["orthographic"])
    print("ground truth IPA:", truth)
    print()

    allo_pred = allosaurus_ipa(wav_path)
    allo_dist, allo_len, allo_per = per(allo_pred, truth)
    print("Allosaurus IPA:  ", allo_pred)
    print(f"Allosaurus PER (char-level, proxy): {allo_dist}/{allo_len} = {allo_per:.1%}")
    print()

    xeus_pred = phoneticxeus_ipa(wav_path)
    xeus_dist, xeus_len, xeus_per = per(xeus_pred, truth)
    print("PhoneticXEUS IPA:", xeus_pred)
    print(f"PhoneticXEUS PER (char-level, proxy): {xeus_dist}/{xeus_len} = {xeus_per:.1%}")


if __name__ == "__main__":
    main()
