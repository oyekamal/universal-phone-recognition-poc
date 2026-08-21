"""Same PER check as eval.py, for PhoneticXEUS instead of Allosaurus."""
import json
import sys

import torch
import torchaudio
from transformers import AutoModel

from eval import levenshtein


def main():
    audio_path, cuts_path, cut_index = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(cuts_path) as f:
        truth = json.loads(f.readlines()[cut_index])["supervisions"][0]["custom"]["clean"]
    truth_compact = truth.replace(" ", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("changelinglab/PhoneticXeus", trust_remote_code=True).eval().to(device)
    waveform, sr = torchaudio.load(audio_path)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.to(device)

    with torch.no_grad():
        pred_compact = model.transcribe(waveform, sampling_rate=16000)[0]["processed_transcript"]

    dist = levenshtein(pred_compact, truth_compact)
    per = dist / max(len(truth_compact), 1)
    print("ground truth IPA:", truth)
    print("predicted IPA:   ", pred_compact)
    print(f"char-level edit distance: {dist} / {len(truth_compact)} -> PER ~{per:.1%}")


if __name__ == "__main__":
    main()
