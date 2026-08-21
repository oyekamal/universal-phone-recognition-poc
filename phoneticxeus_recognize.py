"""Stage 1 (S2P) via PhoneticXEUS — the real SOTA model, trained on
IPAPack++ (which includes Sindhi), unlike Allosaurus (12 languages only).

Quick-inference path from the model card: transformers AutoModel,
no repo clone needed.
"""
import sys
import torch
import torchaudio
from transformers import AutoModel


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <audio.wav>")
    audio_path = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        "changelinglab/PhoneticXeus", trust_remote_code=True
    ).eval().to(device)

    waveform, sr = torchaudio.load(audio_path)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.to(device)

    with torch.no_grad():
        results = model.transcribe(waveform, sampling_rate=16000)

    print("IPA (joined):", results[0]["processed_transcript"])
    print("IPA (phones):", results[0]["predicted_transcript"])


if __name__ == "__main__":
    main()
