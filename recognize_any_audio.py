"""The actual point of this repo: audio in, meaningful text out. No
ground truth, no IPAPack++, no dataset — just your own audio file.

    ./venv/bin/python recognize_any_audio.py path/to/your.wav
    ./venv/bin/python recognize_any_audio.py path/to/your.wav --lang Pashto

Pipeline:
  1. S2P (Allosaurus + PhoneticXEUS): audio -> raw IPA phone string
  2. romanize.py: IPA -> readable Roman letters (deterministic, no meaning)
  3. P2T (LLM via `claude -p`): IPA -> attempted real sentence + English translation

There is NO ground truth here — this is the real "just an audio file"
case you'll be testing with. Because of that, there is no PER number to
report, and there is no automatic way to tell you whether the P2T output
is actually correct. The one signal you get is the LLM's own self-reported
confidence (high/medium/low) — and per the Sindhi/Urdu experiments in
README.md, when the underlying S2P phone error rate is high (as it often
will be for a language neither model was trained heavily on), that
confidence is honestly `low` and the reconstruction is wrong. Treat `low`
confidence as "not usable," not as "roughly right."
"""
import argparse
import json
import subprocess
from pathlib import Path

from llm_p2t import build_prompt, run_p2t
from romanize import romanize
from s2p import allosaurus_ipa, phoneticxeus_ipa

REPORTS = Path(__file__).parent / "reports"


def to_16k_mono_wav(audio_path):
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    wav_path = audio_path.with_suffix(".16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        check=True, capture_output=True,
    )
    return wav_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--lang", default="unknown (guess from the audio)",
                         help="language name/hint, e.g. 'Pashto' — helps the P2T LLM step, not required")
    args = parser.parse_args()

    wav_path = to_16k_mono_wav(args.audio_path)

    print("############################  INPUT  ############################")
    print(f"audio file: {args.audio_path}")
    print(f"language hint: {args.lang}")
    print()

    allo_ipa = allosaurus_ipa(wav_path)
    xeus_ipa = phoneticxeus_ipa(wav_path)
    allo_roman, _ = romanize(allo_ipa)
    xeus_roman, _ = romanize(xeus_ipa)

    print("############################  STAGE 1 OUTPUT (S2P, no ground truth to score against)  ############################")
    print("[Allosaurus IPA]      ", allo_ipa)
    print("[Allosaurus romanized]", allo_roman)
    print()
    print("[PhoneticXEUS IPA]      ", xeus_ipa)
    print("[PhoneticXEUS romanized]", xeus_roman)
    print()

    print("############################  STAGE 2 OUTPUT (P2T: attempted meaningful text)  ############################")
    result = {"audio_file": str(args.audio_path), "lang_hint": args.lang}
    for model_name, ipa in (("phoneticxeus", xeus_ipa), ("allosaurus", allo_ipa)):
        prompt = build_prompt(args.lang, ipa)
        reply = run_p2t(prompt)
        print(f"--- from {model_name} ---")
        print(reply if reply else "(no claude CLI / API key available — see llm_p2t.py)")
        print()
        result[f"{model_name}_ipa"] = ipa
        result[f"{model_name}_romanized"] = romanize(ipa)[0]
        result[f"{model_name}_p2t_raw"] = reply

    REPORTS.mkdir(exist_ok=True)
    out_path = REPORTS / f"{Path(args.audio_path).stem}_p2t_report.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    print()
    print("Reminder: there is no ground truth for this audio. Trust the LLM's own")
    print("stated confidence, not the fluency of its guess — a fluent wrong guess")
    print("is worse than a hedged one. See README.md 'Plain English, first' section")
    print("for what happens when Stage 1 phone error rate is high.")


if __name__ == "__main__":
    main()
