"""Stage 1 (S2P) + naive Stage 2 (P2T) local test.

Real pipeline per the research notebook:
  S2P: Allosaurus/PhoneticXEUS  audio -> universal IPA phone sequence
  P2T: fine-tuned T5/LLM        IPA sequence -> fluent English text

We only have Stage 1 locally (Allosaurus, pip-installable, no training
needed). Stage 2 here is a placeholder naive IPA->Latin substitution, NOT
the real P2T model — good enough to see the pipeline run end to end.

ponytail: naive substitution dict instead of the T5/LoRA P2T model from
the paper. Upgrade to that (or an LLM prompted with the IPA string) once
Stage 1 output looks right.
"""
import subprocess
import sys

# crude IPA -> readable-Latin fallback, not a real P2T model
IPA_TO_LATIN = {
    "ʃ": "sh", "ʒ": "zh", "θ": "th", "ð": "dh", "ŋ": "ng",
    "ɑ": "a", "ɔ": "o", "ɛ": "e", "ɪ": "i", "ʊ": "u", "ə": "a",
    "ʔ": "'", "χ": "kh", "ʁ": "r", "ɣ": "gh", "ɦ": "h",
}


def naive_p2t(ipa_phones):
    return "".join(IPA_TO_LATIN.get(p, p) for p in ipa_phones)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <audio.wav|mp3>")
    audio_path = sys.argv[1]

    out = subprocess.run(
        [sys.executable, "-m", "allosaurus.run", "-i", audio_path],
        capture_output=True, text=True, check=True,
    )
    phones = out.stdout.strip().split()
    print("IPA phones:", " ".join(phones))
    print("naive transliteration:", naive_p2t(phones))


if __name__ == "__main__":
    main()
