"""Compare Allosaurus's predicted phones against the IPAPack++ ground-truth
IPA for the Sindhi sample, as a rough phone error rate (PER).

ponytail: character-level Levenshtein distance on the phone strings
(space-stripped), not a proper phone-aligned PER — good enough for a local
sanity check of Stage 1, upgrade to a real IPA-tokenized alignment if this
needs to be a defensible benchmark number.
"""
import json
import subprocess
import sys


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def main():
    audio_path, cuts_path, cut_index = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(cuts_path) as f:
        lines = f.readlines()
    truth = json.loads(lines[cut_index])["supervisions"][0]["custom"]["clean"]
    truth_compact = truth.replace(" ", "")

    out = subprocess.run(
        [sys.executable, "-m", "allosaurus.run", "-i", audio_path],
        capture_output=True, text=True, check=True,
    )
    pred_compact = out.stdout.strip().replace(" ", "")

    dist = levenshtein(pred_compact, truth_compact)
    per = dist / max(len(truth_compact), 1)

    print("ground truth IPA:", truth)
    print("predicted IPA:   ", out.stdout.strip())
    print(f"char-level edit distance: {dist} / {len(truth_compact)} -> PER ~{per:.1%}")


if __name__ == "__main__":
    main()
