"""Pull one real speech sample + ground-truth IPA for any FLEURS language
from IPAPack++ (the dataset PhoneticXEUS itself was trained on).

ponytail: downloads the whole shard (one dev/test tar per language, tens
of MB) to grab a single clip — there's no per-clip HTTP range endpoint on
HF for webdataset tars. Fine for a handful of languages; switch to
`datasets` streaming mode if this needs to scale to dozens.
"""
import gzip
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = "anyspeech/ipapack_plus_2"
RAW = Path(__file__).parent / "raw"
SAMPLES = Path(__file__).parent / "samples"


def fetch(lang_code, split="test", index=0):
    """lang_code is a FLEURS code, e.g. 'sd_in', 'ur_pk', 'hi_in'."""
    RAW.mkdir(exist_ok=True)
    SAMPLES.mkdir(exist_ok=True)
    shard = f"fleurs_shar/{lang_code}-{split}"
    cuts_gz = RAW / f"{lang_code}_cuts.jsonl.gz"
    cuts = RAW / f"{lang_code}_cuts.jsonl"
    tar_path = RAW / f"{lang_code}_recording.tar"

    if not cuts.exists():
        subprocess.run(
            ["curl", "-sL", f"https://huggingface.co/datasets/{REPO}/resolve/main/{shard}/cuts.000000.jsonl.gz",
             "-o", str(cuts_gz)], check=True)
        with gzip.open(cuts_gz, "rb") as f_in, open(cuts, "wb") as f_out:
            f_out.write(f_in.read())
    if not tar_path.exists():
        subprocess.run(
            ["curl", "-sL", f"https://huggingface.co/datasets/{REPO}/resolve/main/{shard}/recording.000000.tar",
             "-o", str(tar_path)], check=True)

    with open(cuts) as f:
        entry = json.loads(f.readlines()[index])
    sup = entry["supervisions"][0]
    utt_id = sup["id"]

    with tarfile.open(tar_path) as tar:
        member = next(m for m in tar.getmembers() if m.name.startswith(utt_id))
        tar.extract(member, path=RAW)

    flac_path = RAW / member.name
    wav_path = SAMPLES / f"{lang_code}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(flac_path), "-ar", "16000", "-ac", "1", str(wav_path)],
                    check=True, capture_output=True)

    meta = {
        "lang_code": lang_code,
        "utt_id": utt_id,
        "ground_truth_ipa": sup["custom"]["clean"],
        "orthographic": sup["custom"]["orthographic"],
    }
    (SAMPLES / f"{lang_code}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return wav_path, meta


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <fleurs_lang_code> [split] [index]")
    lang_code = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    wav_path, meta = fetch(lang_code, split, index)
    print("wrote", wav_path)
    print("ground truth IPA:", meta["ground_truth_ipa"])
