"""Fetch a real sample for a FLEURS language, run both S2P models
(Allosaurus, PhoneticXEUS), and print an INPUT vs OUTPUT report scored
against ground truth. This is the one script to run per language.

    ./venv/bin/python run_report.py sd_in
    ./venv/bin/python run_report.py ur_pk
    ./venv/bin/python run_report.py hi_in

Writes reports/<lang_code>.json with the same data, for later use (e.g.
by llm_p2t.py to reconstruct English text from the IPA output).
"""
import json
import sys
from pathlib import Path

from eval import levenshtein
from fetch_sample import fetch
from romanize import romanize
from s2p import allosaurus_ipa, phoneticxeus_ipa

REPORTS = Path(__file__).parent / "reports"


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

    allo_pred = allosaurus_ipa(wav_path)
    allo_dist, allo_len, allo_per = per(allo_pred, truth)

    xeus_pred = phoneticxeus_ipa(wav_path)
    xeus_dist, xeus_len, xeus_per = per(xeus_pred, truth)

    print("############################  INPUT  ############################")
    print(f"language:              {lang_code}   (utterance id: {meta['utt_id']})")
    print(f"audio file:            {wav_path}")
    print(f"spoken sentence:       {meta['orthographic']}   <- what the speaker actually said, native script")
    print(f"ground-truth IPA:      {truth}   <- human-transcribed phones for that sentence (from IPAPack++)")
    print()
    allo_roman, allo_unmapped = romanize(allo_pred)
    xeus_roman, xeus_unmapped = romanize(xeus_pred)

    print("############################  OUTPUT  ############################")
    print("[Allosaurus predicted IPA]  ", allo_pred)
    print("[Allosaurus romanized]      ", allo_roman)
    print(f"  -> PER vs ground truth: {allo_dist}/{allo_len} = {allo_per:.1%}  (char-edit-distance proxy, not real PFER)")
    print()
    print("[PhoneticXEUS predicted IPA]", xeus_pred)
    print("[PhoneticXEUS romanized]    ", xeus_roman)
    print(f"  -> PER vs ground truth: {xeus_dist}/{xeus_len} = {xeus_per:.1%}  (char-edit-distance proxy, not real PFER)")
    print("####################################################################")
    print("NOTE: 'romanized' is a deterministic IPA->Latin-letter lookup (romanize.py)")
    print("      so an English reader can sound it out. It is NOT a translation and")
    print("      NOT the same as llm_p2t.py's meaning-reconstruction attempt.")

    REPORTS.mkdir(exist_ok=True)
    report = {
        "lang_code": lang_code,
        "utt_id": meta["utt_id"],
        "audio_file": str(wav_path),
        "input_spoken_sentence": meta["orthographic"],
        "input_ground_truth_ipa": truth,
        "input_ground_truth_romanized": romanize(truth)[0],
        "output_allosaurus_ipa": allo_pred,
        "output_allosaurus_romanized": allo_roman,
        "output_allosaurus_per": allo_per,
        "output_phoneticxeus_ipa": xeus_pred,
        "output_phoneticxeus_romanized": xeus_roman,
        "output_phoneticxeus_per": xeus_per,
    }
    report_path = REPORTS / f"{lang_code}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
