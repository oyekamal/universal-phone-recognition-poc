# Universal Phone Recognition POC (Sindhi)

Local test of the S2P (speech→phone) stage from the Allosaurus/PhoneticXEUS
research: `venv/`, real Sindhi audio pulled from IPAPack++ (HF:
`anyspeech/ipapack_plus_2`, `fleurs_shar/sd_in-test`), Allosaurus for
recognition.

## Files
- `raw/` — downloaded IPAPack++ Sindhi shard (`sd_cuts.jsonl` ground-truth IPA + Sindhi script, `*.flac` audio)
- `sample_sindhi.wav` — first Sindhi test clip, converted to 16k mono
- `recognize.py` — Stage 1 (Allosaurus) + naive Stage 2 (IPA→Latin substitution, NOT the real P2T model)
- `eval.py` — char-level edit distance vs ground-truth IPA (rough PER)
- `make_sample.py` — dead end: gTTS has no Sindhi voice, kept for other languages

## Result
Allosaurus PER on this Sindhi clip: **~86%** — matches the paper's own
finding that Allosaurus (trained on only 12 languages) degrades to >80% PER
on languages far from its training set. Confirms we need **PhoneticXEUS**
(SOTA, trained on IPAPack++ itself, includes ~9h of Sindhi) for anything
usable, not Allosaurus.

## Next
1. Get PhoneticXEUS running locally (`github.com/changelinglab/PhoneticXeus`,
   needs the XEUS encoder checkpoint) and re-run `eval.py` equivalent — this
   is the real Stage 1 model, should land far below Allosaurus's 86% PER.
2. Real Stage 2 (P2T): replace `recognize.py`'s naive substitution with a
   fine-tuned T5/LoRA model or an LLM prompted with the IPA string, per the
   paper's two-stage S2P→P2T pipeline.
