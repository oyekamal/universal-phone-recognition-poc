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
Rough char-level PER on this Sindhi clip vs IPAPack++ ground truth:

| Model | PER |
|---|---|
| Allosaurus | ~86% |
| PhoneticXEUS | ~73.5% |

Allosaurus's number matches the paper's own finding: trained on only 12
languages, it degrades to >80% PER on languages far outside that set.
PhoneticXEUS (trained on IPAPack++ itself, including ~9h of Sindhi) does
meaningfully better but is still far from the paper's reported 17.7%
multilingual PFER — expected, since this is a raw char-edit-distance proxy
on one 20s clip, not the paper's phone-aligned PFER metric or eval set.

PhoneticXEUS runs via `transformers.AutoModel` (`phoneticxeus_recognize.py`,
`eval_xeus.py`) — no repo clone needed. Pinned `transformers==4.46.3`
(the custom model code breaks on transformers 5.x's `AutoModel` internals)
and added `torchcodec` (torchaudio's audio backend moved to it). Runs on
GPU if `torch.cuda.is_available()` (it is, here — `nvidia-smi` itself is
broken by a driver/library version mismatch, but the CUDA runtime torch
uses works fine).

## Next
1. Real phone-aligned PFER, not char-edit-distance: use
   `src.metrics.phone_recognition.PhoneRecognitionEvaluator` from the
   PhoneticXeus repo (needs `git clone` + `make install`, not just the
   quick-inference path) for a number comparable to the paper's.
2. Real Stage 2 (P2T): replace `recognize.py`'s naive substitution with a
   fine-tuned T5/LoRA model or an LLM prompted with the IPA string, per the
   paper's two-stage S2P→P2T pipeline.
