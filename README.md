# Universal Phone Recognition POC

Local test of the speech→phone (S2P) stage from the Allosaurus / PhoneticXEUS
research, on **real speech** in any FLEURS language, pulled from
[IPAPack++](https://huggingface.co/anyspeech/ipapack_plus_2) — the same
17,000-hour multilingual dataset PhoneticXEUS was trained on. Two S2P models
are run head-to-head and scored against IPAPack++'s own ground-truth IPA:

- **[Allosaurus](https://github.com/xinjli/allosaurus)** — earlier universal
  phone recognizer, trained on 12 languages, `pip install allosaurus`.
- **[PhoneticXEUS](https://huggingface.co/changelinglab/PhoneticXeus)** —
  current SOTA, self-conditioned CTC on the XEUS multilingual speech encoder,
  trained on IPAPack++ itself (88 languages including Sindhi).

Everything here is real: real audio, real ground-truth IPA, no mocked data.
The one honest caveat is the *metric* — see [Results](#results) below.

## Quick start

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# fetch a sample + run both models + print a report, for any FLEURS language code
./venv/bin/python run_report.py sd_in   # Sindhi
./venv/bin/python run_report.py ur_pk   # Urdu
./venv/bin/python run_report.py hi_in   # Hindi
```

Full list of available language codes (77 of them):
```bash
curl -s "https://huggingface.co/api/datasets/anyspeech/ipapack_plus_2" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(sorted(set(s['rfilename'].split('/')[1].rsplit('-',1)[0] for s in d['siblings'] if s['rfilename'].startswith('fleurs_shar/'))))"
```

## Results

Ran on 3 languages so far, one test-split clip each:

| Language | Allosaurus PER | PhoneticXEUS PER |
|---|---|---|
| Sindhi (`sd_in`) | 86.4% | 73.5% |
| Urdu (`ur_pk`) | 79.0% | 77.3% |
| Hindi (`hi_in`) | 77.6% | **39.5%** |

**PhoneticXEUS beats Allosaurus in all 3**, sometimes by a lot (Hindi).
Allosaurus's high error rate matches the paper's own finding: trained on
only 12 languages, it degrades to >80% PER on languages outside that set.

**Read the PER numbers as a rough proxy, not a benchmark score.** This is
plain character-level Levenshtein distance between the predicted and
ground-truth IPA strings (`eval.py:levenshtein`) — no phone tokenization,
no alignment, no partial credit for near-miss phones (e.g. `s` vs `z`). The
paper's real metric is PFER (phone-*feature* error rate, via articulatory
features) on a proper benchmark (PRiSM), and reports 17.7% multilingual /
10.6% accented-English for PhoneticXEUS — this repo's numbers are not
comparable to that and are expected to look worse. Use these numbers to
compare Allosaurus vs PhoneticXEUS on the *same* clip (valid), not to judge
either model's absolute quality (not valid yet — see Next).

## How it works

```
fetch_sample.py <lang_code>   → downloads one IPAPack++/FLEURS shard for
                                 that language, extracts one clip + its
                                 ground-truth IPA + orthographic transcript
run_report.py <lang_code>     → runs Allosaurus and PhoneticXEUS on that
                                 clip, prints both predicted IPA strings and
                                 the PER vs ground truth
```

- `fetch_sample.py`, `run_report.py` — the two scripts you actually run
- `recognize.py` — Allosaurus S2P + a naive IPA→Latin substitution (**not**
  a real phone-to-text model, just enough to see the pipeline run end to end)
- `phoneticxeus_recognize.py` — PhoneticXEUS S2P standalone
- `eval.py`, `eval_xeus.py` — single-model PER checks (superseded by `run_report.py`, kept for reference)
- `make_sample.py` — dead end, gTTS has no Sindhi voice; kept as a note for languages it does support

## Setup notes / gotchas

- PhoneticXEUS's custom model code breaks on `transformers>=5.x`'s
  `AutoModel` internals (`all_tied_weights_keys` AttributeError) — pinned to
  `transformers==4.46.3` in `requirements.txt`.
- `torchaudio.load()` now needs `torchcodec` installed as its backend
  (no longer bundled).
- Runs on GPU automatically if `torch.cuda.is_available()` — check this
  independently of `nvidia-smi`, which can report a broken driver/library
  mismatch while CUDA still works fine for torch.
- gTTS (used in the abandoned `make_sample.py` approach) has no Sindhi
  voice — real audio has to come from an actual corpus (IPAPack++/FLEURS
  here), not TTS.

## Next

1. **Real PFER, not char-edit-distance.** Clone the actual
   [PhoneticXeus repo](https://github.com/changelinglab/PhoneticXeus) (not
   just the `transformers.AutoModel` quick-inference path used here) and use
   its `src.metrics.phone_recognition.PhoneRecognitionEvaluator` for a number
   actually comparable to the paper's.
2. **Real Stage 2 (P2T).** Replace the naive IPA→Latin substitution in
   `recognize.py` with a fine-tuned T5/LoRA model or an LLM prompted with
   the IPA string — the paper's real two-stage S2P→P2T pipeline for turning
   phones into fluent English text.
3. **More languages / more clips per language** for a less anecdotal signal
   than "one 20-second clip."
