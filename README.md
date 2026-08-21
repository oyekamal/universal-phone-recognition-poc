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

## Example run: Sindhi (full input/output)

`./venv/bin/python run_report.py sd_in` on one real IPAPack++/FLEURS test clip:

**INPUT**
- audio file: `samples/sd_in.wav` (real Sindhi speech, from IPAPack++/FLEURS test split)
- spoken sentence (native script, ground truth): `جڏهن ته اڪثر ڪري اهو صرف ناقص اسٽيريوٽائپ هوندو آهي، پيرس ۾ هلڻ جو بهترين طريقو اڃا تائين پنهنجي بهترين رويي تي رهڻ آهي، ڪنهن ماڻهو وانگر عمل ڪرڻ جيڪو bien élevé بهترين پرورش ٿيل آهي. اهو پورو ڪرڻ بابت تمام آسان ٿي ويندو`
- ground-truth IPA (human-transcribed, what a correct S2P model should output): `ɟaɖẽhɪ̃ t̪ʌh akəsʌr kareː ɪhoː ʂərʌf naːqʌʂ asʈiːrjoːʈaːɪp hũdoː aːhja peːrʌs haː həlʌɳ ɟoː bəhət̪riːn təriːqoː aɲaː t̪aːiːn pʌ̃hnɟi bəhət̪riːn rʋjeː t̪eː rəhʌɳ aːhja kʌ̃hɪ̃ maːɳhuː ʋaːngʌr amʌl kərʌɳ ɟeːkoː biːn eːlɛveː bəhət̪riːn pəroːrʌʃ tʰɪal aːheː ɪhoː poːroː kərʌɳ baːbɪt̪ t̪əmaːm aːsaːn tʰi ʋeːndoː`

**OUTPUT**

| Model | Predicted IPA | Romanized (English-readable, `romanize.py`) | PER vs ground truth |
|---|---|---|---|
| *(ground truth)* | `ɟaɖẽhɪ̃ t̪ʌh akəsʌr kareː ɪhoː ʂərʌf naːqʌʂ...` | `gyadenhin tuh akasur karee ihoo sharuf naaqush...` | — |
| Allosaurus | `d͡ʒ̤ɒðɒtsʔaksʌɾkaɾajsɛliːk͡p̚uəoɴnak͡p̚ɯsɯst͡ɕiːœoɾtɒpʰoŋ̟duəaheppʲeɾesmehaɴlɛɴt͡ɕjoɾpuəɒt̪ɒɾiːɴt̪ɪɾiːkuəb̞ɒɴɲjat̪ɒɴʔɯpɒnd͡ʒepb̤ɛt̪eɾiːɴtʂɒb̞ijjet̪iːɾɒɴɒk͡p̚ɒhɛbiaɳmaɾʌnɒvɒŋɡ̤uəɾuːamal̪uɾk͡p̚ʌnɒtɕuuətʂekuəab̤ɒt̪iːɾʲijiːpaɾb̞ɾet͡ʃʲl̪lʲivuəb̞uəpuəɾuəkanɴʌb̥ʌofuəɒtɒmaŋmaʂæɴtʰib̞ɪnɴtouə` | `dzhhodhots'aksurkarayseliikpuaongnakpusustshiieuortophongduaaheppyeresmehanglengtshyorpuaotoriingtiriikuabongnyyatong'upondzhepbheteriingtshobiyyetiirongokpohebianmarunovongghuaruuamalurkpunotshuuatshekuaabhotiiryiyiiparbretshyllyivuabuapuaruakanngubuofuaotomangmashangthibinngtoua` | 228/264 = 86.4% |
| PhoneticXEUS | `d͡ʒən̪et̪əksəɾkəɾes̪ɪɾəon̪ɑkɪs̪ɪsʈeʈɑpɦũd̪oäɦepeɾəs̪mẽɦələ̃ɳd͡ʒobɛt̪əɾĩt̪əɾikoɑnɪɑ̃t̪ɑ̃ĩpəɪ̃d͡ʒebɛt̪əɾiɾəpɪet̪eɾəɦəɳɑekəɦɛ̃bɪənmɑɳʊnʋɑɡʊɾəməlkəɳd͡ʒod͡ʒɪkoäebɛt̪ɾɪ̃ɳpəɾʋəɾəʃl̪iʋʊʋpuɾokəɾəɳbɑpət̪əmɑ̃ɑs̪ɑ̃t̪iʋĩd̪o` | `dzhanetaksarkaresiraonakisistetaphundoaheperasmenhalanndzhobetarintarikoaniantaninpaindzhebetarirapieterahanaekahenbianmanunvaguramalkandzhodzhikoaebetrinnparvarashlivuvpurokaranbapatamanasantivindo` | 194/264 = 73.5% |

`romanize.py` is a **deterministic symbol lookup** (~90 base IPA segments,
diacritics/length/affricates handled by stripping/digraph rules) — no LLM,
no guessing, no meaning. It just spells out how the phones sound in Roman
letters, the same idea as "Tokyo" romanizing 東京: you can sound it out even
though it isn't itself an English word. Notice the ground truth's romanized
form reads close to actual Sindhi ("bahatriin", "aahya", "vaangur") — an
Urdu/Hindi/Sindhi speaker would recognize real words in there. The model
outputs' romanized forms are readable as *sounds* but don't form real words,
because the underlying IPA prediction itself is ~74-86% wrong (see PER) —
romanizing garbled IPA just gives you garbled-but-pronounceable syllables,
it doesn't fix the recognition error.

Full machine-readable version: [`reports/sd_in.json`](reports/sd_in.json).

**Ground-truth English translation** (of what was actually said, for human
readability — NOT a model output): *"Although it's often just an inaccurate
stereotype, the best way to get around in Paris is to keep on your best
behavior — to act like someone who is 'bien élevé' [well brought-up],
well-mannered. That's actually quite easy to do."*

### Stage 2 (P2T): does an LLM recover this from the raw IPA alone?

`llm_p2t.py` feeds each model's raw predicted IPA (above) to an LLM (Claude,
via `claude -p`, zero-shot — no fine-tuning) and asks it to reconstruct the
sentence, with no access to the ground truth. **Short answer: no, not at
this PER.** Full output in [`reports/sd_in_p2t.json`](reports/sd_in_p2t.json):

- On PhoneticXEUS's output (73.5% PER): reconstructed guess talks about
  "genetics," "upbringing," "character" — confidence `low`. Wrong topic
  entirely (actual topic: Paris etiquette), though it did correctly latch
  onto a few real word-level anchors (`hũd̪o`→"happens", `pəɾʋəɾəʃ`→"upbringing").
- On Allosaurus's output (86.4% PER): reconstructed guess talks about "cell
  phone battery" — confidence `low`. Also wrong topic.

This is the honest result, not a cherry-picked one: **raw Stage-1 PER above
~70% is too corrupted for zero-shot LLM P2T to recover the actual sentence.**
The paper's real P2T stage is a model *trained* on IPA→text pairs (T5/LoRA),
not an LLM guessing cold — that's the gap between what's here and a working
system. See [Next](#next).

## Results across languages

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
run_report.py <lang_code>     → Stage 1 (S2P): runs Allosaurus and
                                 PhoneticXEUS on that clip, prints both
                                 predicted IPA strings, a romanized
                                 (English-readable) version of each via
                                 romanize.py, + PER vs ground truth,
                                 writes reports/<lang_code>.json
llm_p2t.py reports/<x>.json   → Stage 2 (P2T): feeds each model's IPA
                                 output to an LLM (claude -p) to attempt
                                 reconstructing the sentence's actual
                                 meaning — no audio involved at this
                                 stage, IPA text only
```

- `fetch_sample.py`, `run_report.py`, `llm_p2t.py` — the scripts you actually run, in that order
- `romanize.py` — deterministic IPA→Roman-letters transliteration (lookup table, no model, no LLM) — different from `llm_p2t.py`: this just makes phones sound-out-able, it doesn't attempt to recover meaning
- `recognize.py` — Allosaurus S2P + a naive IPA→Latin substitution (superseded by `romanize.py`)
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
2. **Trained P2T, not zero-shot LLM.** `llm_p2t.py`'s zero-shot LLM guess
   fails at 70%+ PER (see above) — replace it with a model actually
   fine-tuned on IPA→text pairs (T5/LoRA per the paper) once Stage 1 error
   rates are low enough to make that worthwhile.
3. **More languages / more clips per language** for a less anecdotal signal
   than "one 20-second clip."
