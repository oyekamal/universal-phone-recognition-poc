# Deep Testing Report

Two test sets, both run through the full pipeline (`s2p.py` → `romanize.py` → `llm_p2t.py`/`recognize_any_audio.py`):

1. **5 new FLEURS languages** (ground truth available, real PER) — Vietnamese, Zulu, Finnish, Arabic, Mandarin. Chosen for typological spread: tonal, click consonants, agglutinative, different script, logographic.
2. **8 real long-form Sindhi classroom recordings** (~18-23 min each, no ground truth) — 11 chunks total (one early ~60s chunk from each of 9 files including the `.aac`, plus a second mid-recording chunk from 2 files to check for drift). Source: user-provided Google Drive folder. Real spontaneous classroom speech, not FLEURS's clean read sentences.

All PER numbers are the same char-level Levenshtein proxy used throughout this repo — not real phone-feature error rate. Sample size is N=1 clip per FLEURS language; treat everything here as a testing log, not a benchmark.

## Summary table

| Case | PER: Allosaurus / PhoneticXEUS | P2T outcome (PhoneticXEUS path) | P2T outcome (Allosaurus path) |
|---|---|---|---|
| Finnish (`fi_fi`) | 77.8% / **4.3%** | **Success** — correct sentence, `medium` confidence | Failed — unrecoverable |
| Hindi (`hi_in`, earlier) | 77.6% / 39.5% | **Success** — correct gist, `low` confidence | Failed |
| Mandarin (`cmn_hans_cn`) | 55.8% / **2.7%** | Partial — right keywords (Kundalini, energy, awakening), wrong surrounding narrative, `low` | Failed |
| Arabic (`ar_eg`) | 76.2% / 51.1% | Partial — right topic (child neglect/social upbringing) and much of the structure, `low` | Failed — flagged non-Arabic phones |
| Vietnamese (`vi_vn`) | 78.0% / 22.0% | Partial — several correct word-fragments (ice/snow, many countries, not interrupted), wrong assembly, `low` | Failed |
| Zulu (`zu_za`) | 72.9% / 24.5% | Partial — correct ending fragment ("heal injured bodies"), garbled middle, `low` | Failed — flagged non-Zulu phones |
| Sindhi (`sd_in`, earlier) | 86.4% / 73.5% | Failed — wrong topic (genetics) | Failed — wrong topic (phone battery) |
| Urdu (`ur_pk`, earlier) | 79.0% / 77.3% | Failed | Failed |
| **11 Sindhi classroom chunks** (no ground truth) | n/a | **11/11 failed** — every chunk `low` confidence, "unable to reconstruct reliably" or partial fragments only | **11/11 failed** |

Allosaurus never produced a usable P2T result in any of the 13 total test cases in this report. PhoneticXEUS ranged from clean success to total failure depending on language/PER.

## Part 1: FLEURS languages, full detail

### Finnish (`fi_fi`) — the strongest result

**Input**
- Ground truth: *"suurin osa valtiossa työskentelevistä puhuu italiaa myös arkikielenään mutta uskonnollisissa toimituksissa käytetään usein latinaa"*
- Ground-truth IPA: `suurin osɑ vɑltiossɑ tyøskentelevistæ puhuu itɑliɑɑ myøs ɑrkikielenææn muttɑ uskonnollisissɑ toimituksissɑ kæytetææn usein lɑtinɑɑ`

**Output**
- Allosaurus IPA: `suəɾɒjɯnɒsɒbɒl̪ldjewsat̪ɨʏs̪k͡p̚ɛɴtelʲeɟiːstt̪ɤ̆puəb̞uəʔiːtɒl̪ijjamәɛnaɾkiːkjelɒnɒɳmut̪aʔuəskɔnɴɔnl̪ɯɕiːzsʌt̪uəɒmiːduəksiːsʌkaɪðɪtanɴuəseɪlwatina` → PER 77.8%
- Allosaurus romanized: `suaroyunosobolldyewsatiuskpengtelyegyiistto[̆]puabua'iitoliyyam[ә]enarkiikyelononmuta'uaskonngonlushiizsutuaomiiduaksiisukaidhitannguaseilwatina`
- PhoneticXEUS IPA: `suuriinosɑvɑltiossɑtyøskentelivistæpuhuuitɑliɑmyøsɑrkikiellenæænmuuttɑuskonnollisissɑtoimituksissɑkæytetæænuseinlɑtinɑɑ` → PER **4.3%** (5 char edits out of 117)
- PhoneticXEUS romanized: `suuriinosavaltiossatüeuskentelivistapuhuuitaliamüeusarkikiellenaanmuuttauskonnollisissatoimituksissakaütetaanuseinlatinaa`

**P2T (PhoneticXEUS path, confidence `medium`)**
- Reconstructed: *"Suurin osa valtiossa työskentelevistä puhui italiaa myös arkikielenään, mutta uskonnollisissa toimituksissa käytetään usein latinaa."*
- English: *"The majority of those working in the state also spoke Italian as their everyday language, but in religious services Latin is often used."*
- This is essentially the correct sentence, word for word.

**P2T (Allosaurus path, confidence `low`)**: `[unrecoverable]` — LLM explicitly flagged multiple phones (ɒ, ɯ, ɴ, ɳ, ɕ, b̞, ð, ʔ, k͡p̚, ɤ̆) as not native to Finnish phonology and declined to fabricate a sentence.

Full data: [`reports/fi_fi.json`](reports/fi_fi.json), [`reports/fi_fi_p2t.json`](reports/fi_fi_p2t.json)

### Mandarin (`cmn_hans_cn`) — lowest PER, still wrong

**Input**
- Ground truth (topic: Kundalini yoga): `在昆达里尼瑜伽中昆达里尼能量启迪能量通过瑜伽姿势呼吸练习念语和视觉形象被唤醒` — *"In Kundalini yoga, Kundalini energy — awakening energy — is awakened through yoga postures, breathing exercises, chanting, and visualization."*
- Ground-truth IPA: `tsaɪ kʰwən ta li ni y tɕja ʈʂʊŋ kʰwən ta li ni nɤŋ ljɑŋ tɕʰi ti nɤŋ ljɑŋ tʰʊŋ kwɔ y tɕja tsɯ ʂɨ xu ɕi ljɛn ɕi njɛn y xan ʂɨ tɕɥœ ɕiŋ ɕjɑŋ peɪ xwan ɕiŋ`

**Output**
- Allosaurus IPA: `tsaikuangdaliyuayarongkuunaliminungliangtshhidingingliangtongkuaoyuayatsishrkhuashidhunshiyakhushryouashingngshingreikhuaangnshing` (romanized) → PER 55.8%
- PhoneticXEUS IPA: `tsaɪkʰwəntaliniytɕjaʈʂʊŋkʰwəntalininɤŋljɑŋtɕʰitinɤŋljɑŋtʰʊŋkwɔytɕjatsɯʂɨxuɕiljɛnɕinjɛnyxɤʂɨtɕjœɕiŋɕjɑŋpeɪxwanɕiŋ` → PER **2.7%** (3 char edits out of 113 — near-perfect phone match)

**P2T (PhoneticXEUS path, confidence `low`)**
- Reconstructed: 在昆达里尼与家中，昆达里尼能量，其的能量通过家资式互联信，言与合时，觉醒向北欢兴。
- English: *"In Kundalini and at home, Kundalini energy — its energy passes through home-resource-type interconnection messages; in speech and at the right time, awakening joyfully toward the north."*
- **Got right**: "Kundalini" (twice), "energy" (twice), "awakening" concept present near the end.
- **Got wrong**: invented "home-resource-type interconnection," "toward the north," dropped "yoga," "postures," "breathing," "visualization" entirely, no coherent match to the real sentence's structure.
- The LLM's own notes correctly identified 能量 (energy), 通过 (through/via), and the "kundalini" transliteration as its most confident anchors — appropriately humble about the rest.

**Why this matters**: 2.7% character-level PER is about as good as phone recognition gets in this whole report, yet P2T still produced a *wrong* sentence, only recovering isolated correct keywords. Two likely reasons: (1) Mandarin has no orthographic word boundaries and is monosyllable-per-morpheme dense, so segmenting a phone stream into the right hanzi is a much harder combinatorial problem than in an alphabetic language even at low phone error; (2) our PER metric is computed on **romanized/IPA text**, not tone-aware — it can't detect a tone error, and a wrong tone on a homophone-heavy language like Mandarin changes the character/meaning entirely while leaving the metric looking almost perfect. **Char-edit-distance PER is a genuinely misleading predictor of P2T success for tonal/logographic languages** — this is the most important nuance to come out of this deep-testing pass.

Full data: [`reports/cmn_hans_cn.json`](reports/cmn_hans_cn.json), [`reports/cmn_hans_cn_p2t.json`](reports/cmn_hans_cn_p2t.json)

### Arabic (`ar_eg`) — partial, surprisingly coherent gist

**Input**
- Ground truth (topic: social/child development): *"ومن بين أكثر الطرق شيوعاً التي تستخدم لتوضيح أهمية التنشئة الاجتماعية الاعتماد على الحالات القليلة المؤسفة للأطفال الذين عانوا من خلال الإهمال أو سوء الحظ أو الإيذاء المتعمد غير مرتبطين اجتماعياً من جانب البالغين أثناء نشأتهم"* — roughly: *"Among the most common methods used to demonstrate the importance of social upbringing is reliance on the few unfortunate cases of children who suffered neglect, bad luck, or intentional abuse, [and were] not socially bonded by adults during their upbringing."*

**Output**
- PhoneticXEUS IPA → PER **51.1%**
- Allosaurus IPA → PER 76.2%

**P2T (PhoneticXEUS path, confidence `low`)**
- Reconstructed: *"ومن بيننا أكثر الطرق الشائعة اللي تُستخدم لتوضيح الهامش الاجتماعي، والاعتماد على الحالة وقلة السيئة للأطفال اللذين عانوا من خلال الإهمال أو سوء الحظ أو الإيذاء المتعمد، غير مرتبط اجتماعياً من جانب البالغين أثناء نشأتهم."*
- English: *"And among us are the most common methods used to clarify social marginalization, and reliance on the situation and the poor conditions of children who have suffered through neglect, bad luck, or intentional harm — not socially engaged on the part of adults during their upbringing."*
- **This is close to the actual meaning** — right topic (child welfare, neglect, upbringing), most content words correct (الأطفال "children", الإهمال "neglect", سوء الحظ "bad luck", الإيذاء المتعمد "intentional harm", البالغين "adults"), structure mostly intact. Genuinely one of the better partial recoveries in this whole test, despite 51.1% PER and the LLM itself calling it `low` confidence.

**P2T (Allosaurus path, confidence `low`)**: `[not reliably recoverable]` — the LLM explicitly noted the phones looked more South-Asian than Egyptian-Arabic and speculated the S2P output might reflect cross-language confusion.

Full data: [`reports/ar_eg.json`](reports/ar_eg.json), [`reports/ar_eg_p2t.json`](reports/ar_eg_p2t.json)

### Vietnamese (`vi_vn`) — tonal, partial

**Input**
- Ground truth: *"mặt khác băng và tuyết là hiện tượng bình thường ở nhiều nước và quanh năm giao thông gần như không bị gián đoạn"* — *"besides, ice and snow are a normal phenomenon in many countries, and transportation is almost never interrupted year-round."*

**Output**: PhoneticXEUS PER **22.0%**, Allosaurus PER 78.0%

**P2T (PhoneticXEUS path, confidence `low`)**
- Reconstructed: *"Mắc kẹt băng và tuyết là hẹn trên bệnh thương ơi nhiều nước và văn Nam giáo thông mạng như không bị giản đoàn"*
- English: *"(Approximate) Trapped in ice and snow... many countries and Vietnamese cultural education spreading through the network, as if not simply a group"*
- **Got right** (as fragments, not full sentence): băng và tuyết (ice and snow), nhiều nước (many countries), không bị gián đoạn≈giản đoàn (not interrupted — very close match). **Got wrong**: assembled them around an invented "trapped," "network," "group" narrative instead of the real "normal phenomenon / transportation" framing. Tone marks are entirely absent from the S2P output (these models don't recognize Vietnamese tone), which the LLM flagged as a specific reason full recovery was impossible — Vietnamese is a minimal-pair-heavy tonal language, so untoned phones are systematically ambiguous in a way alphabetic non-tonal languages aren't.

Full data: [`reports/vi_vn.json`](reports/vi_vn.json), [`reports/vi_vn_p2t.json`](reports/vi_vn_p2t.json)

### Zulu (`zu_za`) — click consonants, partial

**Input**
- Ground truth (topic: road accidents and rehabilitation): *"ukuba nezimoto kwabantu abaningi nakho kuholela ezigamekweni eziphakeme zezingozi emigwaqweni okuholela ekusungulweni kwezindlela ezintsha kwezempilo zokulungisa imizimba elimele"* — roughly *"...vehicles for many people also leads to high rates of road accidents, which leads to new health methods being developed to fix injured bodies."*

**Output**: PhoneticXEUS PER 24.5%, Allosaurus PER 72.9%

**P2T (PhoneticXEUS path, confidence `low`)**
- Reconstructed: *"Ukuba nezimo ezinjalo abantu bani, nginawo ukukholela ukuphakameza izingoma zemculo, eqinisweni okholela ekisungeni, izindlela nezinja zokumbira, ozokulunga izimba ezilimele"*
- English: *"...To be in such conditions, who are the people... uplift the songs of music... those who will heal the injured bodies."*
- **Got right**: the closing fragment "izimba ezilimele" (injured bodies) closely matches the ground truth's ending "imizimba elimele" (injured bodies) — a real, specific correct recovery. **Got wrong**: everything in the middle (invented "songs of music" instead of "road accidents/vehicles"), and the click consonant `ǃ` in the S2P output was guessed as an unrelated word rather than recognized as a click letter.
- LLM's own notes: the IPA→Zulu-orthography mapping (ɓ→b, kʰ→kh, ŋ→ng, ɮ→dl, d͡ʒ→j) was done correctly, showing the model does know Zulu phonology — the failure is in word-segmentation of the garbled stream, not lack of language knowledge.

**P2T (Allosaurus path)**: flagged as containing phones "not present in Zulu phonology" (ʁ, ð, æ, ə, ʌ, ʲ, ʂ, ɨ) and returned `[unrecoverable]` rather than fabricate.

Full data: [`reports/zu_za.json`](reports/zu_za.json), [`reports/zu_za_p2t.json`](reports/zu_za_p2t.json)

## Part 2: 8 real Sindhi classroom recordings (11 chunks, no ground truth)

Downloaded from the user-provided Google Drive folder: 9 files (`sindhi_classroom_01.aac` through `_09.mp3`), 17-23 minutes each, real unscripted classroom speech (not FLEURS's clean single-sentence reads). One 25-second chunk extracted from the ~60s mark of each of the 9 files, plus a second chunk from the ~50%-duration mark of 2 files (`01`, `05`) to check for early-vs-mid-recording drift. All run through `recognize_any_audio.py --lang Sindhi` (language given as a hint, unlike a true blind test).

**Headline result: 11/11 chunks failed P2T on both S2P paths.** Every single one came back `confidence: low` with either `[unrecoverable]`, `[partial — see notes]`, or a "best-effort" reconstruction the LLM itself flagged as likely fabrication. This is consistent with — not contradicted by — the short-clip Sindhi FLEURS result (86.4%/73.5% PER, also failed): **Sindhi is a hard case for both S2P models regardless of recording style, length, or content.** There's no evidence long-form/spontaneous speech is uniquely harder than short/read speech here — it's uniformly hard.

**No early-vs-mid drift observed.** Files `01` and `05` each had an early (~60s) and mid (~50%) chunk; both positions failed the same way in both files. Two data points isn't enough to rule out quality drift deeper into long recordings in general, but nothing here suggests it.

**PhoneticXEUS still surfaced real, plausible classroom vocabulary** in nearly every chunk, even though full-sentence reconstruction failed — a genuine partial signal, not pure noise:
- Chunk 02: "ٺيڪ" (correct), "پاڻ" (self), "ڪتاب" (book), "کولي" (open), "نمبر" (number), "پيج" (page) → plausibly "open your book to page number ___", exactly what a teacher would say
- Chunk 05 (early): "پاڻي" (water), "ڇاء" (tea), "ڊاڪٽر" (doctor), "گلاس" (glass) → a domestic/clinical vocabulary scene
- Chunk 07: "اسڪول" (school), "گهر" (home), "سمجهايو" (explained), "مدرسو" (madrassa) → a school-vs-home-vs-madrassa education topic, again classroom-plausible
- Chunk 09: "پاڪ" (pure/Pakistan), "پنج" (five), "ٽامل" (Tamil), "جرمن" (German), "ڳوٺ" (village) → possibly a social-studies/geography topic

These anchor words are thematically coherent with "classroom" as the filename claims, which is a real (if soft) qualitative validation that the models are picking up genuine signal from the audio — the failure is in full-sentence assembly, not total noise.

**Allosaurus systematically hallucinated `k͡p̚`** (a labial-velar unreleased stop, not a Sindhi phoneme) in nearly every single one of the 11 chunks — sometimes 5-8 times per chunk. This is a specific, repeatable Allosaurus failure signature on this dataset, not random noise, and it alone was enough to make every Allosaurus-path P2T attempt fail. Worth flagging if debugging Allosaurus further: this may be a training-data or preprocessing artifact specific to this audio's recording conditions (compressed mp3/aac, classroom acoustics, possible background noise) rather than a Sindhi-specific issue.

Full per-chunk IPA strings and P2T outputs: [`reports/drive/`](reports/drive/) (11 JSON files, `<chunk>_p2t_report.json`).

## Refined finding: PER is not a reliable single predictor of P2T success

Ordering all 8 language data points by PhoneticXEUS PER:

| PER | Language | P2T outcome |
|---|---|---|
| 2.7% | Mandarin | Partial (right keywords, wrong sentence — tonal/logographic, PER metric misleading here) |
| 4.3% | Finnish | **Success** |
| 22.0% | Vietnamese | Partial (right fragments, wrong assembly — tonal, no tone info) |
| 24.5% | Zulu | Partial (one correct clause, garbled middle — click consonants, low-resource) |
| 39.5% | Hindi | **Success** (earlier result) |
| 51.1% | Arabic | Partial (coherent gist, mostly right) |
| 73.5% | Sindhi | Failed |
| 77.3% | Urdu | Failed |

There is a rough trend (higher PER → worse outcome) but it's not monotonic — Mandarin's 2.7% did worse than Finnish's 4.3% and about the same as Vietnamese's 22.0%. What actually seems to matter, based on reading the LLM's own reasoning across all these cases:
1. **Script/tone information lost in the phone stream.** Vietnamese and Mandarin both lose critical disambiguating information (tone) that IPA transcription doesn't carry — this hurts them independent of phone accuracy.
2. **The LLM's familiarity with the language's word-formation patterns.** Finnish's agglutination is regular and the LLM clearly reconstructed real Finnish words confidently; Zulu and Sindhi are comparatively lower-resource for the LLM itself, independent of the S2P model.
3. **Below ~50% PER, partial-but-topically-correct recovery becomes achievable** (Arabic, Vietnamese, Zulu, Mandarin all did this); **above ~70% PER, recovery failed completely** in every case tested (Sindhi, Urdu, and all 11 real classroom chunks).

## Limitations

- N=1 clip per FLEURS language (except Sindhi classroom: N=11 chunks, still N=1 per source file).
- No ground truth for the classroom recordings — assessment is qualitative (LLM confidence + anchor-word plausibility), not a PER number.
- The char-edit-distance PER metric itself is demonstrably unreliable for tonal/logographic languages (see Mandarin) — a real benchmark would need phone-aligned PFER and, for tonal languages, a tone-aware metric.
- This is a testing log across a handful of clips, not a statistically powered benchmark — don't treat any single percentage here as the model's "true" accuracy on that language.
