"""Deterministic IPA -> readable Roman-English transliteration.

This is NOT the P2T stage (llm_p2t.py) — no meaning, no guessing, no model.
Just a symbol lookup: each IPA phone maps to the closest Roman letters so
an English reader can sound it out. Same idea as how "Beijing" or "Tokyo"
are romanizations of Mandarin/Japanese phonology, not translations.

ponytail: covers the ~90 IPA base segments PhoneticXEUS's 428-symbol vocab
is actually built from (diacritics stripped, common affricate digraphs
mapped as units) rather than hardcoding all 428 combinations. Anything still
unmapped after that is left as-is and flagged, not silently dropped.
"""
import re
import sys
import unicodedata

# common multi-character sequences (affricates, digraphs) checked before
# single-character lookup, longest first
DIGRAPHS = {
    "tʃ": "ch", "dʒ": "j", "tɕ": "ch", "dʑ": "j", "ʈʂ": "ch", "ɖʐ": "j",
    "ts": "ts", "dz": "dz", "kp": "kp", "gb": "gb", "ŋ͡m": "ngm",
}

BASE = {
    # plosives
    "p": "p", "b": "b", "t": "t", "d": "d", "ʈ": "t", "ɖ": "d",
    "c": "ky", "ɟ": "gy", "k": "k", "g": "g", "ɡ": "g", "q": "q", "ɢ": "g",
    "ʔ": "'",
    # nasals
    "m": "m", "ɱ": "m", "n": "n", "ɳ": "n", "ɲ": "ny", "ŋ": "ng", "ɴ": "ng",
    # trills / taps / flaps
    "r": "r", "ʀ": "r", "ɾ": "r", "ɽ": "r", "ɹ": "r", "ɻ": "r",
    # fricatives
    "f": "f", "v": "v", "θ": "th", "ð": "dh", "s": "s", "z": "z",
    "ɕ": "sh", "ʑ": "zh",
    "ʃ": "sh", "ʒ": "zh", "ʂ": "sh", "ʐ": "zh", "ç": "h", "ʝ": "y",
    "x": "kh", "ɣ": "gh", "χ": "kh", "ʁ": "r", "ħ": "h", "ʕ": "a",
    "h": "h", "ɦ": "h",
    # affricates handled in DIGRAPHS; lone symbols with tie bars stripped first
    # approximants / laterals
    "l": "l", "ɭ": "l", "ʎ": "ly", "ʟ": "l", "w": "w", "ɰ": "w",
    "j": "y", "ɥ": "y", "ʋ": "v",
    # implosives / ejectives / clicks - approximate to base
    "ɓ": "b", "ɗ": "d", "ʄ": "j", "ɠ": "g", "ʛ": "g",
    # vowels
    "i": "i", "y": "ü", "ɨ": "i", "ʉ": "u", "ɯ": "u", "u": "u",
    "ɪ": "i", "ʏ": "u", "ʊ": "u",
    "e": "e", "ø": "eu", "ɘ": "e", "ɵ": "o", "ɤ": "o", "o": "o",
    "ɛ": "e", "œ": "eu", "ɜ": "er", "ɞ": "er", "ʌ": "u", "ɔ": "o",
    "æ": "a", "ɐ": "a", "a": "a", "ɶ": "a", "ɑ": "a", "ɒ": "o",
    "ə": "a",
}

# combining diacritics (stripped after decomposition) mapped to a suffix
# hint instead of silently dropped
DIACRITIC_SUFFIX = {
    "̃": "n",   # combining tilde - nasalization
    "̤": "h",   # breathy voice
    "̰": "",    # creaky voice - no good roman analog, drop
    "̪": "",    # dental - drop, base letter already close enough
    "̺": "",    # apical - drop
    "̟": "",    # advanced - drop
    "̠": "",    # retracted - drop
    "̈": "",    # centralized (diaeresis) - drop
    "̞": "",    # lowered - drop
    "̝": "",    # raised - drop
    "̩": "",    # syllabic consonant marker - drop
}

MODIFIER_SUFFIX = {
    "ʰ": "h",   # aspiration
    "ʲ": "y",   # palatalization
    "ʷ": "w",   # labialization
    "ˀ": "'",   # glottalization
    "ˠ": "",    # velarization - drop
    "ʼ": "'",   # ejective
    "̚": "",    # unreleased - drop
    "̥": "",    # voiceless - drop
    "̤": "h",   # breathy (spacing form)
}

LENGTH_MARKS = {"ː": "", "ˑ": ""}  # long vowel -> just double letter, handled separately

TIE_BARS = {"͡": "", "͜": ""}


def romanize_phone(phone):
    """One IPA phone (possibly with diacritics) -> roman letters, or None if
    completely unmapped."""
    # strip tie bars first so digraph lookup sees plain sequences
    for tb in TIE_BARS:
        phone = phone.replace(tb, "")

    is_long = any(m in phone for m in LENGTH_MARKS)
    for m in LENGTH_MARKS:
        phone = phone.replace(m, "")

    suffix = ""
    for mod, sfx in MODIFIER_SUFFIX.items():
        if mod in phone:
            suffix += sfx
            phone = phone.replace(mod, "")

    # decompose remaining combining marks (nasalization etc.)
    decomposed = unicodedata.normalize("NFD", phone)
    base_chars = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            suffix += DIACRITIC_SUFFIX.get(ch, "")
        else:
            base_chars.append(ch)
    phone = "".join(base_chars)

    if phone in DIGRAPHS:
        roman = DIGRAPHS[phone]
    elif phone in BASE:
        roman = BASE[phone]
    elif len(phone) == 2 and phone[0] in BASE and phone[1] in BASE:
        roman = BASE[phone[0]] + BASE[phone[1]]
    elif phone in BASE:
        roman = BASE[phone]
    else:
        return None

    if is_long:
        roman = roman + roman[-1] if roman else roman
    return roman + suffix


def romanize(ipa_string, phone_sep=""):
    """Full IPA string, space-separated by word or one run-together blob (or
    /-separated phones as in PhoneticXEUS's predicted_transcript) ->
    (roman_text, unmapped_list). Word/phone separators are preserved.

    phone_sep: string inserted between each phone's romanization (default
    "" for a normal word). Pass e.g. "." for a scannable syllable-spaced
    view of a model's raw output, which has no real word boundaries at all
    — this doesn't invent words, it just breaks up an unreadable wall of
    letters into pronounceable chunks."""
    words = ipa_string.split(" ")
    out_words = []
    unmapped = []
    for word in words:
        phones = word.split("/") if "/" in word else _grapheme_split(word)
        rendered = []
        for p in phones:
            if not p:
                continue
            r = romanize_phone(p)
            if r is None:
                unmapped.append(p)
                rendered.append(f"[{p}]")
            else:
                rendered.append(r)
        out_words.append(phone_sep.join(rendered))
    return " ".join(out_words), unmapped


def _grapheme_split(s):
    """Split a run-together IPA string into (base char + trailing combining
    marks/modifier letters) clusters — good enough without a full Unicode
    grapheme-cluster library."""
    marks = set("̤̰̪̺̟̠̃͜͡ʰʲʷˀˠʼ̥̤̚ːˑ̞̝̩̈")
    clusters = []
    cur = ""
    for ch in s:
        if ch in marks and cur:
            cur += ch
        else:
            if cur:
                clusters.append(cur)
            cur = ch
    if cur:
        clusters.append(cur)
    return clusters


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} '<ipa string>'")
    roman, unmapped = romanize(sys.argv[1])
    print("romanized:", roman)
    if unmapped:
        print("unmapped phones (left in [brackets] above):", unmapped)
