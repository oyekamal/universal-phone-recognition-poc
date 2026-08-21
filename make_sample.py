"""Generate a short Sindhi speech sample via gTTS for local testing.

ponytail: gTTS (Google Translate TTS) instead of a real recorded corpus —
good enough to exercise the pipeline locally, swap for real Sindhi audio
(e.g. IPAPack++ / Common Voice sd) when testing actual recognition accuracy.
"""
import sys
from gtts import gTTS
from gtts.lang import tts_langs

TEXT_SD = "سلام، توهان ڪيئن آهيو؟ اڄ موسم تمام سٺي آهي."


def main():
    langs = tts_langs()
    if "sd" not in langs:
        sys.exit(f"gTTS has no Sindhi voice available. Supported subset: {list(langs)[:10]}...")
    gTTS(text=TEXT_SD, lang="sd").save("sample_sindhi.mp3")
    print("wrote sample_sindhi.mp3")


if __name__ == "__main__":
    main()
