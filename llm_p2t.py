"""Stage 2 (P2T): feed a model's predicted IPA phone string to an LLM and
ask it to reconstruct fluent text — the paper's real P2T stage, done with
a prompted LLM instead of a fine-tuned T5/LoRA model.

Uses the `claude` CLI (`claude -p`, non-interactive print mode) so it works
with your existing Claude Code login — no separate API key needed. Falls
back to ANTHROPIC_API_KEY/OPENAI_API_KEY via their SDKs if `claude` isn't
on PATH.

    ./venv/bin/python llm_p2t.py reports/sd_in.json
"""
import json
import os
import shutil
import subprocess
import sys

PROMPT_TEMPLATE = """You are doing phone-to-text (P2T) reconstruction: turning a raw \
IPA (International Phonetic Alphabet) phone sequence, produced by an automatic \
speech recognizer, back into fluent text in its source language.

Language: {lang_code}
Raw IPA phones from the recognizer (no spaces, no word boundaries, may contain \
recognition errors):
{ipa}

Reconstruct the most likely intended sentence in {lang_code}, then give an \
English translation. If the IPA is too garbled to recover reliably, say so \
plainly rather than guessing confidently — do not invent a clean sentence \
from noise.

Respond as JSON only: {{"reconstructed_native_text": "...", "english_translation": "...", "confidence": "high|medium|low", "notes": "..."}}"""


def build_prompt(lang_code, ipa):
    return PROMPT_TEMPLATE.format(lang_code=lang_code, ipa=ipa)


def call_claude_cli(prompt):
    out = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def call_anthropic(prompt):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_openai(prompt):
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def run_p2t(prompt):
    if shutil.which("claude"):
        return call_claude_cli(prompt)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return call_anthropic(prompt)
    if os.environ.get("OPENAI_API_KEY"):
        return call_openai(prompt)
    return None


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <reports/lang_code.json>")
    report = json.loads(open(sys.argv[1]).read())
    lang_code = report["lang_code"]

    results = {}
    for model_key in ("output_phoneticxeus_ipa", "output_allosaurus_ipa"):
        ipa = report[model_key]
        prompt = build_prompt(lang_code, ipa)
        print(f"=== P2T on {model_key} (PER {report[model_key.replace('_ipa', '_per')]:.1%}) ===")
        reply = run_p2t(prompt)
        if reply is None:
            print("No `claude` CLI on PATH and no ANTHROPIC_API_KEY/OPENAI_API_KEY set.")
            print("Paste this into ChatGPT/Claude by hand:\n")
            print(prompt)
            continue
        print(reply)
        results[model_key] = reply
        print()

    if results:
        out_path = sys.argv[1].replace(".json", "_p2t.json")
        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
