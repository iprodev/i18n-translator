#!/usr/bin/env python3
"""
i18n JSON Translator using LM Studio API
Translates JSON language files to target languages using a local LLM.
Features: Resume capability, nested structure support, placeholder preservation.
"""

import json
import argparse
import time
import hashlib
from pathlib import Path
from typing import Any
import requests

# LM Studio API Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Translation style presets
TRANSLATION_STYLES = {
    "formal": {
        "name": "Formal",
        "prompt": "Use formal, professional language. Use polite pronouns (شما in Persian, Sie in German, etc.).",
        "example": "Formal business tone"
    },
    "friendly": {
        "name": "Friendly",
        "prompt": "Use warm, friendly, conversational language. Use informal pronouns (تو in Persian, du in German, etc.). Sound natural like talking to a friend, not robotic or stiff.",
        "example": "Casual friendly tone"
    },
    "casual": {
        "name": "Casual",
        "prompt": "Use very casual, everyday spoken language. Use colloquial expressions and informal grammar where natural. Like texting a close friend.",
        "example": "Very informal, colloquial"
    },
    "playful": {
        "name": "Playful",
        "prompt": "Use fun, playful, energetic language. Add personality and warmth. Great for apps targeting young users.",
        "example": "Fun and energetic"
    },
    "neutral": {
        "name": "Neutral",
        "prompt": "Use clear, neutral language. Avoid being too formal or too casual. Balanced and accessible.",
        "example": "Balanced, clear"
    }
}

class TranslationState:
    """Manages translation progress and resume capability."""
    
    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state = self._load_state()
    
    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"completed": {}, "partial": {}}
    
    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def get_job_id(self, input_file: str, target_lang: str) -> str:
        return hashlib.md5(f"{input_file}:{target_lang}".encode()).hexdigest()[:12]
    
    def is_key_done(self, job_id: str, key_path: str) -> bool:
        return key_path in self.state["completed"].get(job_id, {})
    
    def get_translated(self, job_id: str, key_path: str) -> str | None:
        return self.state["completed"].get(job_id, {}).get(key_path)
    
    def mark_done(self, job_id: str, key_path: str, original: str, translated: str):
        if job_id not in self.state["completed"]:
            self.state["completed"][job_id] = {}
        self.state["completed"][job_id][key_path] = {"original": original, "translated": translated}
        self.save()
    
    def get_progress(self, job_id: str) -> int:
        return len(self.state["completed"].get(job_id, {}))
    
    def clear_job(self, job_id: str):
        if job_id in self.state["completed"]:
            del self.state["completed"][job_id]
            self.save()

def translate_text(text: str, source_lang: str, target_lang: str, model: str = "local-model", style: str = "friendly") -> str:
    """Translate a single text string using LM Studio API."""
    if not text or not text.strip():
        return text
    
    style_info = TRANSLATION_STYLES.get(style, TRANSLATION_STYLES["friendly"])
    style_instruction = style_info["prompt"]
    
    prompt = f"""Translate the following text from {source_lang} to {target_lang}.

Style: {style_instruction}

Rules:
- Only return the translated text, nothing else
- Do not add quotes or explanations
- Keep any placeholders like {{name}}, {{count}}, %s, %d, {{0}}, etc. unchanged
- Sound natural, not robotic or machine-translated

Text to translate:
{text}"""

    system_prompt = f"""You are a native {target_lang} speaker and professional translator.
Translate naturally from {source_lang} to {target_lang}.
Style: {style_instruction}
The translation should sound like it was originally written in {target_lang}, not translated.
Preserve all formatting, placeholders, and special characters."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 2048
    }
    
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] API request failed: {e}")
        raise  # Re-raise to handle in caller

def count_strings(value: Any) -> int:
    """Count total translatable strings in a structure."""
    if isinstance(value, str):
        return 1 if value.strip() else 0
    elif isinstance(value, dict):
        return sum(count_strings(v) for v in value.values())
    elif isinstance(value, list):
        return sum(count_strings(item) for item in value)
    return 0

def translate_value(value: Any, source_lang: str, target_lang: str, model: str, 
                    state: TranslationState, job_id: str, style: str = "friendly",
                    path: str = "", stats: dict = None) -> Any:
    """Recursively translate values in a nested structure with resume support."""
    if stats is None:
        stats = {"done": 0, "skipped": 0, "total": 0}
    
    if isinstance(value, str):
        if not value.strip():
            return value
        
        # Check if already translated
        cached = state.get_translated(job_id, path)
        if cached and cached.get("original") == value:
            stats["skipped"] += 1
            print(f"  [SKIP] {path} (already translated)")
            return cached["translated"]
        
        stats["done"] += 1
        print(f"  [{stats['done']}/{stats['total']}] Translating: {path}")
        
        try:
            translated = translate_text(value, source_lang, target_lang, model, style)
            state.mark_done(job_id, path, value, translated)
            time.sleep(0.1)
            return translated
        except Exception as e:
            print(f"  [ERROR] Failed at {path}, saving progress...")
            raise
    
    elif isinstance(value, dict):
        result = {}
        for k, v in value.items():
            new_path = f"{path}.{k}" if path else k
            result[k] = translate_value(v, source_lang, target_lang, model, state, job_id, style, new_path, stats)
        return result
    
    elif isinstance(value, list):
        result = []
        for i, item in enumerate(value):
            new_path = f"{path}[{i}]"
            result.append(translate_value(item, source_lang, target_lang, model, state, job_id, style, new_path, stats))
        return result
    
    return value

def translate_json_file(input_path: str, output_path: str, source_lang: str, 
                        target_lang: str, model: str, state: TranslationState,
                        style: str = "friendly", force_restart: bool = False):
    """Translate an entire i18n JSON file with resume capability."""
    job_id = state.get_job_id(input_path, target_lang)
    style_info = TRANSLATION_STYLES.get(style, TRANSLATION_STYLES["friendly"])
    
    print(f"\n{'='*60}")
    print(f"Job ID: {job_id}")
    print(f"Input: {input_path}")
    print(f"From: {source_lang} -> To: {target_lang}")
    print(f"Style: {style_info['name']}")
    print(f"Output: {output_path}")
    
    if force_restart:
        state.clear_job(job_id)
        print("Progress cleared - starting fresh")
    
    existing_progress = state.get_progress(job_id)
    if existing_progress > 0:
        print(f"Resuming: {existing_progress} strings already translated")
    print('='*60)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_strings = count_strings(data)
    print(f"Total strings to translate: {total_strings}")
    
    stats = {"done": 0, "skipped": 0, "total": total_strings}
    
    try:
        translated_data = translate_value(data, source_lang, target_lang, model, state, job_id, style, "", stats)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Translation complete!")
        print(f"  - Newly translated: {stats['done']}")
        print(f"  - Skipped (cached): {stats['skipped']}")
        print(f"  - Output: {output_path}")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted! Progress saved ({stats['done']} strings translated)")
        print(f"  Run the same command again to resume.")
        raise
    except Exception as e:
        print(f"\n\n⚠ Error occurred! Progress saved ({stats['done']} strings translated)")
        print(f"  Run the same command again to resume.")
        raise

def batch_translate(input_dir: str, output_dir: str, source_lang: str, 
                    target_langs: list[str], model: str, state: TranslationState,
                    style: str = "friendly", force_restart: bool = False):
    """Translate all JSON files in a directory to multiple target languages."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s)")
    
    for json_file in json_files:
        for target_lang in target_langs:
            output_file = Path(output_dir) / target_lang / json_file.name
            translate_json_file(str(json_file), str(output_file), source_lang, 
                              target_lang, model, state, style, force_restart)

def show_status(state: TranslationState):
    """Show current translation progress status."""
    print("\n📊 Translation Progress Status")
    print("="*60)
    
    if not state.state["completed"]:
        print("No translations in progress.")
        return
    
    for job_id, translations in state.state["completed"].items():
        print(f"\nJob: {job_id}")
        print(f"  Completed strings: {len(translations)}")
        if translations:
            sample_keys = list(translations.keys())[:3]
            print(f"  Sample keys: {', '.join(sample_keys)}...")

def main():
    parser = argparse.ArgumentParser(description="Translate i18n JSON files using LM Studio API")
    parser.add_argument("input", nargs="?", help="Input JSON file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory", default="./translated")
    parser.add_argument("-s", "--source", help="Source language", default="English")
    parser.add_argument("-t", "--target", help="Target language(s), comma-separated", default="Persian")
    parser.add_argument("-m", "--model", help="Model name in LM Studio", default="local-model")
    parser.add_argument("--url", help="LM Studio API URL", default="http://localhost:1234/v1/chat/completions")
    parser.add_argument("--state-file", help="State file for resume capability", default="./.translation_state.json")
    parser.add_argument("--restart", action="store_true", help="Clear progress and start fresh")
    parser.add_argument("--status", action="store_true", help="Show translation progress status")
    parser.add_argument("--clear", help="Clear progress for specific job ID or 'all'")
    parser.add_argument("--style", choices=list(TRANSLATION_STYLES.keys()), default="friendly",
                        help="Translation style/tone (default: friendly)")
    parser.add_argument("--list-styles", action="store_true", help="Show available translation styles")
    
    args = parser.parse_args()
    
    global LM_STUDIO_URL
    LM_STUDIO_URL = args.url
    
    state = TranslationState(args.state_file)
    
    # Handle list-styles command
    if args.list_styles:
        print("\n🎨 Available Translation Styles")
        print("="*60)
        for key, info in TRANSLATION_STYLES.items():
            print(f"\n  --style {key}")
            print(f"     {info['name']}")
            print(f"     {info['example']}")
        print()
        return
    
    # Handle status command
    if args.status:
        show_status(state)
        return
    
    # Handle clear command
    if args.clear:
        if args.clear == "all":
            state.state["completed"] = {}
            state.save()
            print("✓ All progress cleared")
        else:
            state.clear_job(args.clear)
            print(f"✓ Progress cleared for job: {args.clear}")
        return
    
    if not args.input:
        parser.print_help()
        return
    
    target_langs = [lang.strip() for lang in args.target.split(",")]
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            if len(target_langs) == 1:
                output = args.output if args.output.endswith('.json') else f"{args.output}/{target_langs[0]}/{input_path.name}"
                translate_json_file(args.input, output, args.source, target_langs[0], 
                                  args.model, state, args.style, args.restart)
            else:
                for target in target_langs:
                    output = f"{args.output}/{target}/{input_path.name}"
                    translate_json_file(args.input, output, args.source, target, 
                                      args.model, state, args.style, args.restart)
        elif input_path.is_dir():
            batch_translate(args.input, args.output, args.source, target_langs, 
                          args.model, state, args.style, args.restart)
        else:
            print(f"Error: {args.input} not found")
            exit(1)
    except KeyboardInterrupt:
        print("\nExiting... Progress has been saved.")
        exit(0)

if __name__ == "__main__":
    main()
