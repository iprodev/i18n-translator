#!/usr/bin/env python3
"""
i18n Translator - Multi-format translation tool using LM Studio API
Supports: JSON, YAML, PO/POT, CSV, Android XML, iOS Strings
Features: Batch processing, parallel translation, smart caching, resume capability
"""

import json
import argparse
import time
import hashlib
import re
import csv
import sys
import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading

# Global flag for graceful shutdown
shutdown_event = threading.Event()

# Optional imports with fallbacks
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_BATCH_SIZE = 5

# Translation style presets
TRANSLATION_STYLES = {
    "formal": {
        "name": "Formal",
        "prompt": "Use formal, professional language. Use polite pronouns (Sie in German, vous in French, etc.).",
    },
    "friendly": {
        "name": "Friendly", 
        "prompt": "Use warm, friendly, conversational language. Use informal pronouns (du in German, tu in French, etc.). Sound natural like talking to a friend.",
    },
    "casual": {
        "name": "Casual",
        "prompt": "Use very casual, everyday spoken language. Use colloquial expressions and informal grammar.",
    },
    "playful": {
        "name": "Playful",
        "prompt": "Use fun, playful, energetic language. Add personality and warmth.",
    },
    "neutral": {
        "name": "Neutral",
        "prompt": "Use clear, neutral language. Balanced and accessible.",
    }
}

console = Console() if RICH_AVAILABLE else None

# ============== Smart Cache ==============
class TranslationCache:
    """Smart cache for repeated translations."""
    
    def __init__(self, cache_file: str = ".translation_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.hits = 0
        self.misses = 0
    
    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def _make_key(self, text: str, source: str, target: str, style: str) -> str:
        return hashlib.md5(f"{text}:{source}:{target}:{style}".encode()).hexdigest()
    
    def get(self, text: str, source: str, target: str, style: str) -> str | None:
        key = self._make_key(text, source, target, style)
        result = self.cache.get(key)
        if result:
            self.hits += 1
        else:
            self.misses += 1
        return result
    
    def set(self, text: str, source: str, target: str, style: str, translation: str):
        key = self._make_key(text, source, target, style)
        self.cache[key] = translation
    
    def get_stats(self) -> tuple[int, int]:
        return self.hits, self.misses


# ============== State Management ==============
class TranslationState:
    """Manages translation progress and resume capability."""
    
    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state = self._load_state()
    
    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        return {"completed": {}, "partial": {}}
                    return json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠ State file issue, starting fresh: {e}")
                return {"completed": {}, "partial": {}}
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
        data = self.state["completed"].get(job_id, {}).get(key_path)
        return data.get("translated") if data else None
    
    def mark_done(self, job_id: str, key_path: str, original: str, translated: str):
        if job_id not in self.state["completed"]:
            self.state["completed"][job_id] = {}
        self.state["completed"][job_id][key_path] = {"original": original, "translated": translated}
    
    def get_progress(self, job_id: str) -> int:
        return len(self.state["completed"].get(job_id, {}))
    
    def clear_job(self, job_id: str):
        if job_id in self.state["completed"]:
            del self.state["completed"][job_id]
            self.save()


# ============== File Format Handlers ==============
class FileHandler(ABC):
    """Abstract base class for file format handlers."""
    
    @abstractmethod
    def load(self, path: str) -> dict[str, str]:
        """Load file and return flat dict of key -> text."""
        pass
    
    @abstractmethod
    def save(self, path: str, data: dict[str, str], original_data: Any = None):
        """Save translated data to file."""
        pass
    
    @staticmethod
    def get_handler(path: str) -> 'FileHandler':
        """Get appropriate handler for file type."""
        ext = Path(path).suffix.lower()
        handlers = {
            '.json': JsonHandler(),
            '.yaml': YamlHandler(),
            '.yml': YamlHandler(),
            '.po': PoHandler(),
            '.pot': PoHandler(),
            '.csv': CsvHandler(),
            '.xml': AndroidXmlHandler(),
            '.strings': IosStringsHandler(),
        }
        if ext not in handlers:
            raise ValueError(f"Unsupported file format: {ext}")
        if ext in ['.yaml', '.yml'] and not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")
        return handlers[ext]


class JsonHandler(FileHandler):
    """Handler for JSON i18n files."""
    
    def _flatten(self, data: Any, prefix: str = "") -> dict[str, str]:
        result = {}
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{prefix}.{k}" if prefix else k
                result.update(self._flatten(v, new_key))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                new_key = f"{prefix}[{i}]"
                result.update(self._flatten(v, new_key))
        elif isinstance(data, str) and data.strip():
            result[prefix] = data
        return result
    
    def _unflatten(self, flat: dict[str, str], original: Any) -> Any:
        if isinstance(original, dict):
            result = {}
            for k, v in original.items():
                prefix = k
                if isinstance(v, str):
                    result[k] = flat.get(prefix, v)
                else:
                    nested_flat = {key[len(prefix)+1:]: val for key, val in flat.items() if key.startswith(prefix + ".") or key.startswith(prefix + "[")}
                    result[k] = self._unflatten(nested_flat, v)
            return result
        elif isinstance(original, list):
            result = []
            for i, v in enumerate(original):
                prefix = f"[{i}]"
                if isinstance(v, str):
                    result.append(flat.get(prefix.lstrip("."), v))
                else:
                    nested_flat = {key[len(str(i))+2:]: val for key, val in flat.items() if key.startswith(f"[{i}]")}
                    result.append(self._unflatten(nested_flat, v))
            return result
        elif isinstance(original, str):
            return flat.get("", original)
        return original
    
    def load(self, path: str) -> tuple[dict[str, str], Any]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._flatten(data), data
    
    def save(self, path: str, flat_data: dict[str, str], original_data: Any = None):
        if original_data is not None:
            data = self._unflatten_full(flat_data, original_data)
        else:
            data = flat_data
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _unflatten_full(self, flat: dict[str, str], original: Any, prefix: str = "") -> Any:
        if isinstance(original, dict):
            result = {}
            for k, v in original.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                result[k] = self._unflatten_full(flat, v, new_prefix)
            return result
        elif isinstance(original, list):
            result = []
            for i, v in enumerate(original):
                new_prefix = f"{prefix}[{i}]"
                result.append(self._unflatten_full(flat, v, new_prefix))
            return result
        elif isinstance(original, str):
            return flat.get(prefix, original)
        return original


class YamlHandler(FileHandler):
    """Handler for YAML i18n files."""
    
    def __init__(self):
        self.json_handler = JsonHandler()
    
    def load(self, path: str) -> tuple[dict[str, str], Any]:
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return self.json_handler._flatten(data), data
    
    def save(self, path: str, flat_data: dict[str, str], original_data: Any = None):
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")
        if original_data is not None:
            data = self.json_handler._unflatten_full(flat_data, original_data)
        else:
            data = flat_data
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


class PoHandler(FileHandler):
    """Handler for PO/POT gettext files."""
    
    def load(self, path: str) -> tuple[dict[str, str], list]:
        entries = []
        flat = {}
        current = {"msgid": "", "msgstr": "", "comments": []}
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.rstrip('\n')
            if line.startswith('#'):
                current["comments"].append(line)
            elif line.startswith('msgid '):
                if current["msgid"]:
                    entries.append(current.copy())
                    if current["msgid"].strip('"'):
                        flat[current["msgid"].strip('"')] = current["msgstr"].strip('"')
                current = {"msgid": line[6:], "msgstr": "", "comments": current["comments"] if not current["msgid"] else []}
            elif line.startswith('msgstr '):
                current["msgstr"] = line[7:]
            elif line.startswith('"') and current["msgstr"]:
                current["msgstr"] += line
            elif line.startswith('"') and current["msgid"]:
                current["msgid"] += line
        
        if current["msgid"]:
            entries.append(current)
            if current["msgid"].strip('"'):
                flat[current["msgid"].strip('"')] = current["msgstr"].strip('"')
        
        return flat, entries
    
    def save(self, path: str, flat_data: dict[str, str], original_data: list = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for entry in (original_data or []):
                for comment in entry.get("comments", []):
                    f.write(comment + '\n')
                msgid = entry["msgid"].strip('"')
                msgstr = flat_data.get(msgid, "")
                f.write(f'msgid "{msgid}"\n')
                f.write(f'msgstr "{msgstr}"\n\n')


class CsvHandler(FileHandler):
    """Handler for CSV files (key, source, translation)."""
    
    def load(self, path: str) -> tuple[dict[str, str], list]:
        flat = {}
        rows = []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                key = row.get('key', row.get('id', ''))
                text = row.get('source', row.get('text', row.get('en', '')))
                if key and text:
                    flat[key] = text
        return flat, rows
    
    def save(self, path: str, flat_data: dict[str, str], original_data: list = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'source', 'translation'])
            for key, translation in flat_data.items():
                source = ""
                if original_data:
                    for row in original_data:
                        if row.get('key', row.get('id', '')) == key:
                            source = row.get('source', row.get('text', row.get('en', '')))
                            break
                writer.writerow([key, source, translation])


class AndroidXmlHandler(FileHandler):
    """Handler for Android strings.xml files."""
    
    def load(self, path: str) -> tuple[dict[str, str], str]:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        flat = {}
        pattern = r'<string\s+name="([^"]+)"[^>]*>([^<]*)</string>'
        for match in re.finditer(pattern, content):
            name, value = match.groups()
            flat[name] = value
        
        return flat, content
    
    def save(self, path: str, flat_data: dict[str, str], original_data: str = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if original_data:
            content = original_data
            for name, translation in flat_data.items():
                pattern = rf'(<string\s+name="{re.escape(name)}"[^>]*>)[^<]*(</string>)'
                content = re.sub(pattern, rf'\1{translation}\2', content)
        else:
            lines = ['<?xml version="1.0" encoding="utf-8"?>', '<resources>']
            for name, value in flat_data.items():
                lines.append(f'    <string name="{name}">{value}</string>')
            lines.append('</resources>')
            content = '\n'.join(lines)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)


class IosStringsHandler(FileHandler):
    """Handler for iOS .strings files."""
    
    def load(self, path: str) -> tuple[dict[str, str], str]:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        flat = {}
        pattern = r'"([^"]+)"\s*=\s*"([^"]+)"\s*;'
        for match in re.finditer(pattern, content):
            key, value = match.groups()
            flat[key] = value
        
        return flat, content
    
    def save(self, path: str, flat_data: dict[str, str], original_data: str = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        for key, value in flat_data.items():
            escaped_value = value.replace('"', '\\"')
            lines.append(f'"{key}" = "{escaped_value}";')
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


# ============== Translation Engine ==============
class TranslationEngine:
    """Handles translation with batching."""
    
    def __init__(self, url: str, model: str, style: str, source_lang: str, target_lang: str,
                 cache: TranslationCache, batch_size: int = 5):
        self.url = url
        self.model = model
        self.style = style
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cache = cache
        self.batch_size = batch_size
        self.style_prompt = TRANSLATION_STYLES.get(style, TRANSLATION_STYLES["friendly"])["prompt"]
    
    def _build_batch_prompt(self, items: list[tuple[str, str]]) -> str:
        """Build prompt for batch translation."""
        texts = "\n".join([f"[{i}] {text}" for i, (key, text) in enumerate(items)])
        return f"""Translate the following texts from {self.source_lang} to {self.target_lang}.

Style: {self.style_prompt}

Rules:
- Return ONLY the translations in the exact same format: [number] translated text
- Keep placeholders like {{name}}, {{count}}, %s, %d, {{0}} unchanged
- Sound natural, not robotic

Texts:
{texts}"""

    def _parse_batch_response(self, response: str, count: int) -> list[str]:
        """Parse batch response into individual translations."""
        results = [""] * count
        pattern = r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for idx_str, text in matches:
            idx = int(idx_str)
            if 0 <= idx < count:
                results[idx] = text.strip()
        
        return results
    
    def translate_single(self, text: str) -> str:
        """Translate a single text."""
        if not text or not text.strip() or shutdown_event.is_set():
            return text
        
        # Check cache first
        cached = self.cache.get(text, self.source_lang, self.target_lang, self.style)
        if cached:
            return cached
        
        prompt = f"""Translate from {self.source_lang} to {self.target_lang}.
Style: {self.style_prompt}
Keep placeholders unchanged. Return only the translation.

Text: {text}"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"You are a native {self.target_lang} translator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048
        }
        
        try:
            import requests
            resp = requests.post(self.url, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip()
            self.cache.set(text, self.source_lang, self.target_lang, self.style, result)
            return result
        except Exception as e:
            if not shutdown_event.is_set():
                print(f"Translation error: {e}")
            return text
    
    def translate_batch(self, items: list[tuple[str, str]]) -> dict[str, str]:
        """Translate a batch of items."""
        if shutdown_event.is_set():
            return {}
            
        results = {}
        to_translate = []
        
        # Check cache for all items first
        for key, text in items:
            cached = self.cache.get(text, self.source_lang, self.target_lang, self.style)
            if cached:
                results[key] = cached
            else:
                to_translate.append((key, text))
        
        if not to_translate or shutdown_event.is_set():
            return results
        
        # Batch translate remaining
        prompt = self._build_batch_prompt(to_translate)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"You are a native {self.target_lang} translator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4096
        }
        
        try:
            import requests
            resp = requests.post(self.url, json=payload, timeout=60)
            resp.raise_for_status()
            response_text = resp.json()["choices"][0]["message"]["content"]
            translations = self._parse_batch_response(response_text, len(to_translate))
            
            for (key, original), translated in zip(to_translate, translations):
                if translated:
                    results[key] = translated
                    self.cache.set(original, self.source_lang, self.target_lang, self.style, translated)
                else:
                    results[key] = original
        except Exception as e:
            if not shutdown_event.is_set():
                print(f"\nBatch translation error: {e}")
            # Fallback to single translation
            for key, text in to_translate:
                if shutdown_event.is_set():
                    break
                results[key] = self.translate_single(text)
        
        return results
    
    def translate_all(self, flat_data: dict[str, str], state: TranslationState, job_id: str,
                      progress_callback=None) -> dict[str, str]:
        """Translate all items with batch processing."""
        results = {}
        items_to_translate = []
        
        # Filter out already completed items
        for key, text in flat_data.items():
            existing = state.get_translated(job_id, key)
            if existing:
                results[key] = existing
            else:
                items_to_translate.append((key, text))
        
        if not items_to_translate:
            return results
        
        # Split into batches
        batches = [items_to_translate[i:i+self.batch_size] for i in range(0, len(items_to_translate), self.batch_size)]
        completed = 0
        total = len(items_to_translate)
        
        # Process batches sequentially (more reliable for Ctrl+C)
        for batch in batches:
            if shutdown_event.is_set():
                break
            
            try:
                batch_results = self.translate_batch(batch)
                for key, translation in batch_results.items():
                    results[key] = translation
                    original = dict(batch).get(key, "")
                    state.mark_done(job_id, key, original, translation)
                
                completed += len(batch)
                if progress_callback:
                    progress_callback(completed, total)
                
                state.save()
                
            except KeyboardInterrupt:
                shutdown_event.set()
                break
            except Exception as e:
                if not shutdown_event.is_set():
                    print(f"\nBatch failed: {e}")
            
            time.sleep(0.05)
        
        return results


# ============== Main Functions ==============
def translate_file(input_path: str, output_path: str, source_lang: str, target_lang: str,
                   model: str, style: str, state: TranslationState, cache: TranslationCache,
                   batch_size: int, api_url: str, force_restart: bool = False):
    """Translate a single file."""
    if shutdown_event.is_set():
        return
    
    handler = FileHandler.get_handler(input_path)
    job_id = state.get_job_id(input_path, target_lang)
    
    if force_restart:
        state.clear_job(job_id)
    
    print(f"\n{'='*60}")
    print(f"📄 File: {input_path}")
    print(f"🌐 {source_lang} → {target_lang}")
    print(f"🎨 Style: {TRANSLATION_STYLES.get(style, {}).get('name', style)}")
    print(f"💾 Output: {output_path}")
    print('='*60)
    
    flat_data, original_data = handler.load(input_path)
    total = len(flat_data)
    print(f"Found {total} translatable strings")
    
    engine = TranslationEngine(
        url=api_url, model=model, style=style,
        source_lang=source_lang, target_lang=target_lang,
        cache=cache, batch_size=batch_size
    )
    
    if RICH_AVAILABLE:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Translating...", total=total)
                
                def update_progress(completed, total):
                    progress.update(task, completed=completed)
                
                results = engine.translate_all(flat_data, state, job_id, update_progress)
        except KeyboardInterrupt:
            shutdown_event.set()
            results = {k: state.get_translated(job_id, k) or v for k, v in flat_data.items()}
    else:
        def simple_progress(completed, total):
            pct = 100 * completed // total if total > 0 else 0
            print(f"\r  Progress: {completed}/{total} ({pct}%)", end="", flush=True)
        
        results = engine.translate_all(flat_data, state, job_id, simple_progress)
        print()
    
    if results and not shutdown_event.is_set():
        handler.save(output_path, results, original_data)
        hits, misses = cache.get_stats()
        print(f"\n✅ Done! Cache hits: {hits}, API calls: {misses}")
    elif results:
        # Partial save on shutdown
        handler.save(output_path, results, original_data)
        print(f"\n⚠️ Partial save completed")
    
    cache.save()


def batch_translate(input_dir: str, output_dir: str, source_lang: str, target_langs: list[str],
                    model: str, style: str, state: TranslationState, cache: TranslationCache,
                    batch_size: int, api_url: str, force_restart: bool = False):
    """Translate all supported files in a directory."""
    input_path = Path(input_dir)
    extensions = ['*.json', '*.yaml', '*.yml', '*.po', '*.pot', '*.csv', '*.xml', '*.strings']
    
    all_files = []
    for ext in extensions:
        all_files.extend(input_path.rglob(ext))
    
    if not all_files:
        print(f"No supported files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} file(s)")
    
    for file_path in all_files:
        if shutdown_event.is_set():
            break
        relative_path = file_path.relative_to(input_path)
        for target_lang in target_langs:
            if shutdown_event.is_set():
                break
            output_file = Path(output_dir) / target_lang / relative_path
            try:
                translate_file(
                    str(file_path), str(output_file), source_lang, target_lang,
                    model, style, state, cache, batch_size, api_url, force_restart
                )
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")


def show_status(state: TranslationState, cache: TranslationCache):
    """Show translation status."""
    if RICH_AVAILABLE:
        table = Table(title="Translation Status")
        table.add_column("Job ID")
        table.add_column("Completed")
        
        for job_id, data in state.state.get("completed", {}).items():
            table.add_row(job_id, str(len(data)))
        
        console.print(table)
        console.print(f"\n📦 Cache entries: {len(cache.cache)}")
    else:
        print("\n📊 Translation Status")
        print("="*40)
        for job_id, data in state.state.get("completed", {}).items():
            print(f"  {job_id}: {len(data)} strings")
        print(f"\n📦 Cache entries: {len(cache.cache)}")


def main():
    parser = argparse.ArgumentParser(description="Translate i18n files using LM Studio API")
    parser.add_argument("input", nargs="?", help="Input file or directory")
    parser.add_argument("-o", "--output", default="./translated", help="Output file/directory")
    parser.add_argument("-s", "--source", default="English", help="Source language")
    parser.add_argument("-t", "--target", default="German", help="Target language(s), comma-separated")
    parser.add_argument("-m", "--model", default="local-model", help="LM Studio model name")
    parser.add_argument("--url", default=LM_STUDIO_URL, help="LM Studio API URL")
    parser.add_argument("--style", choices=list(TRANSLATION_STYLES.keys()), default="friendly")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Texts per API call")
    parser.add_argument("--state-file", default=".translation_state.json")
    parser.add_argument("--cache-file", default=".translation_cache.json")
    parser.add_argument("--restart", action="store_true", help="Start fresh")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--clear", help="Clear job ID or 'all'")
    parser.add_argument("--list-styles", action="store_true")
    parser.add_argument("--clear-cache", action="store_true", help="Clear translation cache")
    
    args = parser.parse_args()
    
    state = TranslationState(args.state_file)
    cache = TranslationCache(args.cache_file)
    
    if args.list_styles:
        print("\n🎨 Available Styles:")
        for key, info in TRANSLATION_STYLES.items():
            print(f"  --style {key:10} {info['name']}")
        return
    
    if args.status:
        show_status(state, cache)
        return
    
    if args.clear_cache:
        Path(args.cache_file).unlink(missing_ok=True)
        print("✅ Cache cleared")
        return
    
    if args.clear:
        if args.clear == "all":
            state.state["completed"] = {}
            state.save()
        else:
            state.clear_job(args.clear)
        print(f"✅ Cleared: {args.clear}")
        return
    
    if not args.input:
        parser.print_help()
        return
    
    target_langs = [l.strip() for l in args.target.split(",")]
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            for target in target_langs:
                if shutdown_event.is_set():
                    break
                if len(target_langs) == 1 and args.output.endswith(input_path.suffix):
                    output = args.output
                else:
                    output = f"{args.output}/{target}/{input_path.name}"
                translate_file(args.input, output, args.source, target, args.model,
                             args.style, state, cache, args.batch_size, args.url, args.restart)
        elif input_path.is_dir():
            batch_translate(args.input, args.output, args.source, target_langs, args.model,
                          args.style, state, cache, args.batch_size, args.url, args.restart)
        else:
            print(f"❌ Not found: {args.input}")
    except KeyboardInterrupt:
        shutdown_event.set()
        print("\n\n⚠️ Interrupted!")
    finally:
        if shutdown_event.is_set():
            print("✅ Progress saved. Run again to resume.")
        state.save()
        cache.save()

if __name__ == "__main__":
    main()