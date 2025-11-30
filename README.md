# 🌐 i18n Translator

A powerful, multi-format translation tool using LM Studio's local LLM API. Features parallel processing, smart caching, batch API calls, and support for all major i18n file formats.

## ✨ Features

- **🚀 High Performance**
  - Parallel translation with configurable workers
  - Batch API calls (multiple texts per request)
  - Smart caching for repeated translations
  - Resume capability - interrupt anytime

- **📁 Multi-Format Support**
  - JSON (nested structures)
  - YAML / YML
  - PO / POT (gettext)
  - CSV
  - Android XML (strings.xml)
  - iOS Strings (.strings)

- **🎨 Translation Styles**
  - Formal, Friendly, Casual, Playful, Neutral

- **🖥️ Beautiful UI**
  - Rich progress bars (optional)
  - Detailed statistics

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/user/i18n-translator.git
cd i18n-translator

# Install all dependencies
pip install -r requirements.txt

# Or install only required dependencies
pip install requests
```

## 🚀 Quick Start

```bash
# Start LM Studio with a model loaded, enable local server (port 1234)

# Translate JSON file
python i18n_translator.py en.json -t German

# Translate with formal style
python i18n_translator.py en.json -t German --style formal

# Translate to multiple languages
python i18n_translator.py en.json -t "German,French,Spanish"
```

## 📖 Usage

### Basic Commands

```bash
# Single file
python i18n_translator.py input.json -o output.json -t German

# Directory (recursive, all formats)
python i18n_translator.py ./locales/en -o ./locales -t German

# Multiple languages
python i18n_translator.py en.json -o ./locales -t "German,French,Italian"
```

### Performance Options

```bash
# Faster: More parallel workers and bigger batches
python i18n_translator.py en.json -t German --workers 5 --batch-size 10

# Slower but safer: Sequential processing
python i18n_translator.py en.json -t German --workers 1 --batch-size 1
```

### Resume & Cache

```bash
# Resume interrupted translation (automatic)
python i18n_translator.py en.json -t German

# Start fresh, ignore progress
python i18n_translator.py en.json -t German --restart

# Clear translation cache
python i18n_translator.py --clear-cache

# Check status
python i18n_translator.py --status
```

## 📁 Supported Formats

| Format | Extension | Example |
|--------|-----------|---------|
| JSON | `.json` | `{"hello": "Hello"}` |
| YAML | `.yaml` `.yml` | `hello: Hello` |
| Gettext | `.po` `.pot` | `msgid "Hello"` |
| CSV | `.csv` | `key,source,translation` |
| Android | `.xml` | `<string name="hello">Hello</string>` |
| iOS | `.strings` | `"hello" = "Hello";` |

## 🎨 Translation Styles

| Style | Description | Use Case |
|-------|-------------|----------|
| `formal` | Professional, polite (Sie/vous) | Banking, Enterprise |
| `friendly` | Warm, conversational (du/tu) | Consumer apps |
| `casual` | Very informal, colloquial | Chat apps |
| `playful` | Fun, energetic | Games |
| `neutral` | Clear, balanced | Documentation |

```bash
python i18n_translator.py --list-styles
```

## ⚙️ Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `./translated` | Output path |
| `-s, --source` | `English` | Source language |
| `-t, --target` | `German` | Target language(s) |
| `-m, --model` | `local-model` | LM Studio model |
| `--style` | `friendly` | Translation style |
| `--batch-size` | `5` | Texts per API call |
| `--workers` | `3` | Parallel workers |
| `--url` | `localhost:1234` | LM Studio URL |
| `--restart` | - | Clear progress |
| `--status` | - | Show status |
| `--clear-cache` | - | Clear cache |

## 📂 Example

### Input Structure
```
locales/en/
├── common.json
├── pages/
│   └── home.json
└── components/
    └── buttons.json
```

### Command
```bash
python i18n_translator.py ./locales/en -o ./locales -t "German,French"
```

### Output Structure
```
locales/
├── de/
│   ├── common.json
│   ├── pages/home.json
│   └── components/buttons.json
└── fr/
    ├── common.json
    ├── pages/home.json
    └── components/buttons.json
```

## 🔧 Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) with loaded model

### Dependencies

| Package | Required | Description |
|---------|----------|-------------|
| `requests` | ✅ Yes | HTTP client for API calls |
| `rich` | ❌ Optional | Beautiful progress bars |
| `pyyaml` | ❌ Optional | YAML file support |

Install all: `pip install -r requirements.txt`

## 📄 License

MIT License