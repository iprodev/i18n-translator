# 🌐 i18n JSON Translator

A powerful Python script to translate i18n JSON files using LM Studio's local LLM API. Features resume capability, multiple translation styles, and batch processing.

## ✨ Features

- **Local LLM Translation** - Uses LM Studio API, no cloud services needed
- **Resume Capability** - Interrupt anytime and continue later without losing progress
- **Multiple Translation Styles** - Formal, friendly, casual, playful, or neutral tones
- **Nested JSON Support** - Handles complex nested objects and arrays
- **Placeholder Preservation** - Keeps `{name}`, `{{count}}`, `%s`, `%d` intact
- **Batch Processing** - Translate entire directories at once
- **Multi-language** - Translate to multiple target languages in one run
- **Cross-platform** - Works on Windows, macOS, and Linux

## 📦 Installation

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) with a loaded model

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/i18n-translator.git
cd i18n-translator

# Install dependencies
pip install requests
```

## 🚀 Quick Start

1. Start LM Studio and load your preferred model
2. Enable the local server (default: `http://localhost:1234`)
3. Run the translator:

```bash
python i18n_translator.py en.json -o de.json -t German
```

## 📖 Usage

### Basic Translation

```bash
# Translate English to German (friendly tone)
python i18n_translator.py en.json -t German

# Translate with formal tone
python i18n_translator.py en.json -t German --style formal

# Specify output file
python i18n_translator.py en.json -o ./locales/de.json -t German
```

### Multiple Languages

```bash
# Translate to multiple languages at once
python i18n_translator.py en.json -o ./locales -t "German,French,Spanish,Italian"
```

### Batch Translation

```bash
# Translate all JSON files in a directory
python i18n_translator.py ./locales/en/ -o ./locales -t German
```

### Resume Interrupted Translation

```bash
# Just run the same command again - it automatically resumes
python i18n_translator.py en.json -t German

# Start fresh, ignoring previous progress
python i18n_translator.py en.json -t German --restart
```

### Progress Management

```bash
# Check translation progress
python i18n_translator.py --status

# Clear progress for all jobs
python i18n_translator.py --clear all

# Clear specific job
python i18n_translator.py --clear <job_id>
```

## 🎨 Translation Styles

| Style | Flag | Description | Best For |
|-------|------|-------------|----------|
| Formal | `--style formal` | Professional, polite pronouns (Sie/vous/usted) | Banking, government, enterprise apps |
| Friendly | `--style friendly` | Warm, conversational (du/tu/tú) | Consumer apps, social platforms |
| Casual | `--style casual` | Very informal, colloquial | Chat apps, youth-oriented products |
| Playful | `--style playful` | Fun, energetic, with personality | Games, entertainment apps |
| Neutral | `--style neutral` | Balanced, clear, accessible | Documentation, utilities |

```bash
# List all available styles
python i18n_translator.py --list-styles
```

## ⚙️ Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output file or directory | `./translated` |
| `-s, --source` | Source language | `English` |
| `-t, --target` | Target language(s), comma-separated | `German` |
| `-m, --model` | Model name in LM Studio | `local-model` |
| `--url` | LM Studio API URL | `http://localhost:1234/v1/chat/completions` |
| `--style` | Translation style | `friendly` |
| `--state-file` | State file for resume capability | `./.translation_state.json` |
| `--restart` | Clear progress and start fresh | `false` |
| `--status` | Show translation progress status | - |
| `--clear` | Clear progress for job ID or 'all' | - |
| `--list-styles` | Show available translation styles | - |

## 📁 Example

### Input (`en.json`)

```json
{
  "welcome": "Welcome to our app!",
  "user": {
    "greeting": "Hello, {name}!",
    "messages": "You have {count} new messages"
  },
  "buttons": ["Save", "Cancel", "Delete"]
}
```

### Output (`de.json`) - Friendly Style

```json
{
  "welcome": "Willkommen in unserer App!",
  "user": {
    "greeting": "Hallo {name}!",
    "messages": "Du hast {count} neue Nachrichten"
  },
  "buttons": ["Speichern", "Abbrechen", "Löschen"]
}
```

## 🔧 Configuration

### Using a Different LM Studio Port

```bash
python i18n_translator.py en.json -t French --url http://localhost:8080/v1/chat/completions
```

### Custom State File Location

```bash
python i18n_translator.py en.json -t French --state-file ./my_progress.json
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LM Studio](https://lmstudio.ai/) for providing an easy-to-use local LLM interface
