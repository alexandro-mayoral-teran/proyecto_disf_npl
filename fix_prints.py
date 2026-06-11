import os
import re

file_path = os.path.join(os.path.dirname(__file__), 'src', 'nlp_core', 'generacion.py')

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Emojis to remove or replace
replacements = {
    "🔄 ": "[*] ",
    "🚨 ": "[!] ",
    "✅ ": "[OK] ",
    "⚠️ ": "[WARN] ",
    "❌ ": "[FAIL] ",
    "🧠 ": "[CACHE] ",
    "⚖️ ": "[JUDGE] ",
    "💾 ": "[SAVE] ",
    "📦 ": "[LOAD] ",
    "⚙️ ": "[EXTRACT] ",
    "📊 ": "[DATA] ",
    "⏱️ ": "[TIME] ",
    "📝 ": "[NOTE] ",
    "🔒 ": "[LOCKED] "
}

for emoji, replacement in replacements.items():
    content = content.replace(emoji, replacement)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Removed all emojis from generacion.py")
