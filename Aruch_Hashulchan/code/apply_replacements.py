"""
Apply Post-Processing Replacements
===================================
Reads translated simanim from gemini_saved_data/, applies replacements
from post_processing/replacements.json, and writes results to
gemini_normalized_data/ in each section directory.

Original files are never modified.

Usage:
    python3 apply_replacements.py
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

SECTIONS = {
    "oc": BASE_DIR / "OC",
    "yd": BASE_DIR / "YD",
    "eh": BASE_DIR / "EH",
    "cm": BASE_DIR / "CM",
}

REPLACEMENTS_PATH = BASE_DIR / "post_processing" / "replacements.json"


def load_replacements():
    with open(REPLACEMENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    replacements = {}
    for group_key in ["seif_katan_variants", "seif_variants", "sefer_names"]:
        for variant, canonical in data[group_key].items():
            if not variant.startswith("_"):
                replacements[variant] = canonical
    return replacements


def build_pattern(replacements):
    # Sort longest first so 'seif katan' matches before 'seif',
    # and 'small subparagraph' before 'subparagraph', etc.
    sorted_variants = sorted(replacements.keys(), key=len, reverse=True)
    parts = [r'\b' + re.escape(v) + r'\b' for v in sorted_variants]
    return re.compile('|'.join(parts), re.IGNORECASE)


def apply_to_text(text, pattern, replacements):
    def replace_match(m):
        matched = m.group(0)
        for k, v in replacements.items():
            if k.lower() == matched.lower():
                return v
        return matched
    return pattern.sub(replace_match, text)


def process_section(slug, section_dir, pattern, replacements):
    source_dir = section_dir / "gemini_saved_data"
    dest_dir   = section_dir / "gemini_normalized_data"
    dest_dir.mkdir(parents=True, exist_ok=True)

    siman_files = sorted(
        [f for f in source_dir.glob("siman_*.json") if "partial" not in f.name],
        key=lambda p: int(re.search(r'\d+', p.name).group())
    )

    files_changed = 0
    total_subs = 0

    for path in siman_files:
        with open(path, encoding="utf-8") as f:
            seifim = json.load(f)

        if not isinstance(seifim, list):
            continue

        new_seifim = []
        siman_subs = 0
        for seif in seifim:
            if not isinstance(seif, str):
                new_seifim.append(seif)
                continue
            new_seif = apply_to_text(seif, pattern, replacements)
            if new_seif != seif:
                # Count substitutions made in this seif
                siman_subs += len(pattern.findall(seif))
            new_seifim.append(new_seif)

        dest_path = dest_dir / path.name
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(new_seifim, f, ensure_ascii=False, indent=2)

        if siman_subs > 0:
            files_changed += 1
            total_subs += siman_subs

    print(f"  {slug.upper()}: {len(siman_files)} simanim → {files_changed} changed, {total_subs:,} substitutions made")
    print(f"         output: {dest_dir}")


def main():
    replacements = load_replacements()
    pattern = build_pattern(replacements)

    print(f"Loaded {len(replacements)} replacement rules:")
    for variant, canonical in replacements.items():
        print(f"  '{variant}' → '{canonical}'")
    print()

    for slug, section_dir in SECTIONS.items():
        process_section(slug, section_dir, pattern, replacements)

    print("\nDone. Original files in gemini_saved_data/ are unchanged.")


if __name__ == "__main__":
    main()
