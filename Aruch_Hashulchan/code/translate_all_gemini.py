"""
Aruch HaShulchan — Full Translation with Gemini 3 Flash
Translates all sections: Orach Chaim, Yoreh De'ah, Even HaEzer, Choshen Mishpat.

Hebrew source loaded from local JSON files (or Sefaria API for Choshen Mishpat).
Results saved with seif-level checkpointing — safe to restart at any time.

Usage:
    python3 translate_all_gemini.py           # translate all sections
    python3 translate_all_gemini.py oc        # Orach Chaim only
    python3 translate_all_gemini.py yd        # Yoreh De'ah only
    python3 translate_all_gemini.py eh        # Even HaEzer only
    python3 translate_all_gemini.py cm        # Choshen Mishpat only
"""

import json
import re
import time
import os
import sys
import requests
from datetime import datetime, timezone
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
GEMINI_MODEL = "gemini-3-flash-preview"

SECTIONS = {
    "oc": {
        "label": "Orach Chaim",
        "he_source": BASE_DIR / "OC" / "Arukh HaShulchan - he - Arukh HaShulchan, Orach Chayim -- Wikisource.json",
        "he_key": ["text", "Orach Chaim"],
        "save_dir": BASE_DIR / "OC" / "gemini_saved_data",
        "out_path": BASE_DIR / "OC" / "ah_oc_gemini_translation_full.json",
        "out_key": ["text", "Orach Chaim"],
        "he_cache": None,  # loaded from local file
    },
    "yd": {
        "label": "Yoreh De'ah",
        "he_source": BASE_DIR / "YD" / "Arukh_HaShulchan_Yoreh_Deah_Merged.json",
        "he_key": ["text", "Yoreh De'ah"],
        "save_dir": BASE_DIR / "YD" / "gemini_saved_data",
        "out_path": BASE_DIR / "YD" / "ah_yd_gemini_translation_full.json",
        "out_key": ["text", "Yoreh De'ah"],
        "he_cache": None,
    },
    "eh": {
        "label": "Even HaEzer",
        "he_source": BASE_DIR / "EH" / "Arukh HaShulchan - he - Aruch HaShulchan, Vilna 1923-29.json",
        "he_key": ["text", "Even HaEzer", ""],
        "save_dir": BASE_DIR / "EH" / "gemini_saved_data",
        "out_path": BASE_DIR / "EH" / "ah_eh_gemini_translation_full.json",
        "out_key": ["text", "Even HaEzer", ""],
        "he_cache": None,
    },
    "cm": {
        "label": "Choshen Mishpat",
        "he_source": None,  # fetched from Sefaria API
        "he_key": None,
        "save_dir": BASE_DIR / "CM" / "gemini_saved_data",
        "out_path": BASE_DIR / "CM" / "ah_cm_gemini_translation_full.json",
        "out_key": ["text", "Choshen Mishpat"],
        "he_cache": BASE_DIR / "CM" / "he_source.json",
        "num_simanim": 380,
    },
}


# ── Gemini client ─────────────────────────────────────────────────────────────

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in .env file")

client = genai.Client(api_key=api_key)
print(f"Model: {GEMINI_MODEL}\n")

SYSTEM_INSTRUCTION = """You are an expert rabbinic research translator specializing in halachic literature. \
You have deep knowledge of Jewish law, rabbinic Hebrew, Aramaic, and the legal reasoning style of \
19th-century Eastern European poskim. Your translations are precise, faithful to the original, and \
preserve the legal terminology and structure of the source text. You do not add explanatory glosses \
or paraphrases unless they are clearly present in the original Hebrew.

STYLE GUIDE — follow these conventions consistently:

Rabbinic titles:
- "רבינו X" → "our teacher, the X" (e.g. "our teacher, the Rema") — never "Rabbeinu the X"
- "הרמב\"ם" → always "the Rambam" (never "Maimonides")
- "הרי\"ף" → "the Rif"
- "הרא\"ש" → "the Rosh"
- "הטור" → "the Tur"
- "הבית יוסף" → "the Beit Yosef"
- "הרמ\"א" → "the Rema"
- "רבא" → "Rava"

Standard phrases:
- "עד כאן לשונו" → "Thus far his words." (always as a closing sentence)
- "וזה לשונו" / "זה לשונו" → "This is his wording:" (introduce a block quote)
- "עיין שם" → "see there"
- "כמו שנתבאר" → "as has been explained"
- "בסייעתא דשמיא" → "with the help of Heaven"
- "ז\"ל" → "of blessed memory"
- "שם" (referring to a Talmudic location already cited) → "ibid."

General:
- Keep halachic terms in transliteration: tzitzit, tekhelet, shaatnez, seif, siman, beit din, posek, etc.
- Preserve paragraph breaks and the logical flow of legal argument.
- Do not add glosses or parenthetical definitions unless present in the original.
- Your entire response must be exactly: <translation>[translated English text]</translation> — nothing before the opening tag, nothing after the closing tag. No commentary, no reasoning, no checklists."""

FEW_SHOT_EXAMPLES = """
<examples>
<example>
<passage>
והרמב"ם ז"ל כתב בריש פרק שני: תכלת האמורה בתורה בכל מקום היא הצמר הצבוע בפתוך שבכחול. וזה לשונו: צריך שתהיה צביעתה צביעה ידועה שעומדת ביופיה. ועיין שם. ורבינו הרמ"א בסעיף א פסק כן.
</passage>
<translation>
The Rambam, of blessed memory, wrote at the beginning of the second chapter: The tekhelet mentioned in the Torah everywhere is wool dyed with a specific blue dye. This is his wording: It must be dyed with a known dye that retains its beauty. See there. Our teacher, the Rema, in section 1 ruled accordingly.
</translation>
</example>
<example>
<passage>
וזה לשון הרא"ש שם: כיון שאמרו חכמים שב ואל תעשה יש בידם לעקור דבר מן התורה. עד כאן לשונו. ורבינו הבית יוסף חלק עליו, כמו שנתבאר.
</passage>
<translation>
This is the wording of the Rosh there: Since the sages ruled "sit and do not act," they have the authority to uproot a Torah commandment. Thus far his words. Our teacher, the Beit Yosef, disagreed with him, as has been explained.
</translation>
</example>
</examples>
"""


def _call_gemini(prompt, temperature=0.4, max_tokens=8000):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        )
    )
    if not response.parts:
        finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
        raise ValueError(f"Response blocked by Gemini (finish_reason={finish_reason})")
    usage = response.usage_metadata
    metadata = {
        "tokens_in": getattr(usage, "prompt_token_count", None),
        "tokens_out": getattr(usage, "candidates_token_count", None),
        "tokens_cached": getattr(usage, "cached_content_token_count", None),
        "tokens_thinking": getattr(usage, "thoughts_token_count", None),
        "tokens_total": getattr(usage, "total_token_count", None),
    }
    return response.text, metadata


def call_gemini(prompt, temperature=0.4, max_tokens=8000):
    max_retries = 8
    for attempt in range(max_retries):
        try:
            return _call_gemini(prompt, temperature, max_tokens)
        except Exception as e:
            error_msg = str(e)
            is_resource_exhausted = "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg

            if attempt < max_retries - 1:
                if is_resource_exhausted:
                    backoff = min(2 ** (attempt + 2), 128)
                    if backoff == 128:
                        wait = 600
                        print(f"Attempt {attempt + 1}/{max_retries} — RESOURCE_EXHAUSTED at max backoff, waiting 10 minutes: {e}")
                    else:
                        wait = backoff
                        print(f"Attempt {attempt + 1}/{max_retries} — RESOURCE_EXHAUSTED, backing off {wait}s: {e}")
                else:
                    wait = 30
                    print(f"Attempt {attempt + 1}/{max_retries} — retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {e}")


def analyze_passage(passage):
    prompt = """Translate the following passage from the Aruch HaShulchan into English. \
Follow the style guide and conventions in your instructions exactly.

{examples}

CRITICAL RULES — violation of any of these is a failure:
1. Your entire response must be exactly: <translation>[translated English text]</translation> — nothing before the opening tag, nothing after the closing tag.
2. Do NOT include any commentary, preamble, self-correction notes, style checks, or reasoning anywhere in your response.
3. Follow all style guide conventions (titles, standard phrases, transliterations).
4. Do not add glosses or parenthetical definitions unless present in the original Hebrew.
5. Maintain the legal reasoning structure and paragraph breaks of the original.

<passage>
{passage}
</passage>
""".format(examples=FEW_SHOT_EXAMPLES, passage=passage)
    result, metadata = call_gemini(prompt, temperature=0.4, max_tokens=8000)
    match = re.search(r"<translation>(.*?)</translation>", result, re.DOTALL)
    if match:
        return match.group(1).strip(), metadata
    return result.replace("<translation>", "").replace("</translation>", "").strip(), metadata


# ── Hebrew source loading ─────────────────────────────────────────────────────

def load_he_local(section_cfg):
    """Load Hebrew simanim from a local JSON file."""
    with open(section_cfg["he_source"], encoding="utf-8") as f:
        data = json.load(f)
    # Traverse nested keys
    obj = data
    for key in section_cfg["he_key"]:
        obj = obj[key]
    return obj


def fetch_siman_he_cm(siman_num):
    """Fetch a single CM siman from the Sefaria API."""
    url = f"https://www.sefaria.org/api/texts/Arukh_HaShulchan,_Choshen_Mishpat.{siman_num}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("he", [])


def load_he_cm(section_cfg):
    """Load CM Hebrew from cache or fetch from Sefaria."""
    cache_path = section_cfg["he_cache"]
    if cache_path.exists():
        print("  Loading cached CM Hebrew source...")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    num = section_cfg["num_simanim"]
    print(f"  Fetching {num} CM simanim from Sefaria...")
    all_he = []
    for i in range(1, num + 1):
        if i % 20 == 0:
            print(f"    {i}/{num}...")
        try:
            all_he.append(fetch_siman_he_cm(i))
        except Exception as e:
            print(f"    Error on siman {i}: {e}")
            all_he.append([])
        time.sleep(0.3)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_he, f, ensure_ascii=False, indent=2)
    print(f"  Saved CM Hebrew cache to {cache_path}")
    return all_he


# ── Translation with seif-level checkpointing ─────────────────────────────────

def log_usage(usage_log_path, entry):
    with open(usage_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def translate_section(slug, section_cfg):
    label = section_cfg["label"]
    save_dir = section_cfg["save_dir"]
    out_path = section_cfg["out_path"]
    save_dir.mkdir(parents=True, exist_ok=True)
    usage_log_path = save_dir / "usage_log.jsonl"

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Load Hebrew
    if slug == "cm":
        all_he_simanim = load_he_cm(section_cfg)
    else:
        all_he_simanim = load_he_local(section_cfg)

    print(f"  {len(all_he_simanim)} simanim loaded.")

    # Translate siman by siman
    for i, siman_data in enumerate(all_he_simanim):
        print(f"  __ siman {i + 1} __")
        save_path = save_dir / f"siman_{i}.json"
        partial_path = save_dir / f"siman_{i}_partial.json"

        if save_path.exists():
            print(f"    Skipping — already translated.")
            continue

        if not siman_data:
            print(f"    Skipping — empty.")
            continue

        # Resume from partial if available
        if partial_path.exists():
            with open(partial_path, encoding="utf-8") as f:
                temp_siman_list = json.load(f)
            start_seif = len(temp_siman_list)
            print(f"    Resuming from seif {start_seif}.")
        else:
            temp_siman_list = []
            start_seif = 0

        for seif_num in range(start_seif, len(siman_data)):
            seif_data = siman_data[seif_num]
            print(f"    seif {seif_num}")
            result, metadata = analyze_passage(seif_data)
            temp_siman_list.append(result)

            # Log usage
            log_usage(usage_log_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "section": slug,
                "siman": i + 1,
                "seif": seif_num,
                **metadata,
            })
            print(f"      in={metadata['tokens_in']} out={metadata['tokens_out']} thinking={metadata['tokens_thinking']} cached={metadata['tokens_cached']}")

            # Seif-level checkpoint
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(temp_siman_list, f, ensure_ascii=False, indent=2)

        # Siman complete
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(temp_siman_list, f, ensure_ascii=False, indent=2)
        if partial_path.exists():
            os.remove(partial_path)
        print(f"    Saved siman {i + 1}.")

    # Assemble final JSON
    assemble(slug, section_cfg, all_he_simanim)


def assemble(slug, section_cfg, all_he_simanim):
    save_dir = section_cfg["save_dir"]
    out_path = section_cfg["out_path"]
    out_keys = section_cfg["out_key"]

    translated_simanim = []
    for i in range(len(all_he_simanim)):
        save_path = save_dir / f"siman_{i}.json"
        if save_path.exists():
            with open(save_path, encoding="utf-8") as f:
                translated_simanim.append(json.load(f))
        else:
            translated_simanim.append([])

    # Build nested output dict from out_keys
    # e.g. ["text", "Even HaEzer", ""] → {"text": {"Even HaEzer": {"": [...]}}}
    output = {}
    obj = output
    for key in out_keys[:-1]:
        obj[key] = {}
        obj = obj[key]
    obj[out_keys[-1]] = translated_simanim

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    non_empty = sum(1 for s in translated_simanim if s)
    print(f"\n  Assembled {non_empty}/{len(translated_simanim)} simanim → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slugs = [s.lower() for s in sys.argv[1:]]
        unknown = [s for s in slugs if s not in SECTIONS]
        if unknown:
            print(f"Unknown section(s): {unknown}. Choose from: {list(SECTIONS.keys())}")
            sys.exit(1)
    else:
        slugs = list(SECTIONS.keys())

    for slug in slugs:
        translate_section(slug, SECTIONS[slug])

    print("\nAll done.")
