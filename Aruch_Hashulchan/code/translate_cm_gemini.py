"""
Aruch HaShulchan — Choshen Mishpat Translation
Uses Google Gemini 3 Flash (gemini-3-flash-preview) to translate the Hebrew text seif by seif.
Hebrew source is fetched from the Sefaria API and cached locally.
Results are saved siman-by-siman and assembled into a final JSON.

Usage:
    python3 translate_cm_gemini.py
"""

import json
import time
import os
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-3-flash-preview"
SAVE_DIR = "CM/gpt4_saved_data"
HE_CACHE_PATH = "CM/he_source.json"
OUT_PATH = "CM/ah_cm_gemini_translation_full.json"
NUM_SIMANIM = 380

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("CM", exist_ok=True)


# ── Gemini client ─────────────────────────────────────────────────────────────

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in .env file")

client = genai.Client(api_key=api_key)
print(f"Model: {GEMINI_MODEL}")


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
- "עד כאן לשונו" → "Thus far his words." (always as a closing sentence, not mid-sentence)
- "וזה לשונו" / "זה לשונו" → "This is his wording:" (introduce a block quote)
- "עיין שם" → "see there"
- "כמו שנתבאר" → "as has been explained"
- "בסייעתא דשמיא" → "with the help of Heaven"
- "ז\"ל" → "of blessed memory"
- "שם" (referring to a Talmudic location already cited) → "ibid."

General:
- Keep halachic terms in transliteration: tzitzit, tekhelet, shaatnez, seif, siman, beit din, posek, etc.
- Preserve paragraph breaks and the logical flow of legal argument.
- Do not add glosses or parenthetical definitions unless present in the original."""


def _call_gemini(prompt, temperature=0.7, max_tokens=2000):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    if not response.parts:
        finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
        raise ValueError(f"Response blocked by Gemini (finish_reason={finish_reason})")
    return response.text


def call_gemini(prompt, temperature=0.4, max_tokens=2000):
    max_retries = 6
    for attempt in range(max_retries):
        try:
            return _call_gemini(prompt, temperature, max_tokens)
        except Exception as e:
            error_msg = str(e)
            is_resource_exhausted = "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg

            if attempt < max_retries - 1:
                if is_resource_exhausted:
                    wait = min(2 ** (attempt + 2), 128)
                    print(f"Attempt {attempt + 1}/{max_retries} — RESOURCE_EXHAUSTED, backing off {wait}s: {e}")
                else:
                    wait = 30
                    print(f"Attempt {attempt + 1}/{max_retries} — retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {e}")


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


def analyze_passage(passage):
    # Critical instruction first, few-shot examples, then passage in XML tags.
    # Output extracted from <translation> tags and saved directly to a database —
    # must contain only the translation text, no preamble or commentary.
    prompt = """Translate the following passage from the Aruch HaShulchan (Choshen Mishpat) into English. Follow the style guide and conventions in your instructions exactly.

{examples}

Rules:
1. Return only the translation inside <translation> tags — no preamble, no commentary, no explanation.
2. Follow all style guide conventions (titles, standard phrases, transliterations).
3. Do not add glosses or parenthetical definitions unless present in the original Hebrew.
4. Maintain the legal reasoning structure and paragraph breaks of the original.

<passage>
{passage}
</passage>

<translation>
""".format(examples=FEW_SHOT_EXAMPLES, passage=passage)
    result = call_gemini(prompt, temperature=0.4)
    # Strip closing tag if model included it
    return result.replace("<translation>", "").replace("</translation>", "").strip()


# ── Fetch Hebrew source from Sefaria ──────────────────────────────────────────

def fetch_siman_he(siman_num):
    url = f"https://www.sefaria.org/api/texts/Arukh_HaShulchan,_Choshen_Mishpat.{siman_num}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("he", [])


def load_or_fetch_hebrew():
    if os.path.exists(HE_CACHE_PATH):
        print("Loading cached Hebrew source...")
        with open(HE_CACHE_PATH, encoding="utf-8") as f:
            all_he_simanim = json.load(f)
        print(f"Loaded {len(all_he_simanim)} simanim.")
        return all_he_simanim

    print(f"Fetching {NUM_SIMANIM} simanim from Sefaria...")
    all_he_simanim = []
    for i in range(1, NUM_SIMANIM + 1):
        if i % 20 == 0:
            print(f"  {i}/{NUM_SIMANIM}...")
        try:
            all_he_simanim.append(fetch_siman_he(i))
        except Exception as e:
            print(f"  Error on siman {i}: {e}")
            all_he_simanim.append([])
        time.sleep(0.3)

    with open(HE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_he_simanim, f, ensure_ascii=False, indent=2)
    print(f"Saved to {HE_CACHE_PATH}")
    return all_he_simanim


# ── Translate ─────────────────────────────────────────────────────────────────

def translate(all_he_simanim):
    for i, siman_data in enumerate(all_he_simanim):
        print(f"__{i + 1}__")
        save_path = os.path.join(SAVE_DIR, f"siman_{i}.json")

        if os.path.exists(save_path):
            print(f"  Skipping — already translated.")
            continue

        if not siman_data:
            print(f"  Skipping — empty.")
            continue

        # Load partial progress if it exists (seif-level checkpointing)
        partial_path = os.path.join(SAVE_DIR, f"siman_{i}_partial.json")
        if os.path.exists(partial_path):
            with open(partial_path, encoding="utf-8") as f:
                temp_siman_list = json.load(f)
            start_seif = len(temp_siman_list)
            print(f"  Resuming from seif {start_seif}.")
        else:
            temp_siman_list = []
            start_seif = 0

        for seif_num in range(start_seif, len(siman_data)):
            seif_data = siman_data[seif_num]
            print(f"  __ seif {seif_num} __")
            result = analyze_passage(seif_data)
            temp_siman_list.append(result)

            # Save partial progress after every seif
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(temp_siman_list, f, ensure_ascii=False, indent=2)

        # Siman complete — save final and remove partial
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(temp_siman_list, f, ensure_ascii=False, indent=2)
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"  Saved.")


# ── Assemble final JSON ───────────────────────────────────────────────────────

def assemble(all_he_simanim):
    translated_simanim = []
    for i in range(len(all_he_simanim)):
        save_path = os.path.join(SAVE_DIR, f"siman_{i}.json")
        if os.path.exists(save_path):
            with open(save_path, encoding="utf-8") as f:
                translated_simanim.append(json.load(f))
        else:
            print(f"Missing siman {i + 1}, inserting empty list.")
            translated_simanim.append([])

    output = {"text": {"Choshen Mishpat": translated_simanim}}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    non_empty = sum(1 for s in translated_simanim if s)
    print(f"Done. {non_empty}/{len(translated_simanim)} simanim translated.")
    print(f"Saved to {OUT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_he_simanim = load_or_fetch_hebrew()
    translate(all_he_simanim)
    assemble(all_he_simanim)
