"""
Thinking Level Experiment — Aruch HaShulchan
=============================================
Samples 5 random seifim from each of OC, YD, EH, CM and translates each
at all four Gemini thinking levels: MINIMAL, LOW, MEDIUM, HIGH.

Tracks per-call: thinking tokens, input tokens, output tokens, latency.

Results saved to: thinking_experiment_results/results.json

Usage:
    python3 thinking_level_experiment.py
"""

import json
import re
import time
import os
import random
from pathlib import Path
from datetime import datetime, timezone
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
GEMINI_MODEL = "gemini-3-flash-preview"
RANDOM_SEED = 42
SAMPLES_PER_SECTION = 5
RESULTS_DIR = BASE_DIR / "thinking_experiment_results"

THINKING_LEVELS = ["MINIMAL", "LOW", "MEDIUM", "HIGH"]

# Same output token budget for all levels so thinking level is the only variable.
# 24K is large enough to accommodate HIGH-level thinking tokens (~12-15K) plus output.
MAX_OUTPUT_TOKENS = 24_000

SECTIONS = {
    "oc": {
        "label": "Orach Chaim",
        "he_source": BASE_DIR / "OC" / "Arukh HaShulchan - he - Arukh HaShulchan, Orach Chayim -- Wikisource.json",
        "he_key": ["text", "Orach Chaim"],
    },
    "yd": {
        "label": "Yoreh De'ah",
        "he_source": BASE_DIR / "YD" / "Arukh_HaShulchan_Yoreh_Deah_Merged.json",
        "he_key": ["text", "Yoreh De'ah"],
    },
    "eh": {
        "label": "Even HaEzer",
        "he_source": BASE_DIR / "EH" / "Arukh HaShulchan - he - Aruch HaShulchan, Vilna 1923-29.json",
        "he_key": ["text", "Even HaEzer", ""],
    },
    "cm": {
        "label": "Choshen Mishpat",
        "he_cache": BASE_DIR / "CM" / "he_source.json",
    },
}

# ── Gemini client ─────────────────────────────────────────────────────────────

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in .env file")

client = genai.Client(api_key=api_key)

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

PROMPT_TEMPLATE = """Translate the following passage from the Aruch HaShulchan into English. \
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
"""


# ── API call ──────────────────────────────────────────────────────────────────

def call_gemini_at_level(passage, level):
    """Translate a passage at the given thinking level. Returns (translation, metrics)."""
    prompt = PROMPT_TEMPLATE.format(examples=FEW_SHOT_EXAMPLES, passage=passage)
    max_tokens = MAX_OUTPUT_TOKENS

    t_start = time.time()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=level),
        )
    )
    latency_s = time.time() - t_start

    if not response.parts:
        finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
        raise ValueError(f"Response blocked (finish_reason={finish_reason})")

    usage = response.usage_metadata
    metrics = {
        "tokens_in":       getattr(usage, "prompt_token_count", None),
        "tokens_out":      getattr(usage, "candidates_token_count", None),
        "tokens_thinking": getattr(usage, "thoughts_token_count", None),
        "tokens_cached":   getattr(usage, "cached_content_token_count", None),
        "tokens_total":    getattr(usage, "total_token_count", None),
        "latency_s":       round(latency_s, 2),
    }

    raw = response.text
    match = re.search(r"<translation>(.*?)</translation>", raw, re.DOTALL)
    translation = match.group(1).strip() if match else raw.replace("<translation>", "").replace("</translation>", "").strip()

    return translation, metrics


def call_with_retry(passage, level, max_retries=8):
    for attempt in range(max_retries):
        try:
            return call_gemini_at_level(passage, level)
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg
            if attempt < max_retries - 1:
                wait = min(2 ** (attempt + 2), 128) if is_rate_limit else 30
                if wait == 128:
                    wait = 600
                print(f"      Attempt {attempt+1} failed, waiting {wait}s: {e}")
                time.sleep(wait)
            else:
                raise Exception(f"Failed after {max_retries} attempts: {e}")


# ── Hebrew source loading ─────────────────────────────────────────────────────

def load_section_hebrew(slug):
    cfg = SECTIONS[slug]
    if slug == "cm":
        cache_path = cfg["he_cache"]
        if not cache_path.exists():
            raise FileNotFoundError(f"CM Hebrew cache not found at {cache_path}. Run translate_all_gemini.py cm first.")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(cfg["he_source"], encoding="utf-8") as f:
            data = json.load(f)
        obj = data
        for key in cfg["he_key"]:
            obj = obj[key]
        return obj


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_seifim(all_simanim, n, rng):
    """
    Randomly sample n (siman_idx, seif_idx) pairs from all_simanim.
    Only selects from non-empty simanim with non-empty seifim (strings with actual content).
    Returns list of dicts: {siman_idx, seif_idx, siman_num (1-based), seif_num (0-based), hebrew}.
    """
    candidates = []
    for siman_idx, siman in enumerate(all_simanim):
        if not siman:
            continue
        for seif_idx, seif_text in enumerate(siman):
            if isinstance(seif_text, str) and seif_text.strip():
                candidates.append((siman_idx, seif_idx, seif_text.strip()))
    selected = rng.sample(candidates, n)
    return [
        {
            "siman_idx": s[0],
            "siman_num": s[0] + 1,
            "seif_idx":  s[1],
            "seif_num":  s[1],
            "hebrew":    s[2],
        }
        for s in selected
    ]


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    rng = random.Random(RANDOM_SEED)

    # Load existing results if resuming
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Resuming — loaded {sum(len(v) for v in all_results.values())} existing sample results.")
    else:
        all_results = {}

    print(f"Model: {GEMINI_MODEL}")
    print(f"Thinking levels: {THINKING_LEVELS}")
    print(f"Samples per section: {SAMPLES_PER_SECTION}\n")

    for slug, cfg in SECTIONS.items():
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"  {label} ({slug.upper()})")
        print(f"{'='*60}")

        all_simanim = load_section_hebrew(slug)
        print(f"  {len(all_simanim)} simanim loaded.")

        # Use a per-section rng with a fixed offset so samples are stable across runs.
        section_offset = {"oc": 0, "yd": 1000, "eh": 2000, "cm": 3000}
        section_rng = random.Random(RANDOM_SEED + section_offset.get(slug, 9999))
        samples = sample_seifim(all_simanim, SAMPLES_PER_SECTION, section_rng)

        section_key = slug
        if section_key not in all_results:
            all_results[section_key] = []

        for sample in samples:
            siman_num = sample["siman_num"]
            seif_num  = sample["seif_num"]
            hebrew    = sample["hebrew"]

            # Check if this sample already has all levels done
            existing = next(
                (r for r in all_results[section_key]
                 if r["siman_num"] == siman_num and r["seif_num"] == seif_num),
                None
            )
            if existing is None:
                existing = {
                    "siman_num": siman_num,
                    "seif_num":  seif_num,
                    "hebrew":    hebrew,
                    "levels":    {},
                }
                all_results[section_key].append(existing)

            for level in THINKING_LEVELS:
                if level in existing["levels"]:
                    print(f"  Siman {siman_num} seif {seif_num} @ {level} — already done, skipping.")
                    continue

                print(f"  Siman {siman_num} seif {seif_num} @ {level} ...", end="", flush=True)
                translation, metrics = call_with_retry(hebrew, level)

                existing["levels"][level] = {
                    "translation": translation,
                    **metrics,
                }

                print(
                    f" done  "
                    f"in={metrics['tokens_in']} "
                    f"out={metrics['tokens_out']} "
                    f"thinking={metrics['tokens_thinking']} "
                    f"latency={metrics['latency_s']}s"
                )

                # Save after every call
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"  {label} complete.")

    print(f"\nAll done. Results saved to {results_path}")
    print_summary(all_results)


def print_summary(all_results):
    print(f"\n{'='*60}")
    print("  SUMMARY — avg metrics by thinking level")
    print(f"{'='*60}")

    from collections import defaultdict
    totals = defaultdict(lambda: {"thinking": [], "latency": [], "out": []})

    for slug, samples in all_results.items():
        for sample in samples:
            for level, data in sample["levels"].items():
                totals[level]["thinking"].append(data.get("tokens_thinking") or 0)
                totals[level]["latency"].append(data.get("latency_s") or 0)
                totals[level]["out"].append(data.get("tokens_out") or 0)

    print(f"  {'Level':<10} {'Avg thinking tok':>18} {'Avg output tok':>15} {'Avg latency (s)':>16}")
    print(f"  {'-'*10} {'-'*18} {'-'*15} {'-'*16}")
    for level in THINKING_LEVELS:
        d = totals[level]
        if not d["thinking"]:
            continue
        avg_think  = sum(d["thinking"]) / len(d["thinking"])
        avg_out    = sum(d["out"])      / len(d["out"])
        avg_lat    = sum(d["latency"])  / len(d["latency"])
        print(f"  {level:<10} {avg_think:>18.0f} {avg_out:>15.0f} {avg_lat:>16.2f}")


if __name__ == "__main__":
    run_experiment()
