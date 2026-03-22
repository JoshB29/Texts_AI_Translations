# AI Translation of the Aruch HaShulchan

This repository contains an AI-generated English translation of the complete *Aruch HaShulchan* by Rabbi Yechiel Michel Epstein, covering all four sections of the *Shulchan Aruch*: *Orach Chaim*, *Yoreh De'ah*, *Even HaEzer*, and *Choshen Mishpat*.

The translation is freely available online at [aruch-hashulchan-transla-53666.web.app](https://aruch-hashulchan-transla-53666.web.app).

## Current Translation (Gemini 3 Flash)

The current translation in `gemini_translation/` was produced using **Google's Gemini 3 Flash** model (`gemini-3-flash-preview`) at the MINIMAL thinking level. The Hebrew source text was obtained from [Sefaria](https://www.sefaria.org). Each *seif* was passed individually to the model with a detailed style guide covering rabbinic titles, standard formulaic phrases, and transliteration conventions.

| Section | Simanim | API calls | Total cost |
|---------|--------:|----------:|----------:|
| Orach Chaim | 697 | 8,119 | |
| Yoreh De'ah | 305 | 7,259 | |
| Even HaEzer | 125 | 3,934 | |
| Choshen Mishpat | 380 | 4,849 | |
| **Total** | **1,507** | **24,161** | **$40.73** |

A post-processing normalization pass was applied after translation, standardizing spelling variants across 13,756 substitutions in 1,247 simanim. Original files are preserved; normalized files are what appear here.

For full details on the method, cost analysis, thinking-level experiment, and translation quality assessment, see the accompanying article (`article_draft.md` in the project repository).

## Previous Translation (GPT-4o)

An earlier translation using OpenAI's GPT-4o, covering *Orach Chaim* and *Yoreh De'ah* only, is preserved in `OLD_translation/`. That translation cost approximately $50 and served as the original proof of concept for this project.

## Note

Special thanks to Rabbi Michael Broyde and Emory University's Law and Religion Center, which funded the model calls for this project. Thank you also to [Sefaria](https://www.sefaria.org) for publishing the original Hebrew text.
