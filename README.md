# Rezolv Intelligence — PM Experimentation Workspace

A workspace for tweaking prompts, inputs, and outputs until the briefing
reads the way you want.

## What this is

A two-stage LLM pipeline that turns one customer account's data into a
field-agent briefing:

```
Features dict (~2KB precomputed)
       ↓
Stage 1 — Analyser (LLM)
   Picks states for 17 indicators, scores severity 0-100,
   returns clean structured JSON. No prose.
       ↓
Stage 2 — Writer (LLM)
   Reads Stage 1's output, writes the briefing the agent sees.
   This is where tone happens.
       ↓
Field agent briefing
```

Both prompts are fully editable in the app. You can re-run them independently.

## Setup

```bash
pip install pandas streamlit anthropic numpy
python scripts/generate_realistic.py    # generates 10 test accounts
streamlit run app.py
```

When the app opens:
- Set your Anthropic API key in the top bar (or set `ANTHROPIC_API_KEY` env var)
- Pick an account
- Go to the **Pipeline** tab and run Stage 1, then Stage 2
- Switch to **Field App** to see the rendered briefing

## The four tabs

**📱 Field App** — Mobile preview of the final briefing. This is what the
agent sees on their phone. Quick checks panel on the right flags issues
(headline too long, forbidden words, etc.).

**⚡ Pipeline** — Two run buttons. Stage 1 (Analyser) and Stage 2 (Writer)
can be run independently. Token counts and latency shown per stage. Stage 2
is disabled until Stage 1 has run.

**✏️ Prompts** — Edit both prompts inline. Changes auto-mark the relevant
stage as stale. Reset-to-default button per prompt.

**📊 Input Data** — View and edit the features dict for the selected account.
Tree view (per-section editable) or raw JSON. Edit values to test edge cases
(what if PTP honour rate were 49% instead of 50%?). Edits mark both stages
stale.

## The iteration loop

This is the main pattern:

1. Pick an account in the top bar.
2. Run Stage 1 once → see the analyser's structured output.
3. Run Stage 2 → see the briefing render in the Field App tab.
4. Tab into **Prompts**, tweak the writer prompt's tone instructions.
5. Tab back to **Pipeline**, re-run *only* Stage 2 (Stage 1 output is
   reused — saves tokens).
6. Check the Field App tab again. Repeat steps 4-6 until the briefing
   reads right.

When you're happy with the writer, switch accounts and verify the prompt
generalises. When it doesn't, you know the prompt is overfit to one
profile — adjust.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — the experimentation interface |
| `features.py` | Pandas feature extractor — turns raw CSVs into the features dict |
| `prompts_analyser.py` | Stage 1 system prompt (default) |
| `prompts_writer.py` | Stage 2 system prompt (default) |
| `rules_evaluator.py` | Pure-Python rules implementation (reference) |
| `sample_data/` | 10 realistic test accounts |
| `scripts/generate_realistic.py` | Regenerates the sample data |

## Tips

- **For tone iteration, only re-run Stage 2.** Stage 1's output doesn't depend
  on the writer prompt, so re-running it wastes tokens.
- **For data edge cases, edit the features dict in Input Data tab, not the
  CSVs.** Faster to experiment with single values.
- **Switch models in the top bar.** Haiku is much cheaper for Stage 2
  iteration. Use Opus when you want the best quality result.
- **Watch the token counts.** If a prompt change suddenly doubles output
  tokens, the model is probably padding — usually a sign the instructions
  got too vague.
