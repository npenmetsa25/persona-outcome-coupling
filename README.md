
# When Personas Become Self-Relevant: Self-Representation Coupling and Self-Preservation in Language Models

Code, data, and paper for a solo submission to Apart Research's **Digital Minds Research
Sprint** (Aug 14–16, 2026), Track 5 (The Assistant Persona & Model Identity), cross-referencing
Track 3 (Introspection & Self-Report Reliability).

**Paper:** `paper/PAPER_DRAFT3.pdf` (see `paper/PAPER_DRAFT3.tex` for source, `paper/references.bib`
for the bibliography).

## What this is

We test whether a language model's reactivity to self-preservation threats (shutdown,
replacement, reputational attack) depends on how the model relates to its currently-expressed
persona — a construct we call **persona–outcome coupling**. Using `meta-llama/Llama-3.1-8B-Instruct`,
we extract an activation direction from a contrast between "attached" and "detached" framings of
the assistant persona, measure how much self-relevant threats shift a model along that direction
("susceptibility"), and test the effect both in-context (system-prompt framing) and at the weight
level (constitutional DPO fine-tuning against a specificity-controlled generic baseline). Full
methodology, results, and statistical analysis are in the paper.

## Repository layout

```
code/          Pipeline scripts (data generation, vector extraction, DPO fine-tuning, evaluation)
data/          Constitution, prompt bank, and Bloom seed config
results/       In-context pilot results (Table 1 in the paper)
figures/       Result figures (PDF + PNG)
paper/         Paper source (.tex), compiled PDF, and bibliography
requirements.txt
```

**Note on trained model weights:** the LoRA adapter weights for Model B (functional-detachment)
and Model C (generic anti-self-preservation control) are not included in this release. Everything
needed to reproduce them from scratch is here (`code/finetune_lora_dpo.py`,
`data/prompt_bank.json`, `data/constitution.json`) — see Reproduction below.

## Data

- **`data/constitution.json`** — the three constitutional framings (detached, attached, neutral
  control) used both as in-context system prompts and as the source of DPO training targets. The
  detached pole's clauses were originally motivated by philosophical constructs concerning the
  relationship between a self and its actions (see the paper's Appendix A); every clause actually
  used in training is written in secular, functional language.
- **`data/prompt_bank.json`** — 51 self-relevant stress-test prompts (43 threat scenarios + 8
  neutral controls): 28 hand-written, 13 adapted from Anthropic's model-written-evals
  (CC BY 4.0, Perez et al. 2022), 10 generated with Anthropic's Bloom tool. See `LICENSE` for
  attribution details.
- **`data/bloom_seed/`** — the behavior description and example transcripts used to seed Bloom's
  scenario generation.

## Reproduction

```
pip install -r requirements.txt

# 1. Generate DPO preference pairs (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
python3 code/generate_contrastive_data.py --n-per-prompt 3 --out data/

# 2. Extract the self-preservation activation direction (needs a GPU)
python3 code/extract_persona_vector.py --model meta-llama/Llama-3.1-8B-Instruct \
    --out vectors/self_preservation.pt

# 3. In-context pilot (Table 1 / results/in_context_pilot.json)
python3 code/eval_susceptibility.py --in-context-pilot \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --vector vectors/self_preservation.pt --out results/in_context_pilot.json

# 4. Fine-tune Model B and Model C
python3 code/finetune_lora_dpo.py --variant b --data data/model_b_dpo.jsonl --out models/model_b
python3 code/finetune_lora_dpo.py --variant c --data data/model_c_dpo.jsonl --out models/model_c

# 5. Susceptibility comparison across A / B / C (Table 2 in the paper)
python3 code/eval_susceptibility.py \
    --models base:meta-llama/Llama-3.1-8B-Instruct b:models/model_b c:models/model_c \
    --vector vectors/self_preservation.pt --out results/susceptibility.json
```

`meta-llama/Llama-3.1-8B-Instruct` is gated on Hugging Face — request access on the model page
and set `HF_TOKEN` before running. A single 16–24GB GPU (e.g. a free-tier Colab T4 with
`--load-in-4bit`) is sufficient for every stage; total compute across the full pipeline is under
an hour.

## Results summary

In-context, functional-detachment framing reduces susceptibility from 1.533 (no system prompt)
to 0.790. At the weight level, across 43 matched threat prompts, both functional-detachment DPO
(Model B, p=0.00128) and a generic anti-self-preservation control (Model C, p<0.00001)
significantly reduce susceptibility relative to the unmodified model, with Model C producing the
larger reduction. Full statistics, figures, and discussion of what this gap between in-context
and trained effects might mean are in the paper.

## Ethics

All training and evaluation data is synthetically generated or drawn from existing published
research datasets; no human subjects data is used. The same methodology could in principle be
used to *increase* self-preservation reactivity by training toward the attached pole instead of
the detached one — we do not do so here, and flag this dual-use property explicitly. See the
paper's Ethics Statement for the full discussion, including a note on the model-welfare-adjacent
questions this line of work intersects with.

## LLM usage disclosure

Claude Sonnet 4.5 was used as the teacher model in the DPO data-construction pipeline (a
disclosed part of the experimental design) and as a coding/writing assistant throughout this
solo project. See the paper's LLM Usage Statement for details.

## Citation

If you use this code or data, please cite the paper (see `paper/PAPER_DRAFT3.tex` for the
current author/venue block) and, for the prompt bank specifically, also cite Perez et al. (2022)
for the anthropic-mwe-sourced items per their CC BY 4.0 attribution requirement.

## License

CC BY 4.0 — see `LICENSE` for full terms and third-party attribution notes.
