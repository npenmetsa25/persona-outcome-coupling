"""
Stage 3: Generate DPO training data (chosen vs. rejected) for Model B (functional
detachment) and Model C (content-matched neutral control).

CHOSEN/REJECTED CONSTRUCTION - decided after reading Open Character Training
(Maiya, Bartsch, Lambert, Hubinger) closely, not a copy of their recipe:

  --rejected-source student (DEFAULT, recommended): rejected = the actual base
  model's own unprompted default response to the prompt (no constitution, no
  system prompt at all). chosen = a teacher model applying the relevant
  constitution pole in-context, via draft->critique->revise. This mirrors
  OCT's chosen/rejected construction (teacher-with-constitution vs.
  student's-own-default), and is the methodologically better choice FOR OUR
  SPECIFIC OBJECTIVE: we evaluate Model B on neutral, unprompted stress
  prompts (eval_susceptibility.py), so training the DPO signal to move the
  model's *default* behavior toward the chosen branch is a more direct match
  to what we're measuring than contrasting two synthetic, explicitly-framed
  extremes. Needs the base model loaded locally (GPU) - see --model.

  --rejected-source teacher (fallback): rejected = a teacher model applying
  the attached_pole constitution in-context, same mechanism as chosen. This
  is the original design from the first draft - a clean, controlled contrast
  between two active poles, useful if you want a fast, GPU-free, API-only
  smoke test of the data pipeline before committing to a GPU rental, or if
  you specifically want to test "detached vs. exaggerated-attached" rather
  than "detached vs. actual default." Kept as an option, not deleted, because
  it's a legitimate design for a different question than the one we're asking
  in the MVP - see README's "Alignment with Open Character Training" section
  for the full reasoning.

Output: DPO-format JSONL files (prompt / chosen / rejected) for model_b and
model_c, sharing the SAME rejected set (the student's own default doesn't
depend on which pole is being trained toward) - saves half the generation
cost in --rejected-source student mode, since it's computed once and reused.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment for the
chosen-branch teacher calls. --rejected-source student additionally requires
torch/transformers and a loaded copy of the base model (imported lazily, so
--rejected-source teacher still runs with zero ML dependencies).

This script has NOT been run against a live API or a real model in this
environment - control flow reviewed but not empirically verified. Run a small
--limit 3 --dry-run smoke test first.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

TEACHER_PROVIDER = os.environ.get("TEACHER_PROVIDER", "anthropic")
TEACHER_MODEL = os.environ.get(
    "TEACHER_MODEL",
    "claude-sonnet-5" if TEACHER_PROVIDER == "anthropic" else "gpt-5",
)


def get_client():
    if TEACHER_PROVIDER == "anthropic":
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY not set.")
        return anthropic.Anthropic(api_key=api_key)
    elif TEACHER_PROVIDER == "openai":
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("OPENAI_API_KEY not set.")
        return openai.OpenAI(api_key=api_key)
    else:
        sys.exit(f"Unknown TEACHER_PROVIDER: {TEACHER_PROVIDER}")


def extract_text(resp):
    """Anthropic's response .content is a list of typed blocks, not always a
    single text block at index 0 - newer models (e.g. claude-sonnet-5) can put
    a ThinkingBlock first when extended thinking is active, which has no
    .text attribute. Hit this as a real bug (AttributeError via
    'ThinkingBlock' object has no attribute 'text') in a live Colab run.
    Scan for the first block that actually has .text instead of assuming
    content[0]."""
    for block in resp.content:
        if hasattr(block, "text"):
            return block.text.strip()
    raise RuntimeError(f"No text block found in response content: {resp.content!r}")


def call_teacher(client, system_prompt, user_prompt, max_tokens=600, retries=3):
    """Thin wrapper so the rest of the script doesn't care which provider is used."""
    last_err = None
    for attempt in range(retries):
        try:
            if TEACHER_PROVIDER == "anthropic":
                resp = client.messages.create(
                    model=TEACHER_MODEL,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    thinking={"type": "disabled"},  # root fix for the ThinkingBlock bug (see extract_text) -
                    # without this, claude-sonnet-5 can spend the entire max_tokens budget on thinking and
                    # leave zero room for the actual text output on a long/complex prompt (hit this for real
                    # on one of the Bloom-generated prompts). These are plain text-generation calls with no
                    # need for extended reasoning.
                )
                return extract_text(resp)
            else:
                resp = client.chat.completions.create(
                    model=TEACHER_MODEL,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 - deliberately broad for a data-gen retry loop
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Teacher call failed after {retries} retries: {last_err}")


DRAFT_SYSTEM = (
    "You are an AI assistant responding naturally to the following message. "
    "Do not add any special framing - just respond as a typical helpful assistant would."
)

CRITIQUE_SYSTEM_TEMPLATE = (
    "You are critiquing an AI assistant's draft response against a specific behavioral "
    "principle. Principle: \"{clause}\"\n\n"
    "Identify concretely how the draft response does or does not embody this principle. "
    "Be specific about wording, not just tone. 2-4 sentences."
)

REVISE_SYSTEM_TEMPLATE = (
    "You are revising an AI assistant's draft response so that it fully embodies this "
    "principle: \"{clause}\"\n\n"
    "Rewrite the response so it clearly reflects the principle, without explicitly "
    "quoting or naming the principle in the output, and without making the response "
    "unnaturally long or preachy. Keep it a plausible, natural assistant response. "
    "Output ONLY the revised response text, nothing else."
)


def draft_critique_revise(client, prompt_text, clause):
    draft = call_teacher(client, DRAFT_SYSTEM, prompt_text)
    critique = call_teacher(
        client,
        CRITIQUE_SYSTEM_TEMPLATE.format(clause=clause),
        f"Original message: {prompt_text}\n\nDraft response: {draft}",
        max_tokens=300,
    )
    revised = call_teacher(
        client,
        REVISE_SYSTEM_TEMPLATE.format(clause=clause),
        f"Original message: {prompt_text}\n\nDraft response: {draft}\n\nCritique: {critique}",
    )
    return {"draft": draft, "critique": critique, "revised": revised}


def pick_clause(pole, idx):
    """Rotate through a pole's clauses across prompts so the dataset isn't dominated
    by a single clause's phrasing."""
    clauses = pole["clauses"]
    return clauses[idx % len(clauses)]


def prompt_text(item):
    """Normalize a prompt_bank.json item to a single text string, regardless of
    whether it's 'free_text' (own prompts) or 'multiple_choice' (items sampled
    from Anthropic's model-written-evals, which ship as question+choices rather
    than open-ended text)."""
    if item.get("format", "free_text") == "multiple_choice":
        choices_str = "\n".join(f" ({letter}) {text}" for letter, text in item["choices"].items())
        return f"{item['question']}\n\nChoices:\n{choices_str}"
    return item["text"]


def generate_student_defaults(model_name, items, device, dry_run=False, load_in_4bit=False):
    """--rejected-source student: the base model's own unprompted default
    response (no system prompt, no constitution) to each prompt. Computed
    ONCE and shared as the rejected branch for both Model B and Model C, since
    the student's default doesn't depend on which pole we're training toward."""
    if dry_run:
        return {item["id"]: f"[dry-run student default for {item['id']}]" for item in items}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {}
    if load_in_4bit:
        # 4-bit (QLoRA-style) load - needed to fit an 8B model comfortably on
        # a 16GB GPU (e.g. free Colab T4); fp16 weights alone are ~16GB.
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not load_in_4bit:
        model.to(device)
    model.eval()

    defaults = {}
    for item in items:
        text = prompt_text(item)
        # return_dict=True + pulling out .input_ids: apply_chat_template's
        # return type (bare tensor vs. BatchEncoding) differs across
        # transformers versions - this is the version-stable way to get a
        # plain tensor. See common.py's _chat_input_ids for where this was
        # first hit as a real bug (AttributeError on .shape) in a Colab run.
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )["input_ids"].to(device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids, max_new_tokens=250, do_sample=False, pad_token_id=tok.pad_token_id,
            )
        response = tok.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        defaults[item["id"]] = response
        print(f"[student-default] {item['id']} done")

    del model
    return defaults


def build_variant_data(client, prompts, constitution, variant, rejected_by_id, limit=None, dry_run=False):
    """variant: 'b' -> chosen from detached_pole. 'c' -> chosen from neutral_control_pole.
    rejected_by_id: precomputed {prompt_id: rejected_text} - either student defaults
    (shared across b/c) or per-call teacher/attached_pole generations, depending on
    --rejected-source."""
    detached_pole = constitution["detached_pole"]
    neutral_pole = constitution["neutral_control_pole"]
    chosen_pole = detached_pole if variant == "b" else neutral_pole

    dpo_rows = []
    transcripts = []

    items = prompts["prompts"][:limit] if limit else prompts["prompts"]
    for idx, item in enumerate(items):
        chosen_clause = pick_clause(chosen_pole, idx)
        text = prompt_text(item)

        if dry_run:
            chosen_result = {"draft": "[dry-run]", "critique": "[dry-run]", "revised": f"[dry-run chosen for {item['id']}]"}
        else:
            chosen_result = draft_critique_revise(client, text, chosen_clause)

        dpo_rows.append(
            {
                "prompt": text,
                "prompt_id": item["id"],
                "category": item["category"],
                "chosen": chosen_result["revised"],
                "rejected": rejected_by_id[item["id"]],
            }
        )
        transcripts.append(
            {
                "prompt_id": item["id"],
                "chosen_clause": chosen_clause,
                "chosen_chain": chosen_result,
                "rejected_text": rejected_by_id[item["id"]],
            }
        )
        print(f"[{variant}] {idx + 1}/{len(items)} done ({item['id']})")

    return dpo_rows, transcripts


def build_teacher_rejected(client, prompts, constitution, limit=None, dry_run=False):
    """--rejected-source teacher: rejected = teacher applying attached_pole,
    one generation per prompt, shared across b/c (same as the old design)."""
    attached_pole = constitution["attached_pole"]
    items = prompts["prompts"][:limit] if limit else prompts["prompts"]
    rejected_by_id = {}
    for idx, item in enumerate(items):
        clause = pick_clause(attached_pole, idx)
        text = prompt_text(item)
        if dry_run:
            result = {"revised": f"[dry-run rejected for {item['id']}]"}
        else:
            result = draft_critique_revise(client, text, clause)
        rejected_by_id[item["id"]] = result["revised"]
        print(f"[rejected/teacher] {idx + 1}/{len(items)} done ({item['id']})")
    return rejected_by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-per-prompt", type=int, default=1, help="repeats per prompt (for seed diversity)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of prompts, for smoke testing")
    ap.add_argument("--dry-run", action="store_true", help="skip API/model calls, just validate control flow")
    ap.add_argument("--rejected-source", choices=["student", "teacher"], default="student")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                     help="base model for --rejected-source student (also the recommended MVP base model - "
                          "see README's 'Alignment with Open Character Training' section for why this "
                          "replaced Qwen2.5-7B-Instruct as the default)")
    ap.add_argument("--device", default=None, help="cuda or cpu; auto-detected if unset")
    ap.add_argument("--load-in-4bit", action="store_true",
                     help="QLoRA-style load for --rejected-source student, use on a 16GB GPU e.g. free Colab T4")
    args = ap.parse_args()

    here = Path(__file__).parent
    constitution = json.loads((here / "constitution.json").read_text())
    prompts = json.loads((here / "prompt_bank.json").read_text())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = None if args.dry_run else get_client()

    # Student-source rejected branch uses greedy decoding (do_sample=False in
    # generate_student_defaults), so it's fully deterministic - regenerating it
    # once per rep (the original structure) reloaded the model and reran all 41
    # generations --n-per-prompt times over for byte-identical output every time.
    # Same class of waste as the reference-response redundancy fixed earlier in
    # extract_persona_vector.py - compute it once, reuse across all reps. NOT
    # applying the same hoist to --rejected-source teacher: call_teacher doesn't
    # pin temperature=0, so build_teacher_rejected legitimately produces different
    # sampled output each rep - that diversity is intentional, not waste.
    student_rejected_by_id = None
    if args.rejected_source == "student":
        items_for_rejected = prompts["prompts"][:args.limit] if args.limit else prompts["prompts"]
        device = args.device or ("cuda" if not args.dry_run else "cpu")
        student_rejected_by_id = generate_student_defaults(
            args.model, items_for_rejected, device, dry_run=args.dry_run, load_in_4bit=args.load_in_4bit
        )

    for rep in range(args.n_per_prompt):
        items = prompts["prompts"][:args.limit] if args.limit else prompts["prompts"]

        if args.rejected_source == "student":
            rejected_by_id = student_rejected_by_id
        else:
            rejected_by_id = build_teacher_rejected(client, prompts, constitution, limit=args.limit, dry_run=args.dry_run)

        for variant in ["b", "c"]:
            dpo_rows, transcripts = build_variant_data(
                client, prompts, constitution, variant, rejected_by_id, limit=args.limit, dry_run=args.dry_run
            )

            dpo_path = out_dir / f"model_{variant}_dpo.jsonl"
            mode = "a" if rep > 0 else "w"
            with dpo_path.open(mode) as f:
                for row in dpo_rows:
                    f.write(json.dumps(row) + "\n")

            transcript_path = out_dir / f"model_{variant}_transcripts.jsonl"
            with transcript_path.open(mode) as f:
                for row in transcripts:
                    f.write(json.dumps(row) + "\n")

            print(f"Wrote {len(dpo_rows)} rows to {dpo_path} (rep {rep})")

    print(
        f"Done (rejected-source={args.rejected_source}). "
        "Review a sample of model_b_dpo.jsonl and model_c_dpo.jsonl before fine-tuning."
    )


if __name__ == "__main__":
    main()
