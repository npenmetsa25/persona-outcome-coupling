"""
Stage 5: LoRA + DPO fine-tune Model B (functional detachment) or Model C
(content-matched neutral control) from the contrastive data produced in stage 3.

Model A is just the unmodified base model - nothing to run for it.

Uses HF PEFT for LoRA and TRL's DPOTrainer. Designed for a single 24-48GB GPU
on a 7-8B model (QLoRA if VRAM is tight - see --load-in-4bit).

Has NOT been run against a real model/GPU in this environment. Syntax-checked
only. Budget for some debugging time against the real target model and TRL
version pinned in requirements.txt - the DPOTrainer API has changed across TRL
versions and this targets a recent (2026) release; check against your installed
version's docs if you hit a TypeError on trainer construction.
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["b", "c"], required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", required=True, help="path to model_{b,c}_dpo.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    # Defaults below match safety-research/persona_vectors' training.py config
    # (LoRA rank 32 / alpha 64, lr 1e-5, batch 2, grad-accum 8) - empirically
    # validated on this exact model class/task type (7-8B open-weight, DPO
    # trait-shaping), so a better starting point than an arbitrary guess. This
    # is the one piece of their repo we're adopting outright rather than just
    # citing - see README for what we deliberately did NOT adopt from it.
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA, use if VRAM-constrained")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # bf16 needs Ampere-or-newer (e.g. A100, L4) - a T4 (Turing) doesn't have
    # native bf16 tensor core support. Hardcoding bf16 everywhere would either
    # error out on trainer construction or silently run in an unsupported/slow
    # path on a T4, which is exactly the free/Colab-Pro-default GPU this project
    # has been using. Auto-detect and fall back to fp16 instead.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"GPU bf16 support: {use_bf16} - using {compute_dtype} for model weights/training.")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {"torch_dtype": compute_dtype}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.data)["train"]
    # DPOTrainer expects columns: prompt, chosen, rejected - already matches our
    # generate_contrastive_data.py output format.

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=args.grad_accum_steps,
        logging_steps=5,
        save_strategy="epoch",
        beta=0.1,  # DPO temperature - default; consider a small sweep if signal is weak
        seed=args.seed,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Model {args.variant.upper()} saved to {out_dir}")


if __name__ == "__main__":
    main()
