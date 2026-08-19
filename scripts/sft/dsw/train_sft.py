#!/usr/bin/env python3
import argparse
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainingArguments


class AgentSFTCollator:
    def __init__(self, pad_token_id, pad_to_multiple_of=8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        input_ids, attention_mask, labels = [], [], []
        for x in features:
            n = len(x["input_ids"])
            pad = max_len - n
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def freeze_non_text(model):
    frozen = 0
    for name, p in model.named_parameters():
        trainable = "language_model" in name or name.startswith("lm_head") or ".lm_head" in name
        p.requires_grad = trainable
        if not trainable:
            frozen += p.numel()
    return frozen


def apply_lora(model, args):
    for p in model.parameters():
        p.requires_grad = False
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        bias="none",
    )
    model = get_peft_model(model, config)
    for name, p in model.named_parameters():
        if "lora_" in name and "language_model" not in name:
            p.requires_grad = False
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nas/bihaoran/common_agent/model/Qwen3.5-4B-Base")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--deepspeed", default="")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", default="q_proj,v_proj")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "right"

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if args.lora_r > 0:
        model = apply_lora(model, args)
        frozen = 0
    else:
        frozen = freeze_non_text(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora_r > 0 and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"parameters total={total:,} trainable={trainable:,} frozen_non_text={frozen:,} lora_r={args.lora_r}")

    train_dataset = load_from_disk(args.dataset_path)
    collator = AgentSFTCollator(tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        report_to=["tensorboard"],
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        group_by_length=True,
        length_column_name="length",
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=args.deepspeed if args.deepspeed else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    resume = os.environ.get("RESUME_FROM_CHECKPOINT", "")
    trainer.train(resume_from_checkpoint=resume if resume else None)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
