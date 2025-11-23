import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def prepare_dataset(corpus_path : str , tokenizer , max_length: int = 1024 , ):
    """
    Loads the text corpus from the file and prepares it for training by tokenizing it.
    """
    print(f"\nPreparing dataset from: {corpus_path}")

    # `load_dataset` can read various formats. Here, we're just loading a plain text file.
    dataset = load_dataset('text', data_files={'train': corpus_path}, split='train')
    print(f"  - Loaded {len(dataset)} raw text examples (lines from the file).")

    # Filter out empty lines to prevent runtime errors with empty tensors
    original_rows = len(dataset)
    dataset = dataset.filter(lambda example: example['text'] is not None and len(example['text'].strip()) > 0)
    filtered_rows = len(dataset)
    if original_rows > filtered_rows:
        print(f"  - Filtered out {original_rows - filtered_rows} empty or whitespace-only lines.")

    # This is the function that will be applied to every example in our dataset.
    def tokenize_function(examples):
        # The tokenizer converts the text to token IDs.
        return tokenizer(
            examples['text'],
            truncation=True,      # Truncate examples longer than `max_length`.
            max_length=max_length,
            padding=False,        # We don't pad here; the data collator will handle it later.
            return_tensors=None   # Return Python lists instead of PyTorch tensors.
        )

    print("  - Tokenizing dataset...")
    # The `.map()` function is highly efficient. It applies `tokenize_function` to the entire dataset,
    # using multiple processes and caching the results.
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,              # Process multiple examples at once for speed.
        remove_columns=['text'],   # We no longer need the original text column after tokenization.
        desc="Running tokenizer on dataset"
    )
    
    print(f"✓ Dataset prepared with {len(tokenized_dataset)} tokenized examples.")
    return tokenized_dataset


def main():

    # -----------------------------
    # Config
    # -----------------------------
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    corpus_path = "training_data/graph_corpus_v1.txt"   # <-- your corpus
    output_dir = "pre_trained_model"
    max_length = 2048
    seed = 42

    # HF token (optional if model is gated/private)
    hf_token = os.environ.get("HF_TOKEN", None)

    # -----------------------------
    # Load tokenizer & model
    # -----------------------------
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_auth_token=hf_token         # FIXED
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=hf_token         # FIXED
    )

    # -----------------------------
    # Load and tokenize dataset
    # -----------------------------
    print(f"Loading corpus: {corpus_path}")
    tokenized_dataset = prepare_dataset(corpus_path , tokenizer,max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_steps=800,
        logging_steps=20,
        save_steps=2000,
        save_total_limit=5,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=seed
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("\n============================================")
    print("Starting continued pre-training on graph corpus")
    print("============================================\n")
    trainer.train()
   

    # -----------------------------
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    print(f"Saving model to: {final_dir}")

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("✓ Training complete")

if __name__ == "__main__":
    main()