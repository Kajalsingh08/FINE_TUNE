import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from datasets import Dataset

def create_dataset(dataset_path : str):
        """
        Loads the instruction dataset from a .jsonl file and formats it into a
        conversational prompt structure that the SFTTrainer can use.
        """
        print(f"Loading and formatting dataset from: {dataset_path}")

        def format_instruction(sample):
            messages = sample["messages"]
            formatted = ""

            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    formatted += f"<|system|>\n{content}<|end|>\n"
                elif role == "user":
                    formatted += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    formatted += f"<|assistant|>\n{content}<|end|>\n"

            return formatted


        dataset = Dataset.from_json(dataset_path)
        # The SFTTrainer expects a 'text' column containing the fully formatted prompt.
        dataset = dataset.map(lambda sample: {"text": format_instruction(sample)})

        print(f"Dataset loaded and formatted. Number of examples: {len(dataset)}")
        print("\n--- Example of Formatted Text ---")
        print(dataset[0]['text'])
        print("---------------------------------\n")
        return dataset # Added this line to return the dataset

def train_lora(
    pretrained_model_path: str = "pre_trained_model/final",
    instruction_data_path: str = "training_data/instructions_v1.json",
    output_dir: str = "lora_finetuned_model"
):
    """Train LoRA adapter on pre-trained model"""

    print("Loading pre-trained model...")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load instruction dataset
    dataset = create_dataset(instruction_data_path)
    def tokenize_function(sample):
      tokenized = tokenizer(
          sample["text"],
          truncation=True,
          padding=False
      )
      # SFTTrainer usually needs labels same as input_ids
      tokenized["labels"] = tokenized["input_ids"].copy()
      return tokenized

    
    dataset = dataset.map(tokenize_function, batched=True)
    print(dataset)
    cols_to_remove = [c for c in ["messages", "text"] if c in dataset.column_names]
    dataset = dataset.remove_columns(cols_to_remove)
    print(dataset)
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none"
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    # Save
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"✓ LoRA adapter saved to: {output_dir}/final")


if __name__ == "__main__":
    train_lora()