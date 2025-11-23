import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_schema_aware_model(
    base_model_path: str = "pre_trained_model/final",
    lora_adapter_path: str = "lora_finetuned_model/final"
):
    """Load schema-aware model with LoRA adapter"""
    
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    return model, tokenizer


def test_knowledge(model, tokenizer):
    """Test model's internal knowledge"""
    
    test_questions = [
        "What measures are in EditedAlleleCallID?",
        "What is the primary key of deployments?",
        "How many dimensions does FieldObservationsV1 have?",
        "What cubes are in the Gene Editing functional area?",
    ]
    
    print("\n" + "="*60)
    print("Testing Schema Knowledge (No Context Provided)")
    print("="*60 + "\n")
    
    for question in test_questions:
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("Answer:")[-1].strip()
        
        print(f"Q: {question}")
        print(f"A: {answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    model, tokenizer = load_schema_aware_model()
    test_knowledge(model, tokenizer)