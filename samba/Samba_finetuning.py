import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

# Model
base_model = "state-spaces/mamba-2.8b-hf"
new_model = "mamba-2-finetune"

# Set torch dtype and attention implementation
torch_dtype = torch.float16

dataset_name = "mlabonne/mini-platypus"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.train_test_split(test_size=0.01)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.unk_token

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['instruction'], padding="max_length", truncation=True, max_length=512)

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch_dtype,
)

# Training arguments
training_arguments = TrainingArguments(
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    num_train_epochs=1,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=100,  
    logging_steps=1,
    optim="paged_adamw_8bit",
    warmup_steps=10,
    output_dir="./results",
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=50,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    dataset_text_field="input_ids",  
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

# Generate text with the fine-tuned model
prompt = "What is a large language model?"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"

# Reload the fine-tuned model for inference
pipe = pipeline("text-generation", model=new_model, tokenizer=tokenizer, max_length=128)
result = pipe(instruction)
print(result[0]["generated_text"][len(instruction):])

