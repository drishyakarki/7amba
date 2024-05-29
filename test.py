from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from Samba_modeling import SambaForCausalLM, SambaConfig

# Load dataset
dataset = load_dataset("BEE-spoke-data/fineweb-1M_longish")
train_testSplit = dataset["train"].train_test_split(test_size=0.1)

# Tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
block_size = 2048

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=block_size)

tokenized_datasets = train_testSplit['train'].map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = Dataset.load_from_disk("7amba-first-lm")

valid_dataset = train_testSplit['test'].map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])
eval_dataset = valid_dataset.map(group_texts, batched=True, batch_size=1000, num_proc=16)

data_collator = default_data_collator

# Model configuration and initialization
config = SambaConfig(
    vocab_size=50280,
    hidden_size=4096,
    num_hidden_layers=64,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_attention_heads=8
)
model = SambaForCausalLM(config)

print(f'size of the model is {sum(p.numel() for p in model.parameters())}')

# Training setup
training_args = TrainingArguments(
    output_dir="./custom",
    num_train_epochs=10,
    per_device_train_batch_size=3,
    learning_rate=0.005,
    warmup_steps=5,
    eval_accumulation_steps=5,
    per_device_eval_batch_size=3,
    bf16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=lm_datasets,
    data_collator=data_collator,
    eval_dataset=eval_dataset,
)

# Train and evaluate the model
trainer.train()
result = trainer.evaluate()

# Save the model
trainer.save_model('samba-first-model')

# Load the trained model for inference
tokenizer = AutoTokenizer.from_pretrained('samba-first-model')
model = SambaForCausalLM.from_pretrained('samba-first-model')

# Generate text
input_ids = tokenizer("Tell me about ", return_tensors="pt")["input_ids"]
out = model.generate(input_ids, max_new_tokens=1000)
generated_text = tokenizer.batch_decode(out)

print(generated_text)
