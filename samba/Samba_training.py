import logging
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, TrainingArguments, Trainer, default_data_collator, AdamW, get_scheduler)
from samba.Samba_modeling import SambaForCausalLM, SambaConfig
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load dataset
dataset = load_dataset("kenhktsui/fineweb-100k_en-med_quality_score_v1")
train_test_split = dataset["train"].train_test_split(test_size=0.1)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
block_size = 1024

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=block_size)

# Tokenize datasets
tokenized_datasets = train_test_split['train'].map(tokenize_function, batched=True, num_proc=16, remove_columns=["text", "quality_score_v1"])

# Group texts
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

# Uncomment these 2 lines if you are running it for the first time
# lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=16)
# lm_datasets.save_to_disk('see-what-wrong-grouped')

# Uncomment this line after first run, since the dataset is already saved to disk
lm_datasets = Dataset.load_from_disk('see-what-wrong-grouped')

# Uncomment these 2 lines if you are running it for the first time
# valid_dataset = train_test_split['test'].map(tokenize_function, batched=True, num_proc=16, remove_columns=["text", "quality_score_v1"])
# eval_dataset = valid_dataset.map(group_texts, batched=True, batch_size=1000, num_proc=16)
# eval_dataset.save_to_disk('see-what-wrong-grouped-eval')

# Uncomment this line after first run, since the dataset is already saved to disk
eval_dataset = Dataset.load_from_disk('see-what-wrong-grouped-eval')

data_collator = default_data_collator

# Define model configuration
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
logging.info(f'Size of the model: {sum(p.numel() for p in model.parameters())} parameters')

training_args = TrainingArguments(
    output_dir="./custom",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=0.005,
    warmup_steps=5,
    eval_accumulation_steps=4,
    per_device_eval_batch_size=2,
    bf16=True,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    optim = 'adamw_bnb_8_bit',
    torch_compile= True,
    # gradient_checkpointing=True # Good for memory efficiency but slows training by 20%
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # optimizers=(optimizer, lr_scheduler)
)

trainer.train()
result = trainer.evaluate()
print('Evaluation results: ', result)

trainer.save_model('samba-first-model')
tokenizer.save_pretrained('samba-first-model')

tokenizer = AutoTokenizer.from_pretrained('samba-first-model')
model = SambaForCausalLM.from_pretrained('samba-first-model')

input_ids = tokenizer("Tell me about ", return_tensors="pt")["input_ids"]
out = model.generate(input_ids, max_new_tokens=1000)
generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)

print(generated_text)
