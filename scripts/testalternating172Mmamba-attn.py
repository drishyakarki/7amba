'''
This is the script to train the model with the architecture:
introduce attention layers after every 7 mamba blocks, with 162M parameters
'''


import logging
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from Samba_modeling import SambaForCausalLM, SambaConfig

# Set up logging
logging.basicConfig(level=logging.INFO)

tokenized_dataset = Dataset.load_from_disk('fineweb-tokenized-10B')
lm_datasets = Dataset.load_from_disk('fineweb-tokenized-grouped-10B')

valid_dataset = Dataset.load_from_disk('fineweb-tokenized-eval-10B')
eval_dataset = Dataset.load_from_disk('fineweb-tokenized-eval-grouped-10B')

data_collator = default_data_collator

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
# Define model configuration
config = SambaConfig(
    vocab_size=50280,
    hidden_size=1024,
    num_hidden_layers=16,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_attention_heads=16,
    attention_layer=2,
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
    # torch_compile= True, # I think you said it caused problem earlier
    # gradient_checkpointing=True # Good for memory efficiency but slows training by 20%
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
result = trainer.evaluate()
print('Evaluation results: ', result)

trainer.save_model('samba-172M-model-alternating')
tokenizer.save_pretrained('samba-172M-model-alternating')

tokenizer = AutoTokenizer.from_pretrained('samba-172M-model-alternating')
model = SambaForCausalLM.from_pretrained('samba-172M-model-alternating')

input_ids = tokenizer("One of the most ", return_tensors="pt")["input_ids"]
out = model.generate(input_ids, max_new_tokens=1000)
generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)

print(generated_text)
