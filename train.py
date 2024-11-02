import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load the model and tokenizer
model_name = "zake7749/gemma-2-2b-it-chinese-kyara-dpo"
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load dataset from JSON file
with open("train.json") as f:
    data = json.load(f)

# Split the data into training and validation sets (80/20 split)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["instruction"] + examples["output"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
)


# Define Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []

    def compute_loss(self, model, inputs, return_outputs=False):
        # Calculate loss and track it
        loss = super().compute_loss(model, inputs, return_outputs)
        self.losses.append(loss.item())
        return loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train and validate the model
trainer.train()

# Plot training loss
plt.plot(trainer.losses)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Saving model predictions for the validation set
predictions = []
for example in tqdm(val_data, desc="Predicting"):
    inputs = tokenizer(example["instruction"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append({"id": example["id"], "output": prediction})

# Save predictions to prediction.json
with open("prediction.json", "w") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=4)

print("Training and prediction completed.")
