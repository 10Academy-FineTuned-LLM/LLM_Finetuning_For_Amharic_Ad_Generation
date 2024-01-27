# fine_tune/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import LlamaForSequenceClassification, LlamaTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

# Load data
df = pd.read_csv('')  
train_data, val_data = train_test_split(df, test_size=0.3, random_state=42)

# Define dataset class
class AdDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = int(self.labels[item])
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('model_checkpoint_path')  # Replace with your actual model checkpoint path
model = LlamaForSequenceClassification.from_pretrained('model_checkpoint_path', num_labels=2)  # Assuming 2 classes for ad and non-ad
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create datasets and dataloaders
train_dataset = AdDataset(train_data['text'], train_data['label'], tokenizer, max_len=128)
val_dataset = AdDataset(val_data['text'], val_data['label'], tokenizer, max_len=128)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training settings
epochs = 3
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation loop
    model.eval()
    val_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1} - Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()

    average_val_loss = val_loss / len(val_dataloader)
    accuracy = correct_predictions / len(val_dataset)

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'  Training Loss: {loss:.4f}')
    print(f'  Validation Loss: {average_val_loss:.4f}')
    print(f'  Validation Accuracy: {accuracy * 100:.2f}%')

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model_path')  # Replace with your desired path
tokenizer.save_pretrained('fine_tuned_model_path')
