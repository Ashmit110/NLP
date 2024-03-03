import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load the IMDb dataset from Hugging Face
dataset = load_dataset("imdb")

# Load a pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Prepare DataLoader
train_loader = DataLoader(dataset["train"], batch_size=4, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=4)

# Define model
class SentimentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Initialize model
model = SentimentClassifier(num_classes=2)  # 2 classes: positive and negative

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(3):  # 3 epochs
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
