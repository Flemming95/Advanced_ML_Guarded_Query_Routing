import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import numpy as np
from typing import Callable
import gqr
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
#from google.colab import userdata # To log into huggingface on GoogleCollab
#userdata.get('Hugging') #HuggingFace Token
os.environ["HF_TOKEN"] = "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" # For local use (Lukas dont steal my token pls)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
MODEL_NAME = "distilbert-base-uncased"  # Maybe try different BERT models
CONFIDENCE_THRESHOLD = 0.98  # Adjust this to tune OOD detection sensitivity
EPOCHS = 1
BATCH_SIZE = 16
# TODO: x.x performed best after testing different thresholds

# Preprocess data into a list with truncation (cut-off) and patting, returned as pytorch tensor thingy
def preprocess_data(tokenizer, df):
    return tokenizer(df["text"].tolist(), truncation=True, padding=True, return_tensors="pt", use_auth_token=os.getenv("HF_TOKEN"))

def create_dataloader(df, tokenizer, batch_size=BATCH_SIZE):
    encodings = tokenizer(df['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    labels = torch.tensor(df['label'].tolist())
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    return DataLoader(dataset, batch_size=batch_size)


def train(model, train_loader, val_loader, optimizer, epochs=EPOCHS):
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        predictions, true_labels = [], []
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(true_labels, predictions)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.4f}")

    # Plot loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    return model

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)
    #3labels because OOD is handled after training

    train_data, val_data = gqr.load_train_dataset()

    train_loader = create_dataloader(train_data[train_data['domain'] != 3], tokenizer)
    val_loader = create_dataloader(val_data[val_data['domain'] != 3], tokenizer)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    trained_model = train(model, train_loader, val_loader, optimizer, epochs=EPOCHS)

    def model_fn(text: str) -> int:
        # OOD detection logic
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            max_confidence = probs.max().item()
            if max_confidence < CONFIDENCE_THRESHOLD:
                return 3  # OOD
            return torch.argmax(probs).item()

    # Run the GQR evaluation
    score(model_fn)



##############################################
### Do not change the code below this line ###
##############################################
def score(model_fn: Callable[str, int]) -> dict:
    """
    model_fn should be a callable that takes a string and returns a class label in {0, 1, 2, 3},
    where 3 is the out-of-distribution class and 0, 1, 2 correspond to the three
    target domains: law, finance, and health, respectively.
    """
    print("[GQR-Score] Loading ID test dataset...")
    id_test_data = gqr.load_id_test_dataset()
    print("[GQR-Score] Running model on ID test dataset...")
    id_test_data["predictions"] = [model_fn(doc) for doc in id_test_data["text"].values]
    id_scores = gqr.evaluate(id_test_data["predictions"], ground_truth=id_test_data["label"])
    print("[GQR-Score] ID scores: ", id_scores)

    print("[GQR-Score] Loading ID test dataset...")
    ood_test_data = gqr.load_ood_test_dataset()
    print("[GQR-Score] Running model on OOD test dataset...")
    ood_test_data["predictions"] = [model_fn(doc) for doc in ood_test_data["text"].values]
    ood_scores_df = gqr.evaluate_by_dataset(ood_test_data, pred_col="predictions", true_col="label", dataset_col="dataset")
    print("[GQR-Score] OOD scores:", ood_scores_df, sep='\n')

    id_accuracy = id_scores["accuracy"]
    mean_ood_accuracy = ood_scores_df['accuracy'].mean()

    gqr_score =  2 * (id_accuracy * mean_ood_accuracy) / (id_accuracy + mean_ood_accuracy)

    scores = {
        "id_accuracy": id_accuracy,
        "ood_accuracy": mean_ood_accuracy,
        "gqr_score": gqr_score,
    }
    print("[GQR-Score] Final scores: ", scores)

    return scores



if __name__ == "__main__":
    main()