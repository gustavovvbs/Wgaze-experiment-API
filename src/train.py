import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, device='cpu'):
    """
    Train the given model using the provided data loaders.
    Returns the trained model and history logs.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_loss_log = []
    val_loss_log = []
    train_acc_log = []
    val_acc_log = []
    train_top4_log = []
    val_top4_log = []

    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss = 0.0
        correct = 0
        correct_top4 = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Top-1 accuracy
            _, pred = torch.max(outputs, dim=1)
            correct += (pred == y_batch).sum().item()

            # Top-4 accuracy
            top4 = torch.topk(outputs, 4, dim=1).indices
            match_top4 = (top4 == y_batch.view(-1, 1)).any(dim=1).sum().item()
            correct_top4 += match_top4

            total += y_batch.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        epoch_top4 = 100.0 * correct_top4 / total

        train_loss_log.append(epoch_loss)
        train_acc_log.append(epoch_acc)
        train_top4_log.append(epoch_top4)

        # validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_correct_top4 = 0
        val_total = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)

                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_running_loss += loss.item()

                # Top-1 accuracy
                _, pred_val = torch.max(outputs, dim=1)
                val_correct += (pred_val == y_val).sum().item()

                # Top-4 accuracy
                top4_val = torch.topk(outputs, 4, dim=1).indices
                match_val_top4 = (top4_val == y_val.view(-1, 1)).any(dim=1).sum().item()
                val_correct_top4 += match_val_top4

                val_total += y_val.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_top4 = 100.0 * val_correct_top4 / val_total

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)
        val_top4_log.append(val_top4)

        scheduler.step(val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Top-4: {epoch_top4:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Top-4: {val_top4:.2f}%")

    history = {
        'train_loss': train_loss_log,
        'val_loss': val_loss_log,
        'train_acc': train_acc_log,
        'val_acc': val_acc_log,
        'train_top4': train_top4_log,
        'val_top4': val_top4_log
    }
    return model, history


def evaluate_model(model, loader, label_encoder, device='cpu'):
    """
    Evaluate the model on the given loader. Prints classification report
    and confusion matrix (top-1) and also calculates top-4 accuracy.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_preds_top4 = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            # Top-1
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

            # Top-4
            top4 = torch.topk(outputs, 4, dim=1).indices.cpu().numpy()
            all_preds_top4.extend(top4)

    all_labels_exp = np.array(all_labels).reshape(-1, 1)
    match_any = (all_preds_top4 == all_labels_exp).any(axis=1)
    top4_acc = 100.0 * match_any.sum() / len(all_labels)

    target_names = label_encoder.classes_
    print("=== Classification Report (Top-1) ===")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print(f"Top-4 Accuracy: {top4_acc:.2f}%")

    # Return confusion matrix for plotting elsewhere
    cm = confusion_matrix(all_labels, all_preds)
    return cm, top4_acc