"""
gender_trainer.py - Gender Detection Model Training

Fine-tunes a pre-trained model on custom gender/age labeled dataset.

Usage:
    python gender_trainer.py --dataset data/gender_dataset --epochs 10 --output models/custom_gender_model.pt
    or use as module: from gender_trainer import GenderModelTrainer
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import cv2
from pathlib import Path


class GenderDataset(Dataset):
    """PyTorch Dataset for gender classification."""

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: Directory containing 'male' and 'female' subdirectories
            transform: Image transforms
        """
        self.images = []
        self.labels = []
        self.transform = transform

        # Load male images (label 0)
        male_dir = os.path.join(image_dir, 'male')
        if os.path.exists(male_dir):
            for img_file in os.listdir(male_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(male_dir, img_file))
                    self.labels.append(0)  # Male

        # Load female images (label 1)
        female_dir = os.path.join(image_dir, 'female')
        if os.path.exists(female_dir):
            for img_file in os.listdir(female_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(female_dir, img_file))
                    self.labels.append(1)  # Female

        print(f"[Dataset] Loaded {len(self.images)} images")
        print(f"  Male: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Female: {sum(1 for l in self.labels if l == 1)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a dummy image if load fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


class GenderModelTrainer:
    """Trains fine-tuned gender detection models."""

    def __init__(self, output_dir="models"):
        """Initialize trainer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Using device: {self.device}")

    def create_model(self, num_classes=2):
        """Create ResNet50 model for gender classification."""
        model = resnet50(pretrained=True)
        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def get_transforms(self):
        """Define data augmentation transforms."""
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return train_transform, val_transform

    def train_model(self, dataset_dir, epochs=10, batch_size=16, learning_rate=0.001):
        """
        Train the model on the dataset.

        Args:
            dataset_dir: Directory containing 'train' and 'val' subdirectories
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer

        Returns:
            trained model and metrics
        """
        print("\n" + "="*60)
        print("Training Gender Detection Model")
        print("="*60)

        # Load datasets
        train_transform, val_transform = self.get_transforms()

        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val')

        train_dataset = GenderDataset(train_dir, transform=train_transform)
        val_dataset = GenderDataset(val_dir, transform=val_transform)

        if len(train_dataset) == 0:
            print("[Trainer] ⚠ No training data found. Check directory structure.")
            return None, None

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Create model
        model = self.create_model(num_classes=2)
        model = model.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

        best_val_acc = 0.0
        best_model_path = os.path.join(self.output_dir, 'best_gender_model.pt')

        # Training loop
        for epoch in range(epochs):
            # Train phase
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(train_labels, train_preds)
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_acc = accuracy_score(val_labels, val_preds)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Best model saved (val_acc: {val_acc:.4f})")

            scheduler.step(val_acc)

        print("\n" + "="*60)
        print("Training Complete")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Model saved to: {best_model_path}")
        print("="*60 + "\n")

        # Load best model
        model.load_state_dict(torch.load(best_model_path))

        # Final metrics
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds)

        metrics = {
            'best_val_accuracy': best_val_acc,
            'confusion_matrix': cm.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
        }

        print("Final Metrics (Validation Set):")
        print(f"  Accuracy: {best_val_acc:.4f}")
        print(f"  Male Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}")
        print(f"  Female Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}")
        print(f"\nConfusion Matrix (rows=true, cols=pred):")
        print(cm)

        return model, metrics

    def save_model_for_inference(self, model, output_path):
        """Save model in a format compatible with gender_detector.py"""
        torch.save(model.state_dict(), output_path)
        print(f"[Trainer] Model saved for inference: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gender Detection Model Trainer")
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to dataset directory (should have train/ and val/ subdirs)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--output", type=str, default="models/custom_gender_model.pt",
        help="Output model path"
    )

    args = parser.parse_args()

    trainer = GenderModelTrainer(output_dir=os.path.dirname(args.output))
    model, metrics = trainer.train_model(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    if model is not None:
        trainer.save_model_for_inference(model, args.output)
