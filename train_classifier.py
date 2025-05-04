import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from models import Classifier
from config import Config

def train_classifier():
    # Load configuration
    cfg = Config()

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_data_loaders(cfg)

    # Initialize classifier
    csf = Classifier(cfg.nc, cfg.ncf).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(csf.parameters(), lr=0.001, weight_decay=1e-5)

    # Training parameters
    num_epochs = 20
    best_acc = 0.0
    best_model_path = 'best_classifier.pth'

    # Training loop
    for epoch in range(num_epochs):
        csf.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = csf(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        csf.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = csf(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Test Accuracy: {accuracy:.2f}%")

        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(csf.state_dict(), best_model_path)
            print(f"Saved best model with accuracy {best_acc:.2f}% to {best_model_path}")

if __name__ == "__main__":
    train_classifier()