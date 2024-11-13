import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np

class CustomFashionMNIST(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        self.original_dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=True
        )
        
        self.data = self.original_dataset.data.numpy()
        self.targets = self.original_dataset.targets.numpy()
        self.data = self.data.astype(np.float32) / 255.0
        self.transform = transform
        
        self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        image = torch.FloatTensor(image).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FashionNet(nn.Module):
    def __init__(self):
        super(FashionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def save_model(model, optimizer, epoch, accuracy, is_best=False):
   
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    
   
    last_filename = 'fashion_model_last.pt'
    torch.save(checkpoint, last_filename)  
    print(f"Last model saved to {last_filename}")
    
    
    if is_best:
        best_filename = 'fashion_model_best.pt'
        torch.save(checkpoint, best_filename)  
        print(f"Best model saved to {best_filename}")

def load_model(filename='fashion_model_best.pt'):
    """Load model weights"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FashionNet().to(device)
    
    try:
        checkpoint = torch.load(filename, map_location=device)  # Removed weights_only parameter
        model.load_state_dict(checkpoint['model_state_dict'])
        accuracy = checkpoint.get('accuracy', 0.0)
        epoch = checkpoint.get('epoch', 0)
        print(f"Model loaded from {filename}")
        print(f"Accuracy: {accuracy:.2f}%")
        return model, accuracy, epoch
    except FileNotFoundError:
        print(f"No saved model found at {filename}")
        return model, 0.0, 0

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets and loaders
    train_dataset = CustomFashionMNIST(train=True)
    test_dataset = CustomFashionMNIST(train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = FashionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 100 epochs
    num_epochs = 100
    best_accuracy = 0.0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch: {epoch}, Accuracy: {accuracy:.2f}%')
        
        # Save both best and last models
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            print(f'New best accuracy: {accuracy:.2f}%')
        save_model(model, optimizer, epoch, accuracy, is_best=is_best)
        
    print(f"\nTraining completed!")
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")