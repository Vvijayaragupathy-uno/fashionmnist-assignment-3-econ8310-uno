import torch
from torch.utils.data import DataLoader
from fashion_model import CustomFashionMNIST, FashionNet, load_model

def evaluate_model(model_path='fashion_model_best.pt'): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model, saved_accuracy, last_epoch = load_model(model_path)  
        model.eval()
        test_dataset = CustomFashionMNIST(train=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0

        print(f"\nEvaluating model from epoch {last_epoch}")
        print(f"Saved accuracy: {saved_accuracy:.2f}%")

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        current_accuracy = 100 * correct / total
        print(f'Current Evaluation Accuracy: {current_accuracy:.2f}%')

        # Show sample predictions
        print("\nSample Predictions:")
        num_samples = 5 
        for i in range(num_samples):
            sample_image, sample_label = test_dataset[i]
            sample_image = sample_image.unsqueeze(0).to(device)
            prediction = model(sample_image).argmax(1)[0].item()
            print(f"Sample {i + 1}:")
            print(f"Model predicts: {test_dataset.classes[prediction]}")
            print(f"Actual label: {test_dataset.classes[sample_label]}")
            print()
            
    except FileNotFoundError:
        print(f"Error: Could not find model file '{model_path}'")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":    # Fixed the name dunder
    print("Fashion MNIST Model Evaluation ")
    
    print("\nEvaluating Best Model:")
    evaluate_model('fashion_model_best.pt')
    
    print("\nEvaluating Last Model:")
    evaluate_model('fashion_model_last.pt')