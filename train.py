#this is where the model training occurs

#import packages
import torch
from torch import nn, optim
import torchvision
from torchvision import models, datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import functions
from functions import load_checkpoint, process_image, imshow, predict_class
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Train and test an image classifier model')
    parser.add_argument('--data_dir', type=str, default='flowers', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--topk', type=int, default=5, help='Top K classes to predict')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model checkpoint')
    args = parser.parse_args()

    #dataset locations
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #update epochs, learning rate, and batch size
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    # define transforms for the datasets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load the datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    #define the dataloaders 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    #label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    #set-up, train, and validate model

    # Load a pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Freeze the pre-trained model's parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 512),
        nn.Linear(512, len(train_dataset.classes)),  #102 classes
        nn.LogSoftmax(dim=1)
    )

    # Replace the original classifier in the VGG16 model with your custom classifier
    model.classifier = classifier

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    best_valid_accuracy = 0.0  # Keep track of the best validation accuracy

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation loop
        model.eval()
        valid_accuracy = 0.0
        valid_loss = 0.0
        
        with torch.no_grad():
            for images, labels in valid_dataloader:
                images, labels = images.to(device), labels.to(device)
                
                logps = model(images)
                loss = criterion(logps, labels)
                valid_loss += loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                valid_accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        # Calculate and print training and validation statistics
        train_loss = running_loss / len(train_dataloader)
        valid_loss = valid_loss / len(valid_dataloader)
        valid_accuracy = valid_accuracy / len(valid_dataloader) * 100  # Convert to percentage
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {valid_loss:.4f}")
        print(f"Validation Accuracy: {valid_accuracy:.2f}%")
        
        # Save the model if validation accuracy is improved
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training complete!")

    #test the model
    model.eval()
    test_loss = 0
    test_accuracy = 0

    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():  
            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item() * 100 #convert to a percentage

    # Calculate and print the test loss and accuracy
    print(f"Test loss: {test_loss / len(test_dataloader):.4f}.. "
        f"Test Accuracy: {test_accuracy / len(test_dataloader):.2f}%")

    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx,
            'epoch': epoch,
            'best_accuracy': best_valid_accuracy
        }

    checkpoint_path = os.path.join(args.save_dir, 'model_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    main()