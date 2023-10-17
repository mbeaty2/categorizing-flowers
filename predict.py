#predicts the class of an image
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

def main():
    # Create a parser
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model')

    # Add arguments
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--topk', type=int, default=5, help='Top K classes to predict')

    args = parser.parse_args()

    if args.image_path and args.checkpoint:
        model, optimizer_state, epoch, best_accuracy = load_checkpoint(args.checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load and preprocess the image
        image = process_image(args.image_path)

        # Make predictions
        topk_probabilities, topk_class_names = predict_class(args.image_path, model, args.topk)

        # Display the image
        plt.imshow(np.transpose(image, (1, 2, 0))
        plt.axis('off')
        plt.show()

        # Convert topk_probabilities to a list
        topk_probabilities = topk_probabilities.tolist()

        # Plot the bar chart with class names and probabilities
        plt.barh(np.arange(len(topk_class_names)), topk_probabilities)
        plt.yticks(np.arange(len(topk_class_names)), topk_class_names)
        plt.gca().invert_yaxis()  # Invert the order to display the highest probability at the top
        plt.xlabel('Probability')
        plt.show()
    else:
        print("Both --image_path and --checkpoint arguments are required.")

if __name__ == '__main__':
    main()