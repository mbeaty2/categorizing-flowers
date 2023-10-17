#contains all functions necessary for this project

# Load's the checkpoint and rebuilds the model
def load_checkpoint(filepath):
    # Loading the checkpoint from before
    checkpoint = torch.load(filepath)
    
    #recreating the model
    model = models.vgg16(pretrained=True)
    classifier = nn.Sequential(
        nn.Linear(25088, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 512),
        nn.Linear(512, 102),
        nn.LogSoftmax(dim=1)
    )
    
    #updating the classifier to the one saved in the checkpoint
    model.classifier = classifier
    
    #loading the saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #loading the class to index
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, checkpoint['optimizer_state_dict'], checkpoint['epoch'], checkpoint['best_accuracy']

#process the image 
def process_image(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Resize the image where the shortest side is 256 pixels, maintaining aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = int(new_height * aspect_ratio)
    
    resized_image = image.resize((new_width, new_height))
    
    # Crop out the center 224x224 portion
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    # Convert color values to a NumPy array and normalize
    np_image = np.array(cropped_image) / 255.0
    
    # Normalize the image using mean and standard deviation
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    np_image = (np_image - means) / stds
    
    # Reorder dimensions to match PyTorch format (C, H, W)
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

#diplay the image 

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#predict the image's top 5 classes 
def predict_class(image_path, model, topk=5):
    # Load and preprocess the image
    image = process_image(image_path)
    
    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.FloatTensor(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension
    
    # Move the tensor to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Perform forward pass
    with torch.no_grad():
        output = model(image_tensor)
    
    # Calculate class probabilities
    probabilities = torch.exp(output)
    
    # Get the top K probabilities and indices
    topk_probabilities, topk_indices = torch.topk(probabilities, topk)
    
    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    topk_class_labels = [idx_to_class[idx.item()] for idx in topk_indices[0]]
    
    # Convert probabilities and class labels to lists
    topk_probabilities = topk_probabilities[0].cpu().numpy()
    
    return topk_probabilities, topk_class_labels