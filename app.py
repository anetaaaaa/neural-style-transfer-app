from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, g
from mode_style_transfer import StyleTransferModel
from PIL import Image
import io
import torch
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import base64
import torch.nn as nn

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Image transformation
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable to store the output tensor
output_tensor = None

def image_loader(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor_to_image(tensor):
    image = tensor.clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

def image_to_base64(image):
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global output_tensor
        content_image_file = request.files['content_image']
        style_image_file = request.files['style_image']

        # Load images directly from the uploaded files
        content_image = Image.open(content_image_file)
        style_image = Image.open(style_image_file)

        # Pass the images to the StyleTransferModel
        style_transfer = StyleTransferModel(content_image, style_image)
        output = style_transfer.run_style_transfer()
        output_tensor = output
        # Convert the output tensor to an image
        output_image = tensor_to_image(output)

        # Convert the image to Base64 for JSON response
        image_base64 = image_to_base64(output_image)

        return jsonify({'image': image_base64})

    return render_template('index.html')

@app.route('/visualize', methods=['GET'])
def visualize():
    pretrained_model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)

    # Extract convolutional layers from VGG19
    conv_layers = []
    for module in pretrained_model.children():
        if isinstance(module, nn.Conv2d):
            print("Adding module to the layers... ")
            conv_layers.append(module)

    # Pass the resulting image through the convolutional layers and capture feature maps
    feature_maps = []
    layer_names = []
    input_image = output_tensor.clone()

    for i, layer in enumerate(conv_layers):
        input_image = layer(input_image)
        print("Passing through feature maps - layer " , i)
        feature_maps.append(input_image)
        layer_names.append(f"Layer {i + 1}: {str(layer)}")

    # Process and feature maps
    processed_feature_maps = []
    for feature_map in feature_maps:
        print("Processing feature map...")
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        mean_feature_map = torch.mean(feature_map, dim=0).cpu().detach().numpy()  # Compute mean across channels
        processed_feature_maps.append(mean_feature_map)

    # Plot the feature maps
    fig = plt.figure(figsize=(20, 20))
    for i, fm in enumerate(processed_feature_maps):
        print("Plotting feature maps... now at map number ", i)
        ax = fig.add_subplot(4, 4, i + 1)  # Adjust grid size as needed
        ax.imshow(fm, cmap='viridis')  # Display feature map as image
        ax.axis("off")
        ax.set_title(layer_names[i], fontsize=8)

    plt.tight_layout()

    # Save the plot to a BytesIO object and encode it as base64
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close(fig)
    plot_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Return the image as a base64-encoded string that can be embedded in HTML
    return f'<img src="data:image/png;base64,{plot_base64}" alt="Layer Visualizations"/>'

#run the app
if __name__ == '__main__':
    app.run(debug=True)
