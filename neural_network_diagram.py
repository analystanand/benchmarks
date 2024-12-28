import os
import torch
import torch.nn as nn
from graphviz import Digraph
from PIL import Image

def visualize_model_with_separate_hidden_layers():
    class PyTorchModel(nn.Module):
        def __init__(self):
            super(PyTorchModel, self).__init__()
            self.fc1 = nn.Linear(200, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 16)
            self.fc5 = nn.Linear(16, 8)
            self.fc6 = nn.Linear(8, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.relu(self.fc5(x))
            x = self.sigmoid(self.fc6(x))
            return x

    def make_detailed_architecture_diagram():
        layers = [
            ("Input", "200"),
            ("Hidden Layer 1", "128"),
            ("Hidden Layer 2", "64"),
            ("Hidden Layer 3", "32"),
            ("Hidden Layer 4", "16"),
            ("Hidden Layer 5", "8"),
            ("Output", "1")
        ]

        dot = Digraph(format="png")
        dot.attr(rankdir="LR", size="12,6", pad="0.2")  # Left-to-right layout with padding

        for i, (layer_name, details) in enumerate(layers):
            dot.node(layer_name, f"{layer_name}\n{details}", shape="box", style="filled", fillcolor="lightblue")
            if i > 0:
                dot.edge(layers[i - 1][0], layer_name)

        dot.render("detailed_model_architecture", format="png", cleanup=True)
        print("Detailed model architecture saved as 'detailed_model_architecture.png'.")

    def add_white_padding_to_image():
        # Load the generated image
        img = Image.open("detailed_model_architecture.png")

        # Create a new image with white background and target size
        padded_img = Image.new("RGB", (1280, 720), "white")

        # Center the original image on the white background
        img_w, img_h = img.size
        padded_img.paste(img, ((1280 - img_w) // 2, (720 - img_h) // 2))

        # Save the padded image
        padded_img.save("detailed_model_architecture_with_padding.png")
        print("Image with white padding saved as 'detailed_model_architecture_with_padding.png'.")

    # Instantiate the model
    model = PyTorchModel()

    # Generate the detailed diagram
    make_detailed_architecture_diagram()

    # Post-process the image to add white padding
    add_white_padding_to_image()

visualize_model_with_separate_hidden_layers()
