# %%
import os

img_dir = '/content/sample_image' # need to upload image here for demo

if not os.path.exists(img_dir):
  os.makedirs(img_dir)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from google.colab import drive
import os
import time as time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageLoader:
    """Class for loading images from URLs or local paths and preprocessing them"""
    def __init__(self, max_dim=512):
        self.max_dim = max_dim

    def load_image_from_url(self, url):
        """Loads an image from a URL"""
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return self._process(image)

    def load_image_from_path(self, path):
        """Loads an image from a local path (e.g., Google Drive)"""
        image = Image.open(path).convert('RGB')
        return self._process(image)

    def _process(self, image):
        """Resize and convert image to tensor"""
        long = max(image.size)
        scale = self.max_dim / long
        image = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.LANCZOS)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0).to(device)


class VGGFeatures:
    """Class for extracting features from a pre-trained VGG19 model"""
    def __init__(self):
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers):
        """Extracts feature maps from specific layers"""
        features = {}
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[name] = x
        return features

    @staticmethod
    def gram_matrix(tensor):
        """Computes the Gram matrix for style representation"""
        b, c, h, w = tensor.size()
        features = tensor.view(c, h * w)
        return torch.mm(features, features.t()) / (c * h * w)


class StyleTransfer:
    """Main class for performing style transfer"""
    def __init__(self, content_path, style_path, style_weight=5e7, num_steps=400, lr=0.003, optimizer_type="Adam"):
        self.style_weight = style_weight
        self.num_steps = num_steps
        self.lr = lr
        self.optimizer_type = optimizer_type  # Adam or LBFGS

        # Load images from file paths
        loader = ImageLoader()
        self.content_img = loader.load_image_from_path(content_path)
        self.style_img = loader.load_image_from_path(style_path)

        # Load pre-trained VGG19 model
        self.vgg = VGGFeatures()

        # Define content and style layers
        self.content_layers = ['22']  # Conv4_2
        self.style_layers = ['0', '5', '10', '19', '28']

        # Extract content and style features
        self.content_features = self.vgg.get_features(self.content_img, self.content_layers)
        self.style_features = self.vgg.get_features(self.style_img, self.style_layers)

        # Compute Gram matrices for style layers
        self.style_grams = {layer: self.vgg.gram_matrix(self.style_features[layer]) for layer in self.style_layers}

        # Initialize generated image
        self.generated_img = self.content_img.clone().requires_grad_(True)

        # Choose optimizer
        if self.optimizer_type == "LBFGS":
            self.optimizer = optim.LBFGS([self.generated_img])
        else:  # Default to Adam
            self.optimizer = optim.Adam([self.generated_img], lr=self.lr)

        # Store intermediate images
        self.images_at_steps = []

    def show_initial_images(self):
        """Displays content and style images before training starts"""
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Convert tensors to PIL images
        content_pil = transforms.ToPILImage()(self.content_img.cpu().squeeze(0).clamp(0, 1))
        style_pil = transforms.ToPILImage()(self.style_img.cpu().squeeze(0).clamp(0, 1))

        # Show images
        ax[0].imshow(content_pil)
        ax[0].set_title("Content Image")
        ax[0].axis("off")

        ax[1].imshow(style_pil)
        ax[1].set_title("Style Image")
        ax[1].axis("off")

        plt.show()

    def train(self):
        """Runs the optimization process for style transfer"""
        self.show_initial_images()  # Show content & style images

        if self.optimizer_type == "LBFGS":
            self.train_lbfgs()
        else:
            self.train_adam()

        self.show_progress()
        self.save_image()

    def train_adam(self):
        """Trains the model using Adam optimizer"""
        # list for graph
        total_losses = []
        content_losses = []
        style_losses = []

        for step in range(self.num_steps):
            self.optimizer.zero_grad()

            # Extract features
            generated_features = self.vgg.get_features(self.generated_img, self.content_layers + self.style_layers)

            # Compute losses
            content_loss, style_loss = self.compute_losses(generated_features)
            total_loss = content_loss + style_loss * self.style_weight

            # Backprop
            total_loss.backward()
            self.optimizer.step()

            # loss for graph
            total_losses.append(total_loss.item())
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())

            if step % 50 == 0:
                print(f"Step {step}, Total Loss: {total_loss.item():.4f}, Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.10f}")
                self.images_at_steps.append(self.generated_img.clone().detach())

        # Plot Graph
        plt.figure(figsize=(10, 5))
        plt.plot(total_losses, label='Total Loss')
        plt.plot(content_losses, label='Content Loss')
        plt.plot(style_losses, label='Style Loss')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Adam Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def train_lbfgs(self):
        """Trains the model using LBFGS optimizer"""
        loss_value = None
        content_loss_value = None
        style_loss_value = None

        # list for graph
        total_losses = []
        content_losses = []
        style_losses = []

        def closure():
            nonlocal loss_value, content_loss_value, style_loss_value
            self.optimizer.zero_grad()
            generated_features = self.vgg.get_features(self.generated_img, self.content_layers + self.style_layers)
            content_loss, style_loss = self.compute_losses(generated_features)
            total_loss = content_loss + style_loss * self.style_weight
            total_loss.backward()

            # Save loss values for printing & graph
            content_loss_value = content_loss.item()
            style_loss_value = style_loss.item()
            loss_value = total_loss.item()

            return total_loss

        for step in range(self.num_steps):
            self.optimizer.step(closure)

            # Save to graph lists
            total_losses.append(loss_value)
            content_losses.append(content_loss_value)
            style_losses.append(style_loss_value)

            if step % 50 == 0:
                print(f"Step {step}, Total Loss: {loss_value:.4f}, Content Loss: {content_loss_value:.4f}, Style Loss: {style_loss_value:.10f}")
                self.images_at_steps.append(self.generated_img.clone().detach())

        # Plot Graph
        plt.figure(figsize=(10, 5))
        plt.plot(total_losses, label='Total Loss')
        plt.plot(content_losses, label='Content Loss')
        plt.plot(style_losses, label='Style Loss')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("LBFGS Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def compute_losses(self, generated_features):
        """Computes content and style loss"""
        content_loss = sum(
            torch.mean((generated_features[layer] - self.content_features[layer]) ** 2)
            for layer in self.content_layers
        )

        style_loss = sum(
            torch.mean((self.vgg.gram_matrix(generated_features[layer]) - self.style_grams[layer]) ** 2)
            for layer in self.style_layers
        )

        return content_loss, style_loss

    def show_progress(self):
        """Displays a sequence of generated images during training"""
        num_images = len(self.images_at_steps)
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

        for i, img_tensor in enumerate(self.images_at_steps):
            img_pil = transforms.ToPILImage()(img_tensor.cpu().squeeze(0).clamp(0, 1))
            axes[i].imshow(img_pil)
            axes[i].set_title(f"Step {i * 50}")
            axes[i].axis("off")

        plt.show()

    def save_image(self, output_path='generated_image.jpg'):
        """Saves the final generated image and prints final losses"""
        # Calculate final losses
        with torch.no_grad():
            features = self.vgg.get_features(self.generated_img, self.content_layers + self.style_layers)
            content_loss, style_loss = self.compute_losses(features)
            total_loss = content_loss + style_loss * self.style_weight

        # Save and show image
        output_image = transforms.ToPILImage()(self.generated_img.cpu().squeeze(0).clamp(0, 1))
        output_image.save(output_path)
        print(f"Generated image saved to {output_path}")

        # Print losses
        print(f"Final Total Loss: {total_loss.item():.4f}")
        print(f"Final Content Loss: {content_loss.item():.4f}")
        print(f"Final Style Loss: {style_loss.item():.10f}")

        # Show image
        plt.imshow(output_image)
        plt.title("Final Generated Image")
        plt.axis("off")
        plt.show()


# %%
# Optimizer: LBFGS
# Style Weight: Big

if __name__ == "__main__":
    # Image paths
    content_path = "/content/sample_image/aityan.jpg"
    style_path = "/content/sample_image/blue.jpeg"

    # Hyperparameters
    style_weight = 5e7 # 1e5 - 5e7
    num_steps = 150
    learning_rate = 0.1 # No need for LBFGS
    optimizer_type = "LBFGS" # Adam or LBFGS

    # (optional) Check execution time
    start_time = time.time()

    # Run Style Transfer with selectable optimizer
    style_transfer = StyleTransfer(content_path, style_path, style_weight, num_steps, learning_rate, optimizer_type)
    style_transfer.train()

    # (optional) Check execution time
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")


# %%
# Optimizer: LBFGS
# Style Weight: Big

if __name__ == "__main__":
    # Image paths
    content_path = "/content/sample_image/neckarfront-tbingen.jpg"
    style_path = "/content/sample_image/Gustave-Caillebotte-Petit-Gennevilliers.jpg"

    # Hyperparameters
    style_weight = 5e7 # 1e5 - 5e7
    num_steps = 200
    learning_rate = 0.1 # No need for LBFGS
    optimizer_type = "LBFGS" # Adam or LBFGS

    # (optional) Check execution time
    start_time = time.time()

    # Run Style Transfer with selectable optimizer
    style_transfer = StyleTransfer(content_path, style_path, style_weight, num_steps, learning_rate, optimizer_type)
    style_transfer.train()

    # (optional) Check execution time
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")

# %%
# Optimizer: LBFGS
# Style Weight: Big

if __name__ == "__main__":
    # Image paths
    content_path = "/content/sample_image/Gustave-Caillebotte-Petit-Gennevilliers.jpg"
    style_path = "/content/sample_image/Amelanchier-x-grandiflora-in-fall.jpg"

    # Hyperparameters
    style_weight = 5e7 # 1e5 - 5e7
    num_steps = 200
    learning_rate = 0.1 # No need for LBFGS
    optimizer_type = "LBFGS" # Adam or LBFGS

    # (optional) Check execution time
    start_time = time.time()

    # Run Style Transfer with selectable optimizer
    style_transfer = StyleTransfer(content_path, style_path, style_weight, num_steps, learning_rate, optimizer_type)
    style_transfer.train()

    # (optional) Check execution time
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")

# %%
# --- For DEMO ---
# Optimizer: Adam
# Style Weight: Big

if __name__ == "__main__":
    # Image paths
    content_path = "/content/sample_image/neckarfront-tbingen.jpg"
    style_path = "/content/sample_image/Gustave-Caillebotte-Petit-Gennevilliers.jpg"

    # Hyperparameters
    style_weight = 5e7 # 1e5 - 5e7
    num_steps = 200
    learning_rate = 0.1 # No need for LBFGS
    optimizer_type = "Adam" # Adam or LBFGS

    # (optional) Check execution time
    start_time = time.time()

    # Run Style Transfer with selectable optimizer
    style_transfer = StyleTransfer(content_path, style_path, style_weight, num_steps, learning_rate, optimizer_type)
    style_transfer.train()

    # (optional) Check execution time
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")


