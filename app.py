import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image
import uuid
import os

# **Device Configuration**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 512 if torch.cuda.is_available() else 256  # Adjust based on GPU availability

# **Preprocessing Functions**
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found")
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# **Normalization Values**
mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

def normalize(tensor):
    return (tensor - mean) / std

def denormalize(tensor):
    return tensor * std + mean

# **VGG19 Model for Feature Extraction**
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(weights="IMAGENET1K_V1").features.eval().to(device)
        
        self.layers = {
            '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
            '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
        }
        
        self.model = nn.Sequential(*list(vgg.children())[:29])  # Extract first 29 layers

    def forward(self, x):
        features = {}
        for i, layer in enumerate(self.model):
            x = layer(x)
            if str(i) in self.layers:
                features[self.layers[str(i)]] = x
        return features

# **Loss Functions**
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)

def gram_matrix(tensor):
    batch_size, f, h, w = tensor.size()
    features = tensor.view(f, h * w)
    gram = torch.mm(features, features.t())
    return gram / (f * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target_gram = gram_matrix(target_features).detach()

    def forward(self, input):
        return nn.functional.mse_loss(gram_matrix(input), self.target_gram)

# **Neural Style Transfer Function**
def neural_style_transfer(content_path, style_path, num_steps=300, content_weight=1e5, style_weight=1e10):
    try:
        # **Load Images**
        content_img = load_image(content_path)
        style_img = load_image(style_path)
        generated_img = content_img.clone().requires_grad_(True)

        # **Initialize Model**
        model = VGGFeatures().to(device)
        content_features = model(normalize(content_img))
        style_features = model(normalize(style_img))

        # **Set Up Loss Functions**
        content_loss = ContentLoss(content_features["conv4_2"])
        style_losses = [StyleLoss(style_features[layer]) for layer in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]]

        # **Optimizer**
        optimizer = optim.LBFGS([generated_img])

        # **Optimization Loop**
        run = [0]
        while run[0] < num_steps:
            def closure():
                optimizer.zero_grad()
                gen_features = model(normalize(generated_img))

                # Compute Losses
                content_score = content_loss(gen_features["conv4_2"])
                style_score = sum(sl(gen_features[layer]) for layer, sl in zip(["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"], style_losses))

                loss = content_weight * content_score + style_weight * style_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step {run[0]}, Content Loss: {content_score.item():.4f}, Style Loss: {style_score.item():.4f}")

                return loss

            optimizer.step(closure)

        # **Save and Return Image**
        output_path = f"output_{uuid.uuid4().hex}.png"
        save_image(generated_img.clamp(0, 1), output_path)
        print(f"Stylized image saved as: {output_path}")

        return output_path

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# **Example Usage**
if __name__ == "__main__":
    content_image_path = "CONTENT.jpg"   # Content image
    style_image_path = "STYLE.png"     # Style image
    neural_style_transfer(content_image_path, style_image_path)