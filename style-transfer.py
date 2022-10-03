import torch

import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGFeature(nn.Module):
    def __init__(self):
        super(VGGFeature, self).__init__()

        # self.model = self.model.cuda()
        # self.model.eval()
        self.chosen_features = [0, 5, 10, 19, 28]
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:29]

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.chosen_features:
                features.append(x)
        return features


def load_image(image_name: str) -> torch.Tensor:
    image_size = 365
    loader = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    raw_image = Image.open(image_name)

    image = raw_image.convert("RGB")

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def print_torch_info():
    print("Torch version: ", torch.__version__)
    print("CUDA available: ", torch.cuda.is_available())
    print("CUDA version: ", torch.version.cuda)
    print("CUDNN version: ", torch.backends.cudnn.version())


def get_progress_bar(percentage: float, bar_length=20) -> str:
    """
    Get progress bar
    :param percentage:
    :param bar_length:
    :return:
    """
    progress = "=" * int(percentage * bar_length)
    spaces = " " * (bar_length - len(progress))
    return f"\rProgress: [{progress}{spaces}] {percentage:.0%}"


def main():
    """
    Main function
    :return:  None
    """
    print_torch_info()
    model = models.vgg19(weights=VGG19_Weights.DEFAULT).features
    # print(model)

    content_name = "Wyeth.png"
    style_name = "VanGogh.png"

    content_img = load_image(f"images/{content_name}")
    style_img = load_image(f"images/{style_name}")

    # torch.randn(content_img.data.size(), device=device, requires_grad=True)

    generated_img = content_img.clone().requires_grad_(True)
    # Hyper parameters
    total_steps = 6000
    learning_rate = 0.001
    alpha = 1
    beta = 0.01
    optimizer = optim.Adam([generated_img], lr=learning_rate)

    model = VGGFeature().to(device).eval()

    start_time = time.time()

    for step in range(total_steps):
        generated_features = model(generated_img)
        content_features = model(content_img)
        style_features = model(style_img)

        style_loss = 0
        content_loss = 0

        for gen_feature, content_feature, style_feature in zip(
                generated_features, content_features, style_features
        ):
            batch_size, channel, height, width = gen_feature.shape

            content_loss += torch.mean((gen_feature - content_feature) ** 2)
            # Gram Matrix
            g = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )
            a = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )
            style_loss += torch.mean((g - a) ** 2)

        total_loss = alpha * content_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 200 == 0:
            curr_time = time.time()
            print(get_progress_bar(step / total_steps))
            print(f"Total loss: {total_loss.item():.4f}")
            print(f"Time elapsed: {curr_time - start_time:.2f}s")
            print(f"Time remaining: {(curr_time - start_time) / (step + 1) * (total_steps - step):.2f}s")

            save_image(generated_img, f"generated/generated{content_name}{step}.png")


if __name__ == "__main__":
    main()
