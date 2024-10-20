from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        #Define the convolutional layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), #(B, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), #(B, 32, 32, 32)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 128, 8, 8)

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # (B, 256, 4, 4)

        )
        
        #Define the fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass

        #Pass through the convolutional layer
        z = self.conv_layers(z)

        #Flatten the output from the convolutional layer
        z = z.view(z.size(0), -1)

        logits = self.fc_layers(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()

        # Down-sampling layers
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 16),  # Down1
            self.conv_block(16, 32),            # Down2
            self.conv_block(32, 64),            # Down3
        )

        # Up-sampling layers with skip connections
        self.up1 = self.upconv_block(64, 32)  # Up1
        self.up2 = self.upconv_block(32, 16)  # Up2
        self.up3 = self.upconv_block(16, 16)  # Additional upsample to reach original size

        # Output layers
        self.logits_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_conv = nn.Conv2d(16, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsample by a factor of 2
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        enc_out1 = self.encoder[0](x)  # Down1
        enc_out2 = self.encoder[1](enc_out1)  # Down2
        enc_out3 = self.encoder[2](enc_out2)  # Down3

        # Decoder with skip connections
        dec_out1 = self.up1(enc_out3) + enc_out2  # Skip from Down2
        dec_out2 = self.up2(dec_out1) + enc_out1  # Skip from Down1
        dec_out3 = self.up3(dec_out2)  # Additional upsampling

        # Output layers
        logits = self.logits_conv(dec_out3)  # Ensure the shape matches target size
        depth = self.depth_conv(dec_out3)

        return logits, depth

    def predict(self, x: torch.Tensor, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels and normalized depth for the input images in batches.

        Args:
            x (torch.FloatTensor): image tensor of shape (B, 3, H, W) with values in [0, 1].
            batch_size (int): the number of samples to process at a time.

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: predicted class labels {0, 1, 2} with shape (B, H, W)
                - depth: normalized depth values in [0, 1] with shape (B, H, W)
        """
        # Split the input into smaller batches
        num_samples = x.size(0)
        preds = []
        depths = []

        # Loop over the input data in batches
        for i in range(0, num_samples, batch_size):
            # Get the current batch
            x_batch = x[i:i + batch_size]

            # Perform inference on the batch
            logits, raw_depth = self(x_batch)

            # Get predicted class labels
            pred_batch = logits.argmax(dim=1)  # (B, H, W)

            # Normalize depth to [0, 1] range using sigmoid
            depth_batch = torch.sigmoid(raw_depth).squeeze(1)  # (B, H, W)

            # Append the batch results
            preds.append(pred_batch)
            depths.append(depth_batch)

        # Concatenate all batch results along the batch dimension
        preds = torch.cat(preds, dim=0)
        depths = torch.cat(depths, dim=0)

        return preds, depths
    


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
