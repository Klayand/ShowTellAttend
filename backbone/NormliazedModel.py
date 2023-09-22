import torch
from torchvision import transforms


class FlickrNormModel(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Module,
            transform=transforms.Compose(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
            **kwargs
    ):
        super(FlickrNormModel, self).__init__()
        self.model = model
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, y):
        x = self.transform(x)
        return self.model(x, y)




