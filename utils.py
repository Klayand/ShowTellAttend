import torch
import torchvision.transforms as transforms
from PIL import Image


def save_checkpoint(state, filename='checkpoint.ckpt'):
    print("===========> Saving checkpoint <==============")
    torch.save(state, filename)


def load_checkpoint(check_point, model, optimizer):
    print("===========> Loading checkpoint <==============")
    model.load_state_dict(check_point['model'])
    optimizer.load_state_dict(check_point['optimizer'])
    step = check_point['step']

    return step


def test(model, image_path, device, dataset):

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    print(
        f"Output: "
        + " ".join(model.caption_image(image_tensor.to(device), dataset.vocab))
          )

    model.train()


