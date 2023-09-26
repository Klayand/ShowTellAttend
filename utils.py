import torch
import torchvision.transforms as transforms
from PIL import Image


def save_checkpoint(state, filename='checkpoint.ckpt'):
    print("===========> Saving checkpoint <==============")
    torch.save(state, filename)


def load_checkpoint(check_point, model, optimizer=None):
    print("===========> Loading checkpoint <==============")
    check_point = torch.load(check_point, map_location='cuda')
    model.load_state_dict(check_point['model'], strict=False)

    if optimizer:
        optimizer.load_state_dict(check_point['optimizer'])

    step = check_point['step']

    return step


def test(model, dataset, image_path, device, ckpt):
    model.to(device)

    load_checkpoint(ckpt, model)

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


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: A black dog and a spotted dog are fighting")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )

    test_img2 = transform(
        Image.open("test_examples/child.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: A child playing on a rope net")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )

    test_img3 = transform(Image.open("test_examples/couple.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: A couple and an infant, being held by the male, sitting next to a pond with a near by stroller")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )

    model.train()
