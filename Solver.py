import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from optimizer import default_optimizer, ALRS
from torch.utils.tensorboard import SummaryWriter

from cfg import CFG
from utils import load_checkpoint, save_checkpoint


def default_loss(x, y):
    cross_entropy = F.cross_entropy(x, y, ignore_index=dataset.vocab.stoi["<PAD>"])
    return cross_entropy


class Solver:
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 loss_function: Callable or None = None,
                 optimizer: torch.optim.Optimizer or None = None,
                 scheduler=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        self.model = model
        self.criterion = loss_function if loss_function is not None else default_loss
        self.optimizer = optimizer if optimizer is not None else default_optimizer(self.model)
        self.scheduler = scheduler if scheduler is not None else ALRS(self.optimizer)
        self.dataset = dataset

        self.device = device

        # initialization
        self.init()

    def init(self):
        # change device
        self.model.to(self.device)

        # tensorboard
        self.writer = SummaryWriter(log_dir="runs/")

        # finetuning
        for name, param in self.model.encoderCNN.encoder.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def train(self,
              train_loader: DataLoader,
              total_epoch=400,
              load_model=False,
              save_model=True,
              ):

        step = 0

        if load_model:
            step = load_checkpoint('checkpoint.ckpt', self.model, self.optimizer)

        self.model.train()
        for epoch in range(1, total_epoch + 1):
            print_examples(self.model, self.device, self.dataset)

            train_loss = 0

            # train
            pbar = tqdm(train_loader)
            for idx, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x, y[:-1])

                loss = self.criterion(
                    out.reshape(-1, out.shape[2]), y.reshape(-1)
                )

                step += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss={train_loss / step}')

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.writer.add_scalar("Training loss", train_loss, global_step=step)

            self.scheduler.step(train_loss, epoch)
            # self.optimizer.step()

            print(f'epoch {epoch}, train_loss = {train_loss}')
            print('-' * 100)

            if save_model:
                checkpoint = {
                    'model': self.model.state_dict(),
                    'encoder': self.model.encoderCNN.state_dict(),
                    'decoder': self.model.decoderRNN.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'step': step
                }
                save_checkpoint(checkpoint)


if __name__ == '__main__':
    from backbone import CNNtoRNN
    from data import get_flickr_loader
    from utils import test, print_examples

    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloader, dataset = get_flickr_loader(root_dir=CFG.ROOT_DIR, annotation_file=CFG.ANNOTATION_DIR, transform=transform, num_workers=0)

    CFG.VOCAB_SIZE = len(dataset.vocab)

    a = CNNtoRNN(CFG)

    solver = Solver(model=a, dataset=dataset)
    solver.train(train_loader=dataloader, total_epoch=500)

    test(a, dataset, 'flickr8k/images/109202756_b97fcdc62c.jpg', 'cuda', 'checkpoint.ckpt')
