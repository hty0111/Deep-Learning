import torch
from torch import nn
import torchvision
from torchvision import transforms
from models.LeNet import LeNet
from config.baseConfig import BaseConfig
import os
from tqdm import tqdm

def main():
    cfg = BaseConfig().getArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = torchvision.datasets.FashionMNIST(root=cfg.data_root, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model_path = os.path.join(cfg.model_root, f"{cfg.model_name}_{cfg.epochs}epochs.pth")
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path))

    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

if __name__=="__main__":
    main()