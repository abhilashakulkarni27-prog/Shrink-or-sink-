import torch as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import argparse

from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from model import Model


# ✅ argparse added
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data')
args = parser.parse_args()


# device
if pt.cuda.is_available():
    device = "cuda"
elif pt.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)

    if pt.cuda.is_available():
        pt.cuda.manual_seed(seed)
        pt.cuda.manual_seed_all(seed)

    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False

set_seed(42)


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# dataset
current_path = os.getcwd()
data_path = os.path.join(current_path, "data")

train_dataset = STL10(root=args.data_path, split='train', download=True)
test_dataset = STL10(root=args.data_path, split='test', download=True)


train_transform = transforms.Compose([
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4467, 0.4398, 0.4066),
        (0.2603, 0.2566, 0.2713)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
])


train_dataset = STL10(root=args.data_path, split='train',
                      transform=train_transform, download=False)

test_dataset = STL10(root=args.data_path, split='test',
                     transform=test_transform, download=False)


g = pt.Generator()
g.manual_seed(42)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=0, pin_memory=True,
                          worker_init_fn=seed_worker, generator=g)

test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False,
                          num_workers=0, pin_memory=True)


model = Model()
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

criterion = nn.CrossEntropyLoss()


# ===== FIRST LOOP (UNCHANGED) =====
for epoch in range(50):
    model.train()
    total_loss = 0

    for img, label in train_loader:
        img, label = img.to(device), label.to(device)

        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Loss of epoch {epoch} is {total_loss/len(train_loader)}")

    model.eval()
    correct = 0
    total = 0

    with pt.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)

            output = model(img)
            preds = pt.argmax(output, dim=1)

            correct += (preds == label).sum().item()
            total += label.size(0)

    print(f"Accuracy: {correct/total:.4f}")

    scheduler.step()


# ===== SECOND LOOP (UNCHANGED) =====
for param_group in optimizer.param_groups:
    param_group['lr'] = 5e-5


best_acc = 0.7262
for epoch in range(50,80):
    model.train()
    total_loss = 0

    for img, label in train_loader:
        img, label = img.to(device), label.to(device)

        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Loss of epoch {epoch} is {total_loss/len(train_loader)}")

    model.eval()
    correct = 0
    total = 0
    acc=0

    with pt.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)

            output = model(img)
            preds = pt.argmax(output, dim=1)

            correct += (preds == label).sum().item()
            total += label.size(0)

        acc = correct / total

    print(f"Accuracy: {correct/total:.4f}")


# ===== QUANTIZATION (UNCHANGED except earlier fixes) =====

import torch
if "fbgemm" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "fbgemm"
else:
    torch.backends.quantized.engine = torch.backends.quantized.supported_engines[0]

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

qconfig = get_default_qconfig(torch.backends.quantized.engine)

model.to('cpu')
model.eval()

example_input = torch.randn(1, 3, 96, 96)

prepared_model = prepare_fx(model, {"": qconfig}, example_input)


calibration_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

prepared_model.to('cpu')

with torch.no_grad():
    for i, (images, _) in enumerate(calibration_loader):
        prepared_model(images.to("cpu"))
        if i > 200:
            break


quantized_model = convert_fx(prepared_model)

# After quantization and convert_fx:
torch.save(quantized_model.state_dict(), "best_quantized_model.pth")


size_kb = os.path.getsize("best_quantized_model.pth") / 1024
print(f"Model size: {size_kb:.2f} KB")