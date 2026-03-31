import torch
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision import transforms
import argparse


# ✅ argparse added
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data')
args = parser.parse_args()


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4467, 0.4398, 0.4066),
                         (0.2603, 0.2566, 0.2713))
])

test_dataset = STL10(root=args.data_path, split='test',
                     transform=test_transform, download=False)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
import torch.fx.graph_module

import torch
from model import Model
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig
if "fbgemm" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "fbgemm"
else:
    torch.backends.quantized.engine = torch.backends.quantized.supported_engines[0]
model = Model().eval()
qconfig = get_default_qconfig(torch.backends.quantized.engine)
example_input = torch.randn(1, 3, 96, 96)

prepared = prepare_fx(model, {"": qconfig}, example_input)
model = convert_fx(prepared)


model.load_state_dict(torch.load("best_quantized_model.pth", map_location="cpu"))
model.eval()


correct = 0
total = 0

with torch.no_grad():
    for img, label in test_loader:
        output = model(img)
        preds = torch.argmax(output, dim=1)

        correct += (preds == label).sum().item()
        total += label.size(0)

print("Test Accuracy:", correct / total)