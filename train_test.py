import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from torch import hub
from trainer import Trainer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_cifar = (transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
]))

train_dataset = datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms_cifar)
test_dataset = datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms_cifar)

num_images = 1000
training_subset = Subset(train_dataset, range(min(num_images, 50_000)))
test_subset = Subset(test_dataset, range(min(num_images, 10_000)))

train_loader = torch.utils.data.DataLoader(training_subset, batch_size = 32, shuffle = False, num_workers = 2)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size = 32, shuffle = False, num_workers = 2)


model = hub.load("chenyaofo/pytorch-cifar-models", model='cifar10_repvgg_a2', pretrained=False)
# student = hub.load("chenyaofo/pytorch-cifar-models", model='cifar10_mobilenetv2_x0_5', pretrained=False)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

trainer = Trainer(model, optimizer, loss_fn, train_loader, test_loader, epochs=2, seed=1000)
trainer.train_model(record_train_progress=True, use_tqdm=False)

print(trainer.get_train_progress())