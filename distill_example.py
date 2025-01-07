import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from torch import hub
from models import LightNN
from tqdm.auto import tqdm
from distillation import ResponseDistiller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_cifar = (transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
]))

train_dataset = datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms_cifar)
test_dataset = datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms_cifar)

num_images = 50000
training_subset = Subset(train_dataset, range(min(num_images, 50_000)))
test_subset = Subset(test_dataset, range(min(num_images, 10_000)))

train_loader = torch.utils.data.DataLoader(training_subset, batch_size = 32, shuffle = True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size = 32, shuffle = False, num_workers = 2)


teacher = hub.load("chenyaofo/pytorch-cifar-models", model='cifar10_repvgg_a2', pretrained=True)
student = hub.load("chenyaofo/pytorch-cifar-models", model='cifar10_mobilenetv2_x0_5', pretrained=False)


hard_target_loss_fn = nn.CrossEntropyLoss()
soft_target_loss_fn = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(student.parameters(), lr = 0.001)

distiller = ResponseDistiller(teacher, student, optimizer, train_loader, test_loader, epochs=10, alpha=0.2, soft_loss_temp=5, final_loss_temp=1, seed=1000)
distiller.distill()

# train_knowledge_distillation(teacher, student, train_loader, epochs=10, learning_rate=0.001, alpha=0.2, temp=5, device=device)