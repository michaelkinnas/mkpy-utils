import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from torch import hub
from models import LightNN
from tqdm.auto import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_cifar = (transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
]))

train_dataset = datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms_cifar)
test_dataset = datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms_cifar)

num_images = 2000
training_subset = Subset(train_dataset, range(min(num_images, 50_000)))
test_subset = Subset(test_dataset, range(min(num_images, 10_000)))

train_loader = torch.utils.data.DataLoader(training_subset, batch_size = 32, shuffle = True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size = 32, shuffle = False, num_workers = 2)


teacher = hub.load("chenyaofo/pytorch-cifar-models", model='cifar10_repvgg_a2', pretrained=True)
student = LightNN()


#TODO Find out the order of losses. hard_targets_loss and then soft_target_loss, or the other way around?
#TODO Find out about the temp coefficient on the second term.
def distillation_loss(hard_target_loss, soft_target_loss, alpha, temp=1):
    return alpha * hard_target_loss + (1 - alpha) * soft_target_loss #* temp**2


def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, alpha, temp, device):
    hard_target_loss_fn = nn.CrossEntropyLoss()
    soft_target_loss_fn = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student.parameters(), lr = learning_rate)

    teacher.eval()
    student.train()
    
    epoch_iterator = tqdm(range(epochs), desc='Epoch: ', leave=False, position=0)        
    for epoch in epoch_iterator:
        running_loss = 0.0

        batch_iterator = tqdm(train_loader, total=len(train_loader), desc="Batch: ", leave=False, position=1)            
        for train_batch_idx, (X, y) in enumerate(batch_iterator):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(X)

            student_logits = student(X)

            #needs logSoftmax for teacher logits since the kldivloss function requires the log value for the first input
            teacher_probs = nn.functional.log_softmax(teacher_logits / temp, dim = 1) 
            student_probs = nn.functional.softmax(student_logits / temp, dim = 1)

            hard_target_loss = hard_target_loss_fn(student_logits, y)
            soft_target_loss = soft_target_loss_fn(teacher_probs, student_probs)
            

            loss = distillation_loss(hard_target_loss, soft_target_loss, alpha, temp=temp)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_iterator.set_postfix_str(f"Hard target loss: {hard_target_loss:.4f}, Soft target loss: {soft_target_loss:.4f}, Final loss: {loss:.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")


train_knowledge_distillation(teacher, student, train_loader, epochs=10, learning_rate=0.001, alpha=0.2, temp=5, device=device)