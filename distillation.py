import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from torch import hub
from models import LightNN
from tqdm.auto import tqdm

from torch import manual_seed, inference_mode, argmax, cuda
from tqdm.auto import tqdm
import csv
import time
import os
from torch import save as save_dict

class ResponseDistiller:
    def __init__(self, teacher, student, optimizer, train_dataloader, validation_dataloader = None, lr_scheduler=None,
                 epochs: int=2, alpha=0.2, soft_loss_temp = 1, final_loss_temp = 1, device='auto', seed=None, validation_step: int=1, checkpoint_step: int=0):
        """
        A training loop for the Response Based Distillation method
        """
        self.__teacher = teacher
        self.__student = student
        
        self.__trainloader = train_dataloader
        self.__optimizer = optimizer  #for student
        self.__scheduler = lr_scheduler #for student
        # self.__loss_fn = loss_fn
        self.__epochs = epochs

        self.__alpha = alpha #the weights given to the final loss function hard targetrs and soft targets coefficients
        self.__soft_loss_temp = soft_loss_temp #the temperature to soften the soft target vector values
        self.__final_loss_temp = final_loss_temp

        if device == 'auto':
            self.__device = 'cuda' if cuda.is_available() else 'cpu'
        else:
            self.__device = device

        self.__seed = seed
        self.__valloader = validation_dataloader
        self.__validation_step = validation_step #steps in epochs
        self.__checkpoint_step = checkpoint_step #steps in epochs

        self.__train_progess = {
                'epoch':[],
                'batch':[],
                'train_acc' : [],
                'train_loss': [], 
            }
        
        self.__validation_progress = {
                'epoch':[],
                'val_acc': [],
                'val_loss': [],
        }

        # self.__model = self.__model.to(self.__device) #??

        self.__teacher = self.__teacher.to(self.__device)
        self.__student = self.__student.to(self.__device)

    def distill(self, record_train_progress=False, record_validation_progress=False, window: int = 20):
        if self.__seed:
            manual_seed(self.__seed)
        
        hard_target_loss_fn = nn.CrossEntropyLoss()                 #in case of multiclass classification
        soft_target_loss_fn = nn.KLDivLoss(reduction='batchmean')   #does this work for other logits vectors such as from binary entropy or others?
        optimizer = self.__optimizer

        TRAIN_NUM_BATCHES = len(self.__trainloader)
        TRAIN_BATCH_SIZE = self.__trainloader.batch_size

        if self.__valloader:
            VAL_NUM_BATCHES = len(self.__valloader)
            VAL_BATCH_SIZE = self.__valloader.batch_size

        # self.__start_timer()

        epoch_iterator = tqdm(range(self.__epochs), desc='Epoch: ', leave=False, position=0)
        
        for epoch in epoch_iterator:
            running_loss = 0
            running_acc = 0

            batch_accuracies = []

            self.__student.train()

            if self.__scheduler:
                lr = self.__scheduler.get_last_lr()[0]
            else:
                lr = self.__optimizer.param_groups[-1]['lr']

            batch_iterator = tqdm(self.__trainloader, total=len(self.__trainloader), desc="Batch: ", leave=False, position=1)            
            for train_batch_idx, (X_train, y_train) in enumerate(batch_iterator):
                X_train, y_train = X_train.to(self.__device), y_train.to(self.__device)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_logits = self.__teacher(X_train)

                student_logits = self.__student(X_train)

                #needs logSoftmax for teacher logits since the kldivloss function requires the log value for the first input
                teacher_probs = nn.functional.log_softmax(teacher_logits / self.__soft_loss_temp, dim = 1)
                student_probs = nn.functional.softmax(student_logits / self.__soft_loss_temp, dim = 1)

                hard_target_loss = hard_target_loss_fn(student_logits, y_train)
                soft_target_loss = soft_target_loss_fn(teacher_probs, student_probs)
                

                loss = self.__distillation_loss(hard_target_loss, soft_target_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Report train progress to tqdm
                batch_iterator.set_postfix_str(f"Hard target loss: {hard_target_loss:.4f}, Soft target loss: {soft_target_loss:.4f}, Final loss: {loss:.4f}")

                # if record_train_progress:
                #     self.__record_training_step(loss, 
                #                                 accuracy / len(y_train),
                #                                 train_batch_idx,
                #                                 epoch)              
              

            # Save snapshot
            if self.__checkpoint_step > 0 and self.__checkpoint_step % epoch+1 == 0:
                self.save_model_dictionary(f'./weights_ep:{epoch+1}.pth', append_accuracy=True)                

            if self.__valloader:
                running_loss, running_acc = self.__validation(epoch, record_validation_progress)
                epoch_iterator.set_postfix_str(f"Validation loss: {running_loss / VAL_NUM_BATCHES:.4f} | Validation accuracy: {running_acc / len(self.__valloader) * 100:.2f}%")
            else:
                epoch_iterator.set_postfix_str(f"Previous epoch: Training loss: {running_loss / (train_batch_idx+1):.4f} | Training accuracy: {running_acc / (TRAIN_BATCH_SIZE * (train_batch_idx) + len(y_train)) * 100:.2f}%")

            if self.__scheduler:
                self.__scheduler.step()


    def __distillation_loss(self, hard_target_loss, soft_target_loss):
        return self.__alpha * hard_target_loss + (1 - self.__alpha) * soft_target_loss * self.__final_loss_temp**2


    def __validation(self, epoch, record_validation_progress):
        if epoch % self.__validation_step == 0 or epoch == self.__epochs-1:
            running_acc = 0
            running_loss = 0

            loss_fn = nn.CrossEntropyLoss() 

            self.__student.eval()

            with inference_mode():
                vallidation_batch_iterator = tqdm(self.__valloader, total=len(self.__valloader), desc="Validation: ", leave=False, position=2)
                for val_batch_idx, (X_test, y_test) in enumerate(vallidation_batch_iterator):
                    X_test, y_test = X_test.to(self.__device), y_test.to(self.__device)

                    y_pred_test = self.__student(X_test)

                    if y_pred_test.shape[-1] == 1:
                        y_test = y_test.unsqueeze(dim=1).float()

                    val_loss = loss_fn(y_pred_test, y_test)

                    loss = val_loss.item()
                    running_loss += loss

                    accuracy = (argmax(y_pred_test, dim=1) == y_test).sum().item() / len(y_test)
                    running_acc += accuracy

                    if record_validation_progress: 
                        self.__record_validation_step(loss, accuracy, epoch)
            
            return running_loss, running_acc


    def __record_training_step(self, loss, accuracy, batch, epoch):
        self.__train_progess['epoch'].append(epoch)
        self.__train_progess['batch'].append(batch)
        self.__train_progess['train_loss'].append(loss)
        self.__train_progess['train_acc'].append(accuracy)


    def __record_validation_step(self, loss, accuracy, epoch):
        self.__validation_progress['epoch'].append(epoch)
        self.__validation_progress['val_loss'].append(loss)
        self.__validation_progress['val_acc'].append(accuracy)


    def get_train_progress(self):
        # print(self.__train_progess)
        # import pandas as pd
        # df = pd.DataFrame(self.__train_progess)
        # print(df)
        return self.__train_progess
    
    def get_validation_progress(self):
        # print(self.__validation_progress)
        return self.__validation_progress


    def save_train_progress(self, filepath):
        with open(filepath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.__train_progess.keys())
            writer.writerows(zip(*self.__train_progess.values()))


    def save_model_dictionary(self, filepath, append_accuracy=False):
        dir = os.path.dirname(filepath)
        filename, ext = os.path.splitext(os.path.basename(filepath))

        if append_accuracy:
            if self.__testloader:
                accuracy = self.__train_progess['test_acc'][-1]
                filename += f"_test_acc:{accuracy:.4f}"
            else:
                accuracy = self.__train_progess['train_acc'][-1]
                filename += f"_train_acc:{accuracy:.4f}"

        FILEPATH = dir + "/" + filename + ext
        save_dict(self.__model.state_dict(), FILEPATH)


    def __start_timer(self):
        self.__start_time = time.time()

    def __stop_timer(self):
        self.__end_time = time.time()
        return self.__end_time - self.__start_time
