from torch.nn import KLDivLoss
from torch.nn.functional import softmax, log_softmax
from torch import save as save_dict
from torch import manual_seed, inference_mode, argmax, cuda, backends
from tqdm.auto import tqdm
from .helpers import report_distillation_train, report_validation, report_last_training_step
from sklearn.metrics import accuracy_score
import time
import os


class ResponseDistiller:
    def __init__(self, teacher, student, loss_fn, optimizer,  train_dataloader, validation_dataloader=None, lr_scheduler=None,
                 epochs: int=2, alpha: float=0.8, temperature: float=1, device='auto', seed=None, validation_step: int=1, checkpoint_step: int=0):
        """
        A training loop for the Response Based Distillation method. It receives a teacher and student neural network model and performs
        the knowledge distillation.

        Args:
            teacher (nn module): The trained neural network teacher model.
            student (nn modeul): The untrained neural network student model.
            loss_fn (loss): A loss function that will be used for the hard target loss calculatin of the student.
            optimizer (optim): An optimizer function.
            train_dataloader (Dataloader): A dataloader containing the training dataset.
            validation_dataloader (Dataloader, optional): A dataLoader containing the validation dataset, for testing accuracy
                after every training epoch. If no validation dataloader is provided the testing will be skipped. (default: ``None``)
            lr_scheduler (optim): A learning rate scheduler function (optional)
            epochs (int): The number of epochs for the model to be trained. (default: ``1``).
            alpha (float): The alpha parameter of the final loss calculation. Higher values favor the soft target loss moew, the student's hard target loss less and vice versa. (default: ``0.8``)
            temperature (float): The temperature that will be used to soften the probability distributions of both models. (default: ``1``).
            device (str): The device to be trained on (default: ``AUTO``).
            seed (str): A seed for initializing model parameters. (default: ``None``).
            validation_step (int): Report training (and validation, if valloader is not None) accuracy and loss after every number of epochs (default: ``1``).
            checkpoint_step (int): Save model weights every ``step`` epochs, if set to greater than 0.
        """
        self.__teacher = teacher
        self.__student = student

        self.__trainloader = train_dataloader
        self.__optimizer = optimizer  #for student
        self.__scheduler = lr_scheduler #for student
        self.__loss_fn = loss_fn
        self.__epochs = epochs

        self.__alpha = alpha #the weights given to the final loss function hard targetrs and soft targets coefficients
        self.__temperature = temperature #the temperature to soften the soft target vector values
        # self.__final_loss_temp = final_loss_temp

        if device == 'auto':
            self.__device = 'cuda' if cuda.is_available() else 'cpu'
        else:
            self.__device = device

        self.__seed = seed
        self.__valloader = validation_dataloader
        self.__validation_step = validation_step #steps in epochs
        self.__checkpoint_step = checkpoint_step #steps in epochs

        self.__train_progress = {
                'epoch':[],
                'batch':[],
                'acc' : [],
                'hard_loss': [],
                'soft_loss':[],
                'final_loss':[],
                'learning_rate':[],
                'timestamp':[],
            }

        self.__validation_progress = {
                'epoch':[],
                'acc': [],
                'loss': [],
                'timestamp':[]
        }

        if self.__seed is not None:
            manual_seed(self.__seed)
            if self.__device == 'cuda':
                cuda.manual_seed(self.__seed)
                backends.cudnn.deterministic = True
                # When using CUDA >= 10.2 for reproducability you must use the following command
                # however it requres some additional environment variables to be set
                # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
                # use_deterministic_algorithms(True)
                backends.cudnn.benchmark = False

        self.__teacher.to(self.__device)
        self.__student.to(self.__device)


    def distill(self,
                record_train_progress=False,
                record_validation_progress=False,
                verbose=True,
                use_tqdm=True,
                batch_reporting_step=1,
                ):

        soft_loss = KLDivLoss(reduction='batchmean')

        if verbose and use_tqdm:
            epoch_iterator = tqdm(range(self.__epochs), desc='Epoch: ', leave=False, position=0)
        else:
            epoch_iterator = range(self.__epochs)

        for epoch in epoch_iterator:
            running_loss = 0
            running_acc = 0

            self.__student.train()

            if self.__scheduler:
                lr = self.__scheduler.get_last_lr()[0]
            else:
                lr = self.__optimizer.param_groups[-1]['lr']

            if verbose and use_tqdm:
                batch_iterator = tqdm(self.__trainloader, total=len(self.__trainloader), desc="Batch: ", leave=False, position=1)
            else:
                batch_iterator = self.__trainloader

            for train_batch_idx, (X_train, y_train) in enumerate(batch_iterator):
                X_train, y_train = X_train.to(self.__device), y_train.to(self.__device)

                with inference_mode():
                    teacher_logits = self.__teacher(X_train)

                student_logits = self.__student(X_train)

                #needs logSoftmax for teacher logits since the kldivloss function requires the log value for the first input
                student_probs = log_softmax(student_logits / self.__temperature, dim = 1)
                teacher_probs = softmax(teacher_logits / self.__temperature, dim = 1)

                # Multiplying by T^2 is suggested in the paper https://arxiv.org/abs/1503.02531
                # The scaling factor accounts for the gradient scaling introduced by the temperature above
                L_soft = soft_loss(student_probs, teacher_probs) * self.__temperature ** 2
                L_hard = self.__loss_fn(student_logits, y_train)

                L_final = self.__final_loss(L_soft, L_hard)

                L_final.backward()
                self.__optimizer.step()
                self.__optimizer.zero_grad()

                running_loss += L_final.item()

                batch_accuracy = accuracy_score(y_train.cpu(), argmax(student_logits, dim=1).cpu())
                running_acc += batch_accuracy

                # Report train progress to tqdm
                report_distillation_train(current_epoch=epoch,
                                    total_epochs=self.__epochs,
                                    current_batch=train_batch_idx,
                                    total_batches=len(self.__trainloader),
                                    soft_loss=L_soft,
                                    hard_loss=L_hard,
                                    final_loss=L_final,
                                    learning_rate=lr,
                                    student_acc=batch_accuracy,
                                    verbose=verbose,
                                    use_tqdm=use_tqdm,
                                    reporting_step=batch_reporting_step,
                                    tqdm_batch_iterator=batch_iterator)               


                if record_train_progress:
                    self.__record_training_step(epoch=epoch,
                                                batch=train_batch_idx,
                                                hard_loss=L_hard.item(),
                                                soft_loss=L_soft.item(),
                                                final_loss=L_final.item(),
                                                learning_rate=lr,
                                                accuracy = batch_accuracy)

            # END OF EPOCH --------
            # Save checkpoint
            if self.__checkpoint_step > 0 and self.__checkpoint_step % epoch+1 == 0:
                self.save_model_weights(f'./weights_ep:{epoch+1}.pth', append_accuracy=True)

            
            # If a test dataset is provided perform validation step and make an epoch report
            if self.__valloader is not None:
                if epoch % self.__validation_step == 0 or epoch == self.__epochs-1:
                    val_loss, val_acc = self.__validation(epoch=epoch, 
                                                            record_validation_progress=record_validation_progress, 
                                                            use_tqdm=use_tqdm)
                    
                    report_validation(acc=val_acc,
                                      loss=val_loss,
                                      current_epoch=epoch,
                                      total_epochs=self.__epochs,
                                      use_tqdm=use_tqdm,
                                      epoch_iterator=epoch_iterator)
            

            # If no test validation is provided report last training step matrics to epoch report
            else:
                report_last_training_step(acc =running_acc / len(self.__trainloader), 
                                          loss=running_loss / len(self.__trainloader),
                                          epoch=epoch,
                                          use_tqdm=use_tqdm,
                                          epoch_iterator=epoch_iterator)

                    
            if self.__scheduler:
                self.__scheduler.step()

        # ENF OF TRAINING -----


    
    def __final_loss(self, L_soft, L_hard):
        return self.__alpha * L_soft + (1 - self.__alpha) * L_hard


    def __validation(self, epoch, record_validation_progress, use_tqdm=True):        
        running_accuracy = 0
        running_loss = 0

        self.__student.eval()

        with inference_mode():
            if use_tqdm:
                vallidation_batch_iterator = tqdm(self.__valloader, total=len(self.__valloader), desc="Validation: ", leave=False, position=2)
            else:
                vallidation_batch_iterator = self.__valloader

            for X_test, y_test in vallidation_batch_iterator:
                X_test, y_test = X_test.to(self.__device), y_test.to(self.__device)

                y_pred_test = self.__student(X_test)

                if y_pred_test.shape[-1] == 1:
                    y_test = y_test.unsqueeze(dim=1).float()

                val_loss = self.__loss_fn(y_pred_test, y_test)

                loss = val_loss.item()
                running_loss += loss

                accuracy = accuracy_score(y_test.cpu(), argmax(y_pred_test, dim=1).cpu())
                running_accuracy += accuracy               

        if record_validation_progress:
            self.__record_validation_step(epoch, loss, accuracy)

        return loss / len(self.__valloader), running_accuracy / len(self.__valloader)


    def __record_training_step(self, epoch, batch, hard_loss, soft_loss, final_loss, learning_rate, accuracy):
        self.__train_progress['epoch'].append(epoch)
        self.__train_progress['batch'].append(batch)
        self.__train_progress['hard_loss'].append(hard_loss)
        self.__train_progress['soft_loss'].append(soft_loss)
        self.__train_progress['final_loss'].append(final_loss)
        self.__train_progress['acc'].append(accuracy)
        self.__train_progress['learning_rate'].append(learning_rate)
        self.__train_progress['timestamp'].append(time.time())


    def __record_validation_step(self, epoch, loss, accuracy,):
        self.__validation_progress['epoch'].append(epoch)
        self.__validation_progress['loss'].append(loss)
        self.__validation_progress['acc'].append(accuracy)
        self.__validation_progress['timestamp'].append(time.time())


    def get_train_progress(self):
        return self.__train_progress

    def get_validation_progress(self):
        return self.__validation_progress

    def save_model_weights(self, filepath, append_accuracy=False):
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
