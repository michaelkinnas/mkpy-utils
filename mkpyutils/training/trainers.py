from torch import manual_seed, inference_mode, argmax, cuda, backends
from torch import save as save_dict
from tqdm.auto import tqdm
from ..helpers.helpers import report_training, report_validation, report_last_training_step
from sklearn.metrics import accuracy_score
import time
import os

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_dataloader, validation_dataloader = None, lr_scheduler=None,
                 epochs: int=2, device='auto', seed=None, validation_step: int=1, checkpoint_step: int=0):
        """
        Trainer function. Takes a PyTorch ANN model and a train DataLoader and trains the model

        Args:
            model (nn module): The neural network model to be trained.
            optimizer (optim): An optimizer function.
            loss_fn (loss): A loss function.
            train_dataloader (Dataloader): A dataloader containing the training dataset.
            validation_dataloader (Dataloader, optional): A dataLoader containing the validation dataset, for testing accuracy
                after every training epoch. If no validation dataloader is provided the testing will be skipped. (default: ``None``)
            lr_scheduler (optim): A learning rate scheduler function (optional)
            epochs (int): The number of epochs for the model to be trained. (default: ``1``).
            device (str): The device to be trained on (default: ``AUTO``).
            seed (str): A seed for initializing model parameters. (default: ``None``).
            validation_inference_step (int): Report training (and validation, if valloader is not None) accuracy and loss after
                every number of epochs (default: ``1``).
            checkpoint_step (int): Save model weights every ``step`` epochs, if set to greater than 0.
        """
        self.__model = model
        self.__trainloader = train_dataloader
        self.__optimizer = optimizer
        self.__scheduler = lr_scheduler
        self.__loss_fn = loss_fn
        self.__epochs = epochs

        if device == 'auto':
            self.__device = 'cuda' if cuda.is_available() else 'cpu'
        else:
            self.__device = device

        self.__seed = seed
        self.__valloader = validation_dataloader
        self.__validation_step = validation_step
        self.__checkpoint_step = checkpoint_step

        self.__train_progress = {
                'epoch':[],
                'batch':[],
                'train_acc' : [],
                'train_loss': [],
                'timestamp': []
            }
        
        self.__validation_progress = {
                'epoch':[],
                'val_acc': [],
                'val_loss': [],
                'timestamp':[]
        }

        if self.__seed is not None:
            manual_seed(self.__seed)
            if self.__device == 'cuda':
                cuda.manual_seed(self.__seed)
                backends.cudnn.deterministic = True
                # When using CUDA >= 10.2 for reproducability you must use the folloing command
                # however it requred some additional environment variables to be set
                # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
                # use_deterministic_algorithms(True)
                backends.cudnn.benchmark = False

        self.__model.to(self.__device)

    def train_model(self, 
                    record_train_progress=False, 
                    record_validation_progress=False, 
                    verbose=True, 
                    window: int=100, 
                    use_tqdm=True, 
                    batch_reporting_step=1):
        
        if self.__seed:
            manual_seed(self.__seed)
        
        if use_tqdm:
            epoch_iterator = tqdm(range(self.__epochs), desc='Epoch: ', leave=False, position=0)
        else:
            epoch_iterator = range(self.__epochs)
        
        for epoch in epoch_iterator:
            running_loss = 0
            running_acc = 0

            self.__model.train()

            if self.__scheduler:
                lr = self.__scheduler.get_last_lr()[0]
            else:
                lr = self.__optimizer.param_groups[-1]['lr']

            if use_tqdm:
                batch_iterator = tqdm(self.__trainloader, total=len(self.__trainloader), desc="Batch: ", leave=False, position=1)
            else:
                batch_iterator = self.__trainloader

            for train_batch_idx, (X_train, y_train) in enumerate(batch_iterator):
                
                X_train, y_train = X_train.to(self.__device), y_train.to(self.__device)
                y_pred = self.__model(X_train)

                # In case of binary classification
                if y_pred.shape[-1] == 1:
                    y_train = y_train.unsqueeze(dim=1).float()
                train_loss = self.__loss_fn(y_pred, y_train)
                running_loss += train_loss.item()

                train_loss.backward()

                self.__optimizer.step()
                self.__optimizer.zero_grad()


                # Calculate batch accuracy
                batch_accuracy = accuracy_score(y_train.cpu(), argmax(y_pred, dim=1).cpu())
                running_acc += batch_accuracy

                # Calculate batch loss
                loss = train_loss.item()
                running_loss += loss

                
                if record_train_progress:
                    self.__record_training_step(epoch=epoch,
                                                batch=train_batch_idx,
                                                loss=loss, 
                                                accuracy=batch_accuracy)
                    
                # Repost batch training step if verbose is selected
                if verbose:
                    report_training(current_epoch=epoch,
                                    total_epochs=self.__epochs,
                                    acc = batch_accuracy,
                                    loss=loss,
                                    current_batch=train_batch_idx,
                                    total_batches=len(self.__trainloader),
                                    learning_rate=lr,
                                    use_tqdm=use_tqdm,
                                    reporting_step=batch_reporting_step,
                                    batch_iterator=batch_iterator)
                    
            # END OF EPOCH------------------------
            # Save snapshot
            if self.__checkpoint_step > 0 and self.__checkpoint_step % epoch+1 == 0:
                self.save_model_dictionary(f'./weights_ep:{epoch+1}.pth', append_accuracy=True)
            
            # If a test dataset is provided perform validation step and make an epoch report
            if verbose:
                if self.__valloader is not None:
                    if epoch % self.__validation_step == 0 and epoch != self.__epochs-1:
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

        
        if self.__valloader:
            val_loss, val_acc = self.__validation(epoch=None, 
                                                    record_validation_progress=False, 
                                                    use_tqdm=False)
            
            report_validation(acc=val_acc,
                            loss=val_loss,
                            current_epoch=self.__epochs-1,
                            total_epochs=self.__epochs,
                            use_tqdm=False,
                            # acc_rate=diff(acc_hist, 1, prepend=0),
                            # loss_rate=diff(loss_hist, 1, prepend=0),
                            epoch_iterator=None)


    def __validation(self, epoch, record_validation_progress, use_tqdm):
        running_accuracy = 0
        running_loss = 0

        self.__model.eval()

        with inference_mode():
            if use_tqdm:
                vallidation_batch_iterator = tqdm(self.__valloader, total=len(self.__valloader), desc="Validation: ", leave=False, position=2)
            else:
                vallidation_batch_iterator = self.__valloader

            for X_test, y_test in vallidation_batch_iterator:
                X_test, y_test = X_test.to(self.__device), y_test.to(self.__device)

                y_pred_test = self.__model(X_test)

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


    def __record_training_step(self, epoch, batch, loss, accuracy):
        self.__train_progress['epoch'].append(epoch)
        self.__train_progress['batch'].append(batch)
        self.__train_progress['train_loss'].append(loss)
        self.__train_progress['train_acc'].append(accuracy)
        self.__train_progress['timestamp'].append(time.time())


    def __record_validation_step(self, epoch, loss, accuracy):
        self.__validation_progress['epoch'].append(epoch)
        self.__validation_progress['val_loss'].append(loss)
        self.__validation_progress['val_acc'].append(accuracy)
        self.__validation_progress['timestamp'].append(time.time())


    def get_train_progress(self):
        return self.__train_progress
    
    def get_validation_progress(self):
        return self.__validation_progress

    def save_model_dictionary(self, filepath, append_accuracy=False):
        directory = os.path.dirname(filepath)
        filename, ext = os.path.splitext(os.path.basename(filepath))

        if append_accuracy:
            if self.__testloader:
                accuracy = self.__train_progess['test_acc'][-1]
                filename += f"_test_acc:{accuracy:.4f}"
            else:
                accuracy = self.__train_progess['train_acc'][-1]
                filename += f"_train_acc:{accuracy:.4f}"

        FILEPATH = directory + "/" + filename + ext
        save_dict(self.__model.state_dict(), FILEPATH)
