from torch import manual_seed, inference_mode, argmax
from tqdm.auto import tqdm
import csv
import time
import os
from torch import save as save_dict

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_dataloader, scheduler=None,
                 epochs=1, device='cpu', seed=None, validation_dataloader = None, validation_inference_epoch_step=1):
        """
        Trainer function. Takes a PyTorch ANN model and a train DataLoader and trains the model

        Args:
            model       (nn module): The neural network model to be trained.
            train_dataloader (Dataloader): A dataloader containing the training dataset.
            optimizer (optim): An optimizer function.
            scheduler (optim): A scheduler function (optional)
            loss_fn (loss): A loss function.
            epochs (int): The number of epochs for the model to be trained. (default: ``1``).
            device (str): The device to be trained on (default: ``CPU``).
            seed (str): A seed for initializing model parameters. (default: ``None``).
            validation_dataloader (Dataloader, optional): A dataLoader containing the validation dataset, for testing accuracy
                after every training epoch. If no validation dataloader is provided the testing will be skipped. (default: ``None``)
            validation_inference_step (int): Report training (and validation, if valloader is not None) accuracy and loss after
                every number of epochs (default: ``1``).
        """

        self.__model = model
        self.__trainloader = train_dataloader
        self.__optimizer = optimizer
        self.__scheduler = scheduler
        self.__loss_fn = loss_fn
        self.__epochs = epochs
        self.__device = device
        self.__seed = seed
        self.__valloader = validation_dataloader
        self.__validation_inference_epoch_step = validation_inference_epoch_step

        self.__train_progess = {
                'batch':[],
                'train_acc' : [],
                'train_loss': [], 
            }
        
        self.__validation_progress = {
                'epoch':[],
                'val_acc': [],
                'val_loss': [],
        }

        self.__model = self.__model.to(self.__device)

    def train_model(self, verbose=False):
        if self.__seed:
            manual_seed(self.__seed)

        # self.__model.train()

        TRAIN_NUM_BATCHES = len(self.__trainloader)
        TRAIN_BATCH_SIZE = self.__trainloader.batch_size

        if self.__valloader:
            VAL_NUM_BATCHES = len(self.__valloader)
            VAL_BATCH_SIZE = self.__valloader.batch_size

        self.__start_timer()

        epoch_iterator = tqdm(range(self.__epochs), desc='Epoch: ', leave=False, position=0)
        
        for epoch in epoch_iterator:
            running_loss = 0
            running_acc = 0

            self.__model.train()

            batch_iterator = tqdm(self.__trainloader, total=len(self.__trainloader), desc="Batch: ", leave=False, position=1)            
            for batch_idx, (X_train, y_train) in enumerate(batch_iterator):

                X_train, y_train = X_train.to(self.__device), y_train.to(self.__device)
                y_pred = self.__model(X_train)

                # In case of binary classification
                if y_pred.shape[-1] == 1:
                    y_train = y_train.unsqueeze(dim=1).float()

                train_loss = self.__loss_fn(y_pred, y_train)
                train_loss.backward()

                self.__optimizer.step()
                self.__optimizer.zero_grad()

                # Calculate prediction accuracy
                running_acc += (argmax(y_pred, dim=1) == y_train).sum().item()
                running_loss += train_loss.item()

                # Report train progress to tqdm
                batch_iterator.set_postfix_str(f"Train loss: {running_loss / (batch_idx+1):.4f} | Train accuracy: {running_acc / (TRAIN_BATCH_SIZE * (batch_idx+1)) * 100:.2f}%")

            if self.__valloader:
                if epoch % self.__validation_inference_epoch_step == 0 or epoch == self.__epochs-1:
                    running_acc = 0
                    running_loss = 0

                    self.__model.eval()
                    with inference_mode():

                        vallidation_batch_iterator = tqdm(self.__valloader, total=len(self.__valloader), desc="Validation: ", leave=False, position=2)

                        for batch, (X_test, y_test) in enumerate(vallidation_batch_iterator):
                            X_test, y_test = X_test.to(self.__device), y_test.to(self.__device)

                            y_pred_test = self.__model(X_test)

                            if y_pred_test.shape[-1] == 1:
                                y_test = y_test.unsqueeze(dim=1).float()

                            val_loss = self.__loss_fn(y_pred_test, y_test)
                            running_loss += val_loss.item()

                            running_acc += (argmax(y_pred_test, dim=1) == y_test).sum().item() / len(y_test)

            
            epoch_iterator.set_postfix_str(f"Validation loss: {running_loss / VAL_NUM_BATCHES:.4f} | Validation accuracy: {running_acc / len(self.__valloader) * 100:.2f}%")

            if self.__scheduler:
                # print('Step called')
                self.__scheduler.step()



    # def __save_training_step(self, loss, accuracy, batch):
    #     self.__train_progess['batch'].append(batch)
    #     self.__train_progess['train_loss'].append(loss)
    #     self.__train_progess['train_acc'].append(accuracy)


    def __record_validation_step(self, loss, accuracy, epoch):
        self.__validation_progress['epoch'].append(epoch)
        self.__validation_progress['val_loss'].append(loss)
        self.__validation_progress['val_acc'].append(accuracy)


    def get_train_progress(self):
        return self.__train_progess


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
