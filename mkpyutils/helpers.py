def report_distillation_train(current_epoch, total_epochs, current_batch, total_batches, soft_loss, hard_loss, final_loss, learning_rate, student_acc, verbose, use_tqdm, reporting_step, tqdm_batch_iterator=None):
    if verbose:
        if use_tqdm:
            tqdm_batch_iterator.set_postfix_str(f"Soft target loss: {soft_loss:.4f}, Hard target loss: {hard_loss:.4f}, Final loss: {final_loss:.4f}, Student acc: {student_acc:.2f}%, LR: {learning_rate}")
        else:
            if current_batch % reporting_step == 0:
                print(f"Epoch [{current_epoch+1} / {total_epochs}], Batch [{current_batch+1} / {total_batches}], Soft target loss: {soft_loss:.4f}, Hard target loss: {hard_loss:.4f}, Final loss: {final_loss:.4f}, Student acc: {student_acc:.2f}%, LR: {learning_rate}")



def report_distillation_validation(validation_dataloader, validation_fn, current_epoch, total_epochs, current_batch, train_batch_size, length_y, record_validation_progress, use_tqdm, verbose, tqdm_epoch_iterator, running_loss, running_acc):
    if validation_dataloader:
        val_loss, val_acc = validation_fn(epoch=current_epoch,
                                            record_validation_progress=record_validation_progress,
                                            use_tqdm=use_tqdm)
        if verbose:
            if use_tqdm:
                tqdm_epoch_iterator.set_postfix_str(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc * 100:.2f}%")
            else:
                print(f"  => Epoch [{current_epoch+1} / {total_epochs}], Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc * 100:.2f}%")

    else:
        if verbose:
            if use_tqdm:
                tqdm_epoch_iterator.set_postfix_str(f"Previous epoch: Training loss: {running_loss / (current_batch+1):.4f} | Training accuracy: {running_acc / (train_batch_size * (current_batch) + length_y) * 100:.2f}%")
            else:
                print(f"  => Epoch [{current_epoch+1} / {total_epochs}], Previous epoch: Training loss: {running_loss / (current_batch+1):.4f} | Training accuracy: {running_acc / (train_batch_size * (current_batch) + length_y) * 100:.2f}%")