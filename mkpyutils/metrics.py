def calc_mean_std(dataloader):
    sum_, squared_sum, batches = 0,0,0
    for data, _ in dataloader:
        sum_ += torch.mean(data, dim = ([0,2,3]))
        squared_sum += torch.mean(data**2, dim = ([0,2,3]))
        batches += 1

    mean = sum_/batches
    std = (squared_sum/batches - mean**2)**0.5
    return mean,std
