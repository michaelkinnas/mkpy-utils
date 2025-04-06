from numpy import concatenate, mean, std

def mean_std(dataset) -> tuple[float | tuple[float, float, float], float | tuple[float, float, float]]:
    '''
    Calculate mean and std from a PyTorch dataset class.
    The dataset must contain PIL images with pixel values from range 0-255
    '''    
    rgb_values = concatenate([x[0].getdata() for x in dataset], axis=0) / 255
    mean_rgb = mean(rgb_values, axis=0)
    std_rgb = std(rgb_values, axis=0)
    return mean_rgb, std_rgb
