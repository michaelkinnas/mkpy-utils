from torchvision import utils
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Extract and show or save feature maps
def extract_feature_maps(net, save_path: str):
    cnn_layers = [x for x in net.modules() if (isinstance(x, nn.Conv2d))]
    for i, layer in enumerate(cnn_layers):
        parameter = layer.weight.data
        allkernels = True
        ch = 0
        padding = 1
        b,c,w,h = parameter.shape
        nrow = c
        if allkernels: parameter = parameter.view(b*c, -1, w, h)
        elif c != 3: parameter = parameter[:, ch, :, :].unsqueeze(dim=1)
        # rows = np.min((parameter.shape[0] // nrow + 1, 64)) 
        grid = utils.make_grid(parameter, nrow=nrow, normalize=True, padding=padding)
        plt.figure(figsize=(c, b))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.ylabel("Kernels")
        plt.xlabel("Kernel depth (channels)")
        plt.title(layer)        
        plt.yticks(ticks = [x * (h+1) + 2 for x in range(b)], labels=[x + 1 for x in range(b)])
        plt.xticks(ticks = [x * (w+1) + 2 for x in range(c)], labels=[x + 1 for x in range(c)])
        plt.savefig(os.path.join(save_path, f'{layer}_{i}.png'), bbox_inches='tight')
        plt.close()

