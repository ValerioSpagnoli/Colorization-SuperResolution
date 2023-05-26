# Useful function to plot "n" random samples in a dataset

import matplotlib.pyplot as plt
import random
import numpy as np

def plot_random_samples(n, dataset):   

    '''
    Plot n samples from dataset.\n
    Args:
        - n: number of samples to plot
        - dataset: a PyTorch dataset
    '''

    # Create a figure with a (n, 3) grid of subplots
    # 3 because we will print the tuple (original, reference, restored)
    fig, axs = plt.subplots(n, 3)

    # Enumerate random indices of dataset elements and print them
    for i, j in enumerate(random.sample(range(0, len(dataset)), n)):
        plt.subplot(n,3,3*i+1)
        imshow(img=dataset[j][0], title=f'Original {j}')
        plt.axis('off')

        plt.subplot(n,3,3*i+2)
        imshow(img=dataset[j][1], title=f'Reference {j}')
        plt.axis('off') 

        plt.subplot(n,3,3*i+3)
        imshow(img=dataset[j][2], title=f'Restored {j}')
        plt.axis('off') 

    plt.show()


def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)