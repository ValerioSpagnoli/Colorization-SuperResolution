# Useful function to plot "n" random samples in a dataset

import matplotlib.pyplot as plt
import random

def plot_random_samples(n, dataset):      
    # Create a figure with a (n, 3) grid of subplots
    # 3 because we will print the tuple (original, reference, restored)
    fig, axs = plt.subplots(n, 3)

    # Enumerate random indices of dataset elements and print them
    for i, j in enumerate(random.sample(range(0, len(dataset)), n)):
        plt.subplot(n,3,3*i+1)
        plt.imshow(dataset[j][0].numpy().transpose(1, 2, 0))
        plt.title(f'Original {j}')
        plt.axis('off')

        plt.subplot(n,3,3*i+2)
        plt.imshow(dataset[j][1].numpy().transpose(1, 2, 0))
        plt.title(f'Reference {j}')
        plt.axis('off') 

        plt.subplot(n,3,3*i+3)
        plt.imshow(dataset[j][2].numpy().transpose(1, 2, 0))
        plt.title(f'Restored {j}')
        plt.axis('off') 

    plt.show()