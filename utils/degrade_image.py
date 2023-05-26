import random
import os
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter

import numpy as np

random.seed(0)

BLUR_RADIUS = 0.5


def add_gaussian_noise(image, mean, std_dev):
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Generate Gaussian noise with the same shape as the input image
    noise = np.random.normal(mean, std_dev, image_array.shape)
    
    # Add the noise to the image array
    noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    
    # Convert the noisy image array back to an Image object
    noisy_image = Image.fromarray(noisy_image_array)
    
    return noisy_image



def degrade_image(img=None, textures_folder=None, degradation_level=3, scaling_factor=4):   

    '''
    Args:
        - img: path of the image 
        - texture_folder: path of the folder where are saved the textures
        - degradation_level: can be a positive number between (1, num of texture in texture_folder). 
        - scaling_factor: how much scale the image (>1)
    '''

    # Open image, convert it to greyscale
    image = Image.open(img).convert('L')


    # UNSTRUCTURED DEGRADATION: overall image quality degradation

    # Resize the image w.r.t. scaling factor
    resized_img = image.reduce(scaling_factor)

    # Add gaussian noise
    noise_img = add_gaussian_noise(resized_img, 1, 10)

    # Blur image
    blur_img = noise_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    # Apply BoxBlur filter
    res_img = blur_img.filter(ImageFilter.BoxBlur(radius=BLUR_RADIUS))


    ## STRUCTURED DEGRADATION: adding textures

    # Select a number of "degradation_level" random texture
    random_textures = random.sample(os.listdir(textures_folder), degradation_level)

    # Iterate over the textures
    for i, texture in enumerate(random_textures):
        # Open the texture, convert to greyscale, and resize it
        texture_img = Image.open(os.path.join(textures_folder, texture)).convert('L').resize(res_img.size)

        # Blend the transformed image with the old photo texture
        res_img = Image.composite(res_img, texture_img, ImageOps.invert(texture_img))
        # res_img = Image.blend(res_img, texture_img, alpha=0.5/(i+1))

    return res_img
