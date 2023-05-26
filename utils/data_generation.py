from PIL import Image, ImageFilter, ImageOps
import random
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def data_generation(path_image=None, destination_folder=None, textures_folder=None, threshold=0.3, gaussian_noise=True, black_and_white=False, sepia=False, k=0.5):

    '''
    Args:
        - path_image: path of the image on which apply the texture
        - destination_folder: output folder where results will be saved
        - texturs_folder: path to textures
        - threshold: level of gray of each pixel from the texture. Under this level the pixel will be ignored
        - gaussian_noise: apply or not gaussian noise on image (True, False).
        - black_and_white: apply the black and white filter on the image (True, False)
        - sepia: apply the sepia filter on the image (True, False)
        - k: intensity of the sepia filter (0,1).
    '''
        
    if path_image == None:
        print("Error: the image path can't be None")
        return
    if destination_folder == None:
        print("Error: destination folder path can't be None")
        return
    if k>1 or k<0:
        print("Error: k must be in (0,1)")
        return
    
    # The limit of gray for each pixel that we take from the texture
    threshold_rgb = (int(threshold*255), int(threshold*255), int(threshold*255))

    # Load the image
    input_image = Image.open(path_image)

    # Resize the image manteining the aspect ratio    
    if input_image.width>input_image.height:        # L'immagine è più larga che alta -> width viene portato a 512 pixel
        aspect_ratio = input_image.width/input_image.height
        new_width = 512
        new_height = int(new_width / aspect_ratio)
    elif input_image.height>input_image.width:      # L'immagine è più alto che larga -> heigth viene portato a 512 pixel
        aspect_ratio = input_image.height/input_image.width
        new_height = 512
        new_width = int(new_height / aspect_ratio)
    else:                                           # L'immagine è quadrata -> width e heigth vengono portati a 512 pixel
        new_height = 512
        new_width = 512

    input_image = input_image.resize((new_width, new_height))


    # Get a random texture, open it, and resize it
    random_texture = random.choice(os.listdir(textures_folder))
    texture_jpg = Image.open(os.path.join(textures_folder, random_texture))
    texture_jpg = texture_jpg.resize((new_width, new_height)) # Resize the texture with the same size of the input image

    # Convert the image to RGBA format
    texture_rgba = texture_jpg.convert("RGBA")

    # Create a new image with the same dimensions as the texture image jpg
    texture_png = Image.new("RGBA", texture_rgba.size)

    '''
    # Iterate over each pixel in the image
    for x in range(texture_rgba.width):
        for y in range(texture_rgba.height):
            # Get the pixel value at (x, y)
            pixel = texture_rgba.getpixel((x, y))
    
            # Check if the pixel has all values > 100 and equals each other (grayscale) 
            if pixel[:3] >= threshold_rgb:
                # Set the pixel to white with full opacity
                texture_png.putpixel((x, y), (pixel[0], pixel[1], pixel[2], 255))
            else:
                # Set the pixel to fully transparent
                texture_png.putpixel((x, y), (0, 0, 0, 0))
    '''

    # Create a new image with the same dimensions as the RGB image
    modified_image = Image.new("RGBA", input_image.size)

    # Paste the RGB image onto the new image
    modified_image.paste(input_image, (0, 0))


    # Apply black and white filter
    if black_and_white:
        bw_filter = np.array([[0.2989, 0.5870, 0.1140], # B 
                              [0.2989, 0.5870, 0.1140], # G
                              [0.2989, 0.5870, 0.1140]]) # R
        
        modified_image = cv2.cvtColor(np.array(modified_image), cv2.COLOR_RGB2BGR)
        modified_image = cv2.transform(modified_image, bw_filter)
        modified_image[np.where(modified_image>255)] = 255
        modified_image =  Image.fromarray(modified_image)


    # Apply sepia filter
    if sepia:
        sepia_filter = np.array([[ 0.393 + 0.607 * (1 - k), 0.769 - 0.769 * (1 - k), 0.189 - 0.189 * (1 - k)], # B
                                 [ 0.349 - 0.349 * (1 - k), 0.686 + 0.314 * (1 - k), 0.168 - 0.168 * (1 - k)], # G
                                 [ 0.272 - 0.349 * (1 - k), 0.534 - 0.534* (1 - k), 0.131 + 0.869 * (1 - k)]]) # R
        
        modified_image = cv2.cvtColor(np.array(modified_image), cv2.COLOR_RGB2BGR)
        modified_image = cv2.transform(modified_image, sepia_filter)
        modified_image[np.where(modified_image>255)] = 255
        modified_image =  Image.fromarray(modified_image)


    # Apply gaussian noise
    if gaussian_noise:
        modified_image = modified_image.filter(ImageFilter.GaussianBlur(0.6))


    # Paste the texture image with transparency onto the new image
    modified_image.paste(texture_png, (0, 0), mask=texture_png)


    # Save the resulting stacked image
    path_image_list = path_image.split('/')
    image_name = path_image_list.pop().split('.')[0]
    
    modified_image.save(f"{destination_folder}/{image_name}.png")



def get_image_files(folder_path):
    image_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(folder_path, file_name))
    return image_files



if __name__ == '__main__':

    dataset_path = '/home/luigi/university/vision_perception/Colorization-SuperResolution/data/test/'
    destination_folder = '/home/luigi/university/vision_perception/Colorization-SuperResolution/data/test/modified/'
    textures_folder = '/home/luigi/university/vision_perception/Colorization-SuperResolution/data/textures/'

    image_files = get_image_files(dataset_path)

    textures = ['dust', 'textures']

    for image in image_files:
        random_texture = random.randint(0,1)  
        random_filter = random.randint(0,2)

        texture = textures[random_texture]
        if random_filter == 0:
            data_generation(path_image=image, destination_folder=destination_folder, textures_folder=textures_folder)
        elif random_filter == 1:
            data_generation(path_image=image, destination_folder=destination_folder, textures_folder=textures_folder, black_and_white=True)
        elif random_filter == 2:
            data_generation(path_image=image, destination_folder=destination_folder, textures_folder=textures_folder, sepia=True)
