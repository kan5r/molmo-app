import re
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image

def resize_image(image, long_side_px=640):
    width, height = image.size
    
    if width > height:
        new_width = long_side_px
        new_height = int((long_side_px / width) * height)
    else:
        new_height = long_side_px
        new_width = int((long_side_px / height) * width)
    
    return image.resize((new_width, new_height))



def get_coords(output_string, image):
    """
    Function to get x, y coordinates given Molmo model outputs.

    :param output_string: Output from the Molmo model.
    :param image: Image in PIL format.

    Returns:
        coordinates: Coordinates in format of [(x, y), (x, y)]
    """
    image = np.array(image)
    h, w = image.shape[:2]
    
    if 'points' in output_string:
        matches = re.findall(r'(x\d+)="([\d.]+)" (y\d+)="([\d.]+)"', output_string)
        coordinates = [(int(float(x_val)/100*w), int(float(y_val)/100*h)) for _, x_val, _, y_val in matches]
    elif 'point' in output_string:
        match = re.search(r'x="([\d.]+)" y="([\d.]+)"', output_string)
        if match:
            coordinates = [(int(float(match.group(1))/100*w), int(float(match.group(2))/100*h))]
    else:
        return output_string
    
    return coordinates


def show_points(image, coords, marker_size=375):
    pos_points = coords

    dpi = plt.rcParams['figure.dpi']
    figsize = image.shape[1] / dpi, image.shape[0] / dpi

    plt.figure(figsize=figsize)
    plt.imshow(image)

    ax = plt.gca()
    ax.scatter(
        pos_points[:, 0], 
        pos_points[:, 1], 
        color="#dc5c99",
        marker='.', 
        s=marker_size, 
        # edgecolor='white', 
        linewidth=1.25
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    return plt


def plot_image(image):
    """
    Converts a PIL image to Matplotlib plot.

    :param image: A PIL image.
    """
    image = np.array(image)

    dpi = plt.rcParams['figure.dpi']
    figsize = image.shape[1] / dpi, image.shape[0] / dpi

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    return plt
