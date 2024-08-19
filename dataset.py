import os
import sys
from pycocotools.coco import COCO
#------------------
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

def initialize_coco_api(data_dir, data_type):
    """
    Initializes the COCO API for instance and caption annotations.

    Parameters:
    - data_dir: str, path to the COCO dataset directory.
    - data_type: str, type of dataset (e.g., 'val2014').

    Returns:
    - coco: COCO object for instance annotations.
    - coco_caps: COCO object for caption annotations.
    """
    # Append COCO API path
    sys.path.append('/Users/shreykamoji/Image_Captioning_final/data')

    # Initialize COCO API for instance annotations
    instances_ann_file = os.path.join(data_dir, f'annotations/instances_{data_type}.json')
    coco = COCO(instances_ann_file)

    # Initialize COCO API for caption annotations
    captions_ann_file = os.path.join(data_dir, f'annotations/captions_{data_type}.json')
    coco_caps = COCO(captions_ann_file)

    return coco, coco_caps


def get_image_ids(coco):
    """
    Retrieves a list of image IDs from the COCO annotations.

    Parameters:
    - coco: COCO object, initialized for instance annotations.

    Returns:
    - ids: list of int, containing image IDs.
    """
    ids = list(coco.anns.keys())
    return ids

# Initialize COCO API
data_dir = '/Users/shreykamoji/Image_Captioning_final/data'
data_type = 'val2014'
coco, coco_caps = initialize_coco_api(data_dir, data_type)
print(coco)
print(coco_caps)

# Get image IDs
image_ids = get_image_ids(coco)
print(image_ids[:10])

#-------------------------------------------------------------------------------------
def get_random_image(coco):
    """
    Selects a random image from the COCO dataset.

    Parameters:
    - coco: COCO object, initialized for instance annotations.

    Returns:
    - img: dict, metadata of the selected image.
    - url: str, URL of the image.
    """
    ann_id = np.random.choice(list(coco.anns.keys()))
    img_id = coco.anns[ann_id]['image_id']
    img = coco.loadImgs(img_id)[0]
    url = img['coco_url']
    return img, url


def plot_image(url):
    """
    Plots the image from the given URL.

    Parameters:
    - url: str, URL of the image to be plotted.
    """
    I = io.imread(url)
    plt.axis('off')
    plt.imshow(I)
    plt.show()


def display_captions(coco_caps, img_id):
    """
    Displays the captions associated with the selected image.

    Parameters:
    - coco_caps: COCO object, initialized for caption annotations.
    - img_id: int, ID of the image for which captions are to be displayed.
    """
    ann_ids = coco_caps.getAnnIds(imgIds=img_id)
    anns = coco_caps.loadAnns(ann_ids)
    coco_caps.showAnns(anns)

#-------------------------------------------------------------------------------------

# Get a random image
img, url = get_random_image(coco)

# Print URL and visualize the image
print(url)
plot_image(url)

# Display corresponding captions
display_captions(coco_caps, img['id'])

# Explanation:

# get_random_image: Fetches a random image from the dataset.
# plot_image: Visualizes the image using the URL.
# display_captions: Shows the captions associated with the image.
    