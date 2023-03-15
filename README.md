# [NTIRE 2023 Challenge on Image Denoising](https://cvlai.net/ntire/2023/) @ [CVPR 2023](https://cvpr2023.thecvf.com/)

## How to test the baseline model?

1. `git clone https://github.com/ofsoundof/NTIRE2023_Dn50.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
   
## How to add your model to this baseline?
1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1XVa8LIaAURYpPvMf7i-_Yqlzh-JsboG0hvcnp-oI9rs/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
This repository shows how to add noise to synthesize the noisy image. It also shows how you can save an image.

```python
import numpy as np
import imageio


def add_noise(image, sigma=50):
    """
    image: input image, numpy array, dtype=uint8, range=[0, 255]
    sigma: default 50
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, sigma / 255, image.shape)
    gauss_noise = image + noise
    return gauss_noise * 255


def save_image(image, path):
    """
    image: saved image, numpy array, dtype=float
    path: saving path
    """
    # The type of the image is float, and range of the image might not be in [0, 255]
    # Thus, before saving the image, the image needs to be clipped.
    image = np.round(np.clip(image, 0, 255)).astype(np.uint8)
    imageio.imwrite(path, image)

    
def crop_image(image, s=8):
    h, w, c = image.shape
    image = image[:h - h % s, :w - w % s, :]
    return image


image_name = "example.png"

img = imageio.imread(image_name)

# The provided noisy validation images in the following link are cropped such that the width and height are multiples of 8.
# https://drive.google.com/file/d/1iYurwSVBUxoN6fQwUGP-UbZkTZkippGx/view?usp=share_link
# You can achieve that using the function crop_image.
img = crop_image(img)

# This image is used as the noisy image for training.
img_noise = add_noise(img, sigma=50)

# This function ensures that the image is properly clipped and rounded before saving.
save_image(img_noise, "example_saved.png")

```