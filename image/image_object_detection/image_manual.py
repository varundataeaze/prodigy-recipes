import prodigy
from prodigy.components.loaders import Images
from prodigy.util import split_string
from typing import List, Optional
import io
import copy
import numpy as np 
from PIL import Image
import yaml 
from prodigy.util import log, b64_uri_to_bytes, split_string
from detect_model import get_prediction

def preprocess_pil_image(pil_img, color_mode='rgb', target_size=None):
    """Preprocesses the PIL image
    Arguments
        img: PIL Image
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    Returns
        Preprocessed PIL image
    """
    if color_mode == 'grayscale':
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
    elif color_mode == 'rgba':
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
    elif color_mode == 'rgb':
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if pil_img.size != width_height_tuple:
            pil_img = pil_img.resize(width_height_tuple, Image.NEAREST)
    return pil_img


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
def get_stream(stream):
    # OPTIONS = [
    # {"id": 0, "text": "Ray Ban Wayfarer (Original)"},
    # {"id": 1, "text": "Ray Ban Wayfarer (New)"},
    # {"id": 2, "text": "Ray Ban Clubmaster (Classic)"},
    # {"id": 3, "text": "Ray Ban Aviator (Classic)"},
    # {"id": -1, "text": "Other model"}
    # ]
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    for eg in stream:
        if not eg["image"].startswith("data"):
            msg = "Expected base64-encoded data URI, but got: '{}'."
            raise ValueError(msg.format(eg["image"][:100]))

        pil_image = Image.open(io.BytesIO(b64_uri_to_bytes(eg["image"])))
        pil_image = preprocess_pil_image(pil_image)
        np_image = np.array(pil_image)
        eg["width"] = pil_image.width
        eg["height"] = pil_image.height

        pred = get_prediction(np_image)
        results =[]
        for i,value in enumerate(pred): 
            if str(value['cls'].int().item()) in cfg:
                # print(cfg[str(value['cls'].int().item())])            
                xmin= value['xyxy'][0].float().item()
                ymin= value['xyxy'][1].float().item()
                xmax= value['xyxy'][2].float().item()
                ymax= value['xyxy'][3].float().item()
                results+=[{"score":value['conf'].float().item(),
                    "label": cfg[str(value['cls'].int().item())],
                    "label_id":0,
                    "points": [[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]],
                    "hidden": 0}]
        eg["spans"] = results 
        # eg["options"] = OPTIONS       
        task = copy.deepcopy(eg)
        yield task


@prodigy.recipe(
    "image.manual",
    dataset=("The dataset to use", "positional", None, str),
    source=("Path to a directory of images", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    darken=("Darken image to make boxes stand out more", "flag", "D", bool),
)
def image_manual(
    dataset: str,
    source: str,
    label: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    darken: bool = False,
):
    """
    Manually annotate images by drawing rectangular bounding boxes or polygon
    shapes on the image.
    """
    blocks = [
        {"view_id": "image_manual"},
        # {"view_id": "choice","image":None,"text":None},
    ]
    # Load a stream of images from a directory and return a generator that
    # yields a dictionary for each example in the data. All images are
    # converted to base64-encoded data URIs.
    stream = Images(source)
    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": get_stream(stream),  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "config": {  # Additional config settings, mostly for app UI
            "label": ", ".join(label) if label is not None else "all",
            "labels": label,  # Selectable label options,
            "darken_image": 0.3 if darken else 0,
            "blocks":blocks
        },
    }
