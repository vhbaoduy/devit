import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

torch.set_grad_enabled(False)
import numpy as np
import fire
import os.path as osp
from detectron2.config import get_cfg
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from tools.train_net import Trainer, DetectionCheckpointer
from glob import glob

import torchvision as tv
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

import matplotlib.colors
import seaborn as sns
import torchvision.ops as ops
from torchvision.ops import box_area, box_iou
import random

import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw, ImageFont
from copy import copy
import streamlit as st


# --- Model Loading ---


def process_image(image, format=None):
    image = utils._apply_exif_orientation(image)
    return utils.convert_PIL_to_numpy(image, format)


def list_replace(lst, old=1, new=10):
    """replace list elements (inplace)"""
    i = -1
    lst = copy(lst)
    try:
        while True:
            i = lst.index(old, i + 1)
            lst[i] = new
    except ValueError:
        pass
    return lst


def filter_boxes(instances, threshold=0.0):
    indexes = instances.scores >= threshold
    assert indexes.sum() > 0
    boxes = instances.pred_boxes.tensor[indexes, :]
    pred_classes = instances.pred_classes[indexes]
    return boxes, pred_classes, instances.scores[indexes]


def assign_colors(pred_classes, label_names, seed=1):
    all_classes = torch.unique(pred_classes).tolist()
    all_classes = list(set([label_names[ci] for ci in all_classes]))
    colors = list(sns.color_palette("hls", len(all_classes)).as_hex())
    random.seed(seed)
    random.shuffle(colors)
    class2color = {}
    for cname, hx in zip(all_classes, colors):
        class2color[cname] = hx
    colors = [class2color[label_names[cid]] for cid in pred_classes.tolist()]
    return colors


def load_model(
    config_file="configs/open-vocabulary/lvis/vitl.yaml",
    rpn_config_file="configs/RPN/mask_rcnn_R_50_FPN_1x.yaml",
    model_path="weights/trained/open-vocabulary/lvis/vitl_0069999.pth",
    image_dir="demo/input",
    output_dir="demo/output",
    # category_space="demo/ycb_prototypes.pth",
    device="cpu",
    topk=1,
):
    assert osp.abspath(image_dir) != osp.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = get_cfg()
    config.merge_from_file(config_file)
    config.DE.OFFLINE_RPN_CONFIG = rpn_config_file
    config.DE.TOPK = topk
    config.MODEL.MASK_ON = True

    config.freeze()

    augs = utils.build_augmentation(config, False)
    augmentations = T.AugmentationList(augs)

    # building models
    model = Trainer.build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    model.eval()
    model = model.to(device)

    label_names = model.label_names
    if "mini soccer" in label_names:  # for YCB
        label_names = list_replace(label_names, old="mini soccer", new="ball")
    return model, augmentations


MODEL, AUGMENTATIONS = load_model()


def predict(
    model,
    augmentations,
    image,
    overlapping_mode=True,
    threshold=0.45,
    device="cpu",
    category_space=None,
):
    if category_space is not None:
        category_space = torch.load(category_space)
        model.label_names = category_space["label_names"]
        model.test_class_weight = category_space["prototypes"].to(device)

    image = process_image(image)
    label_names = model.label_names
    if "mini soccer" in label_names:  # for YCB
        label_names = list_replace(label_names, old="mini soccer", new="ball")

    # for img_file in glob(osp.join(image_dir, "*")):
    #     base_filename = osp.splitext(osp.basename(img_file))[0]

    dataset_dict = {}
    # image = utils.read_image(img_file, format="RGB")
    dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1]

    aug_input = T.AugInput(image)
    augmentations(aug_input)
    dataset_dict["image"] = torch.as_tensor(
        np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))
    ).to(device)

    batched_inputs = [dataset_dict]

    output = model(batched_inputs)[0]
    output["label_names"] = model.label_names
    # if output_pth:
    #     torch.save(output, osp.join(output_dir, base_filename + '.pth'))

    # visualize output
    instances = output["instances"]
    boxes, pred_classes, scores = filter_boxes(instances, threshold=threshold)

    if overlapping_mode:
        # remove some highly overlapped predictions
        mask = box_area(boxes) >= 400
        boxes = boxes[mask]
        pred_classes = pred_classes[mask]
        scores = scores[mask]
        mask = ops.nms(boxes, scores, 0.3)
        boxes = boxes[mask]
        pred_classes = pred_classes[mask]
        areas = box_area(boxes)
        indexes = list(range(len(pred_classes)))
        for c in torch.unique(pred_classes).tolist():
            box_id_indexes = (pred_classes == c).nonzero().flatten().tolist()
            for i in range(len(box_id_indexes)):
                for j in range(i + 1, len(box_id_indexes)):
                    bid1 = box_id_indexes[i]
                    bid2 = box_id_indexes[j]
                    arr1 = boxes[bid1].cpu().numpy()
                    arr2 = boxes[bid2].cpu().numpy()
                    a1 = np.prod(arr1[2:] - arr1[:2])
                    a2 = np.prod(arr2[2:] - arr2[:2])
                    top_left = np.maximum(arr1[:2], arr2[:2])  # [[x, y]]
                    bottom_right = np.minimum(arr1[2:], arr2[2:])  # [[x, y]]
                    wh = bottom_right - top_left
                    ia = wh[0].clip(0) * wh[1].clip(0)
                    if ia >= 0.9 * min(
                        a1, a2
                    ):  # same class overlapping case, and larger one is much larger than small
                        if a1 >= a2:
                            if bid2 in indexes:
                                indexes.remove(bid2)
                        else:
                            if bid1 in indexes:
                                indexes.remove(bid1)

        boxes = boxes[indexes]
        pred_classes = pred_classes[indexes]
    colors = assign_colors(pred_classes, label_names, seed=4)
    output = to_pil_image(
        draw_bounding_boxes(
            torch.as_tensor(image).permute(2, 0, 1),
            boxes,
            labels=[label_names[cid] for cid in pred_classes.tolist()],
            colors=colors,
        )
    )
    # output.save(osp.join(output_dir, base_filename + ".out.jpg"))
    return output


# --- Streamlit App ---
def main():
    st.title("Few-Shot Object Detection with DET-ViT")

    # --- Support Images and Annotations ---
    st.header("Support Images (Few-Shot)")
    # support_images = []
    # support_annotations = []
    # new_class_labels = []

    # for i in range(st.number_input("Number of support images", min_value=1, value=1)):
    #     uploaded_support_image = st.file_uploader(
    #         f"Support Image {i+1}", type=["jpg", "jpeg", "png"], key=f"support_{i}"
    #     )
    #     if uploaded_support_image:
    #         image = Image.open(uploaded_support_image)
    #         support_images.append(image)
    #         # Add annotation logic here (e.g., using streamlit-image-coordinates).
    #         # support_annotations.append(annotations)
    #         new_class_labels.append(
    #             st.text_input(f"Label for support image {i+1}", key=f"label_{i}")
    #         )

    # if st.button("Upload support image"):
    #     if support_images:
    #         # adapted_model = adapt_model(
    #         #     support_images, support_annotations, new_class_labels
    #         # )
    #         # st.session_state.adapted_model = adapted_model
    #         pass
    #     else:
    #         st.warning("Please upload support images.")
    agree = st.checkbox("Use protypes in YCB objects")

    # --- Query Image and Detection ---
    st.header("Query Image")
    uploaded_query_image = st.file_uploader(
        "Upload Query Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_query_image:
        query_image = Image.open(uploaded_query_image)
        # query_image = utils.read_image(uploaded_query_image)
        st.image(query_image, caption="Query Image", use_container_width=True)

        if st.button("Predict"):
            detected_results = predict(
                model=MODEL,
                augmentations=AUGMENTATIONS,
                image=query_image,
                category_space="demo/ycb_prototypes.pth" if agree else None,
            )
            # Visualize the results using OpenCV or PIL.
            # Draw bounding boxes and labels.
            # st.write(detected_results)  # print the results for debugging.
            # st.image(visualized_image, caption="Detected Objects", use_column_width=True)
            st.image(detected_results, caption="Result", use_container_width=True)


if __name__ == "__main__":
    main()
