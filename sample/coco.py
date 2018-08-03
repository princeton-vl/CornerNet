import cv2
import math
import numpy as np
import torch
import random
import string

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import random_crop, draw_gaussian, gaussian_radius

def _full_image_crop(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size   = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    detections[:, 0:4:2] += border[2]
    detections[:, 1:4:2] += border[0]
    return image, detections

def _resize_image(image, detections, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections

def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

def kp_detection(db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]

    max_tag_len = 128

    # allocating memory
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    br_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    tl_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    br_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens    = np.zeros((batch_size, ), dtype=np.int32)

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading image
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)

        # reading detections
        detections = db.detections(db_ind)

        # cropping an image randomly
        if not debug and rand_crop:
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)
        else:
            image, detections = _full_image_crop(image, detections)

        image, detections = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        if not debug and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1

        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        for ind, detection in enumerate(detections):
            category = int(detection[-1]) - 1

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)

            if gaussian_bump:
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad

                draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
                draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
            else:
                tl_heatmaps[b_ind, category, ytl, xtl] = 1
                br_heatmaps[b_ind, category, ybr, xbr] = 1

            tag_ind = tag_lens[b_ind]
            tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
            br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
            tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
            br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
            tag_lens[b_ind] += 1

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1

    images      = torch.from_numpy(images)
    tl_heatmaps = torch.from_numpy(tl_heatmaps)
    br_heatmaps = torch.from_numpy(br_heatmaps)
    tl_regrs    = torch.from_numpy(tl_regrs)
    br_regrs    = torch.from_numpy(br_regrs)
    tl_tags     = torch.from_numpy(tl_tags)
    br_tags     = torch.from_numpy(br_tags)
    tag_masks   = torch.from_numpy(tag_masks)

    return {
        "xs": [images, tl_tags, br_tags],
        "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]
    }, k_ind

def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)
