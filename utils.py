import numpy as np
import cv2

import random


def resize_and_crop(image, target_size=(1904, 1120), patch_size=224):
    image = cv2.resize(image, target_size)
    patches = []
    for i in range(0, image.shape[0], patch_size//2):
        for j in range(0, image.shape[1], patch_size//2):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return np.array(patches)


def random_crop_around_point(image, x_rel, y_rel, image_size=(1920, 1080), crop_size=(960, 540)):
    buffer = 50
    x = int(x_rel * image_size[0])
    y = int(y_rel * image_size[1])
    x1min = max(0, x - crop_size[0] + buffer)
    x1max = max(min(x - buffer, image_size[0] - crop_size[0]), x1min)
    y1min = max(0, y - crop_size[1] + buffer)
    y1max = max(min(y - buffer, image_size[1] - crop_size[1]), y1min)
    x1 = random.randint(x1min, x1max)
    y1 = random.randint(y1min, y1max)
    x2 = x1 + crop_size[0]
    y2 = y1 + crop_size[1]
    image = cv2.resize(image, image_size)
    return image[y1:y2, x1:x2]


def yolo_to_patch_matrix(boxes, patch_matrix_size=(16, 9)):
    """
    Convert YOLO bounding box coordinates to patch indices in a matrix.
    Result is a 2D binary matrix with ones indicating the patch indices with where the bounding box is located.
    """

    # Create a binary matrix
    patch_matrix = np.zeros(patch_matrix_size, dtype=np.uint8)

    if len(boxes) == 0:
        return patch_matrix

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # Convert to integer indices
    i1 = np.floor(y1 * patch_matrix_size[0]).astype(int)
    j1 = np.floor(x1 * patch_matrix_size[1]).astype(int)
    i2 = np.floor(y2 * patch_matrix_size[0]).astype(int)
    j2 = np.floor(x2 * patch_matrix_size[1]).astype(int)

    # Ensure indices are within bounds
    i1 = np.clip(i1, 0, patch_matrix_size[0] - 1)
    j1 = np.clip(j1, 0, patch_matrix_size[1] - 1)
    i2 = np.clip(i2, 0, patch_matrix_size[0] - 1)
    j2 = np.clip(j2, 0, patch_matrix_size[1] - 1)

    
    # Fill the matrix with ones in the patch indices
    for i in range(len(boxes)):
        patch_matrix[i1[i]:i2[i] + 1, j1[i]:j2[i] + 1] = 1

    return patch_matrix

def patch_probs_to_probs_matrix(batch_of_patch_probs, patch_matrix_size=(9, 16)):
    """
    Convert a batch of patch probabilities to a matrix of probabilities.
    The resulting matrix has the same size as the patch matrix.
    """
    batch_size = batch_of_patch_probs.shape[0]
    probs_matrix = np.zeros((batch_size, patch_matrix_size[0]+1, patch_matrix_size[1]+1), dtype=np.float32)

    for i in range(batch_size):
        # Divide the patch probabilities by 4 to get the probability of each sub-patch
        p_matrix = np.reshape(batch_of_patch_probs[i], patch_matrix_size) / 4
        # handle consequtive overlaping patches
        q_matrix = 1 - p_matrix
        q_matrix_overlapped = np.zeros((patch_matrix_size[0]+1, patch_matrix_size[1]+1), dtype=np.float32)
        q_matrix_overlapped[1:-1, 1:-1] = (q_matrix[:-1, :-1] * q_matrix[1:, :-1] * q_matrix[:-1, 1:] * q_matrix[1:, 1:])
        q_matrix_overlapped[0, 0] = q_matrix[0, 0]
        q_matrix_overlapped[0, -1] = q_matrix[0, -1]
        q_matrix_overlapped[-1, 0] = q_matrix[-1, 0]
        q_matrix_overlapped[-1, -1] = q_matrix[-1, -1]
        q_matrix_overlapped[0, 1:-1] = q_matrix[0, :-1] * q_matrix[0, 1:]
        q_matrix_overlapped[-1, 1:-1] = q_matrix[-1, :-1] * q_matrix[-1, 1:]
        q_matrix_overlapped[1:-1, 0] = q_matrix[:-1, 0] * q_matrix[1:, 0]
        q_matrix_overlapped[1:-1, -1] = q_matrix[:-1, -1] * q_matrix[1:, -1]
        probs_matrix[i] = 1 - q_matrix_overlapped
        probs_matrix[i] = (probs_matrix[i] // 0.01) * 0.01

    return probs_matrix

def visualize_mask(probs_matrix, mask_size_px=(60, 19)):
    opacity_matrix = 1 - probs_matrix
    opacity_matrix = np.clip(opacity_matrix, 0, 1)
    opacity_matrix = (opacity_matrix * 255).astype(np.uint8)
    opacity_matrix = cv2.cvtColor(opacity_matrix, cv2.COLOR_GRAY2BGR)
    opacity_matrix = cv2.applyColorMap(opacity_matrix, cv2.COLORMAP_JET)
    opacity_matrix = cv2.cvtColor(opacity_matrix, cv2.COLOR_BGR2RGB)
    opacity_matrix = cv2.resize(opacity_matrix, mask_size_px)
    return opacity_matrix