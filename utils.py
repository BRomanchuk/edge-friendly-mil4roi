import numpy as np
import cv2

def resize_and_crop(image, target_size=(1904, 1120), patch_size=224):
    image = cv2.resize(image, target_size)
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return np.array(patches)

def yolo_to_patch_matrix(boxes, patch_matrix_size=(17, 10)):
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

