import cv2
import os
import shutil
import random
from tqdm import tqdm

import sys
sys.path.append('..')

from global_params import *

def load_cropped(path, crop_frac):
    """
    Loads image cropped to crop_frac fraction of width/height.

    Parameters:
    path (str): Path to input image.
    crop_frac (float, range=[0, 0.5)): Fraction of width/height to crop at each edge.

    Returns:
    NumPy ndarray: Cropped image.
    """

    if(crop_frac < 0 or crop_frac >= 0.5):
        raise ValueError('crop_frac must be a float in the range [0, 0.5).')

    img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)

    return img[int(crop_frac * img.shape[0]):int((1 - crop_frac) * img.shape[0]), int(crop_frac * img.shape[1]):int((1 - crop_frac) * img.shape[1])]

if __name__ == '__main__':
    random.seed(SEED)

    data_dir = os.path.join('lfw2', 'lfw2')
    train_dir = 'train'
    test_dir = 'test'

    # Making directories for train and test (overwriting if already exist).
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    print(f'Splitting data into train and test...\n\tmin_img_per_class={min_img_per_class}\n\ttest_frac={test_frac}')
    included_img_count_train = 0        # Count of images included in train.
    included_img_count_test = 0         # Count of images included in test.
    included_class_count = 0            # Count of classes included in train and test.
    for class_ in tqdm(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_)
        # Number of images in current class.
        num_img_class = len(os.listdir(class_path))

        # If not enough images in current class, dropping it.
        if num_img_class < min_img_per_class:
            continue

        included_class_count += 1
        os.makedirs(os.path.join(train_dir, class_))
        os.makedirs(os.path.join(test_dir, class_))

        # Sorting and then shuffling as os.listdir returns in random order. Paths to images in current class.
        img_src_paths = os.listdir(class_path)
        img_src_paths = [os.path.join(data_dir, class_, img_src_path) for img_src_path in img_src_paths]
        random.shuffle(img_src_paths)

        # Performing train-test split.
        split_idx = int((1 - test_frac) * len(img_src_paths))
        img_src_paths_train = img_src_paths[:split_idx]
        img_src_paths_test = img_src_paths[split_idx:]

        included_img_count_train += len(img_src_paths_train)
        included_img_count_test += len(img_src_paths_test)

        for img_src_path in img_src_paths_train:
            img = load_cropped(img_src_path, crop_frac)
            img_basename = os.path.basename(img_src_path)
            img_dest_path = os.path.join(train_dir, class_, img_basename)
            cv2.imwrite(img_dest_path, img)

        for img_src_path in img_src_paths_test:
            img = load_cropped(img_src_path, crop_frac)
            img_basename = os.path.basename(img_src_path)
            img_dest_path = os.path.join(test_dir, class_, img_basename)
            cv2.imwrite(img_dest_path, img)

    print(f'Image count, {included_class_count} classes:\n\ttrain: {included_img_count_train} images\n\ttest:  {included_img_count_test} images')