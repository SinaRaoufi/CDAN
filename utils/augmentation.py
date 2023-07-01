import os
import random
import argparse

import cv2
import albumentations as A
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("trainPath", help="train folder path.")
parser.add_argument("augPath", help="augmentation folder path.")
parser.add_argument(
    "augPercent", help="Percentage[0-1] of images to be augmented.")
parser.add_argument(
    "folderFlag", help="A flag if it is a folder augmentation.")

args, unknown = parser.parse_known_args()

train_folder_path = args.trainPath
augmentaion_folder = args.augPath
augmentation_percentage = float(args.augPercent)
folder_augmentaion = eval(args.folderFlag)

augmentation_settings = [
    A.VerticalFlip(
        always_apply=True,
        p=1
    ),
    A.HorizontalFlip(
        always_apply=True,
        p=1
    )
]

aug_settings_names = [
    'vertical_flip',
    'horizontal_flip'
]

os.makedirs(augmentaion_folder, exist_ok=True)
outer_files = os.listdir(train_folder_path)
if '.DS_Store' in outer_files:
    outer_files.remove('.DS_Store')

for setting, name in zip(augmentation_settings, aug_settings_names):
    print(f'Augmentation setting: {name}')
    if folder_augmentaion:
        for file in tqdm(outer_files):
            os.makedirs(os.path.join(augmentaion_folder, file), exist_ok=True)
            inner_files = os.listdir(os.path.join(train_folder_path, file))
            inner_files = random.sample(inner_files, int(
                len(inner_files)*augmentation_percentage))
            if '.DS_Store' in inner_files:
                inner_files.remove('.DS_Store')
            for inner_file in inner_files:
                try:
                    img = cv2.imread(
                        os.path.join(train_folder_path, file, inner_file),
                        cv2.IMREAD_UNCHANGED
                    )
                    augmented_image = setting(image=img)['image']
                    augmented_image_path = os.path.join(
                        augmentaion_folder,
                        file,
                        f'aug_{name}_{inner_file}')
                    cv2.imwrite(augmented_image_path, augmented_image)
                except:
                    print(inner_file)
    else:
        inner_files = random.sample(outer_files, int(
            len(outer_files)*augmentation_percentage))
        for file in tqdm(inner_files):
            try:
                img = cv2.imread(
                    os.path.join(train_folder_path, file),
                    cv2.IMREAD_UNCHANGED
                )
                augmented_image = setting(image=img)['image']
                augmented_image_path = os.path.join(
                    augmentaion_folder,
                    f'aug_{name}_{file}')
                cv2.imwrite(augmented_image_path, augmented_image)
            except:
                print(file)
