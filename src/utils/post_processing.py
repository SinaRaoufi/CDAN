import os

from PIL import Image, ImageEnhance


def post_processing(image_folder, dest_folder):
    images = os.listdir(image_folder)
    
    for img in images:
        im = Image.open(os.path.join(image_folder, img))

        contrasted = ImageEnhance.Contrast(im)
        contrasted = contrasted.enhance(1.12)

        colored = ImageEnhance.Color(contrasted)
        colored = colored.enhance(1.35)

        colored.save(os.path.join(dest_folder, img))