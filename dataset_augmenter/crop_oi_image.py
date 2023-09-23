import sys
import os
import glob
from PIL import Image
import cv2
import numpy as np


def crop_transp_bg_oi_image_pil(image_path):
    image_dir = os.path.dirname(image_path)
    image_name, image_ext = os.path.basename(image_path).split('.')
    new_file_name = f'crop_{image_name}.{image_ext}'
    new_file_path = os.path.join(image_dir, new_file_name)
    im = Image.open(image_path)
    im.getbbox()
    im2 = im.crop(im.getbbox())
    im2.save(new_file_path)
    return new_file_path

def crop_transp_bg_oi_image_cvs(image_path):
    image_dir = os.path.dirname(image_path)
    image_name, image_ext = os.path.basename(image_path).split('.')
    new_file_name = f'crop_{image_name}.{image_ext}'
    new_file_path = os.path.join(image_dir, new_file_name)

    im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # y,x = im[:,:,3].nonzero() # get the nonzero alpha coordinates

    # Set a threshold for what you consider "close to zero"
    threshold = 30  # Adjust this threshold as needed

    # Get the coordinates where alpha values are not close to zero
    y, x = np.where(np.abs(im[:, :, 3] - 0) > threshold)

    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)
    cropImg = im[miny:maxy, minx:maxx]
    # x, y, w, h = cv2.boundingRect(im[..., 3])
    # im2 = im[y:y+h, x:x+w, :]
    cv2.imwrite(new_file_path, cropImg)

    return new_file_path





if __name__ == '__main__':
    input_path = sys.argv[1]
    # print(crop_transp_bg_oi_image(image_path))
    img_regexp = os.path.join(input_path,'**', '*.png')
    for image_path in glob.glob(img_regexp, recursive=True):
        if 'details' in image_path or 'crop' in image_path:
            continue
        print(crop_transp_bg_oi_image_cvs(image_path))