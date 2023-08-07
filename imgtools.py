import sys
import os

import tqdm
import dotenv
from PIL import Image

# Take the largest possible square from the center of the image  
# Right now, only works with square dimensions

def crop_center(img, tWidth, tHeight):
    width, height = img.size   # Get dimensions
    x = min(width, height)/2
    cx, cy = (width/2, height/2)
    tmp = img.crop((cx-x, cy-x, cx+x, cy+x))
    return tmp.resize((tWidth, tHeight), Image.Resampling.LANCZOS)

def prep_images(srcdir, tmpdir, trainwidth, trainheight, iname=".src.jpg"):
    """Resize and reformat images for training."""

    files = os.listdir(srcdir)
    print(f'Creating {len(files)} training files from {srcdir} -> {tmpdir}.')
    i = 0
    for filename in tqdm.tqdm(files):
        # Skip files that don't end with .png, .jpg or .jpeg
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img = Image.open(os.path.join(srcdir, filename))
        img = img.convert('RGB')
        out = crop_center(img,trainwidth,trainheight)

        fname_out = f'{i}{iname}'
        fname_out = os.path.join(tmpdir, fname_out)
        out.save(fname_out, "jpeg")
        i = i+1




