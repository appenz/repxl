#
#  Fine-tine images with Stable Diffusion on Replicate
#

import os
import replicate
from dotenv import load_dotenv, find_dotenv

# Function that calls replicate to render an image with the standard SD XL model
def upload_images(token,tmpdir,temperature=1.0, masktarget="a face"):
    """Upload images to Replicate."""

    filename = f'{tmpdir}/{token}.zip'
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Did you run prepare first?")
        return None

    caption_text = f'a photo of a {token}'

    output = replicate.run(
        "replicate/sdxl_preprocess:bd1158a5052ed46176da900ad7e2a80ea04a3c46196d93f9e1db879fd1ce7f29",
        input={
            "files": open(filename, "rb"),
            "caption_text": caption_text,
            "mask_target_prompts": masktarget,
            "target_size": 1024,
            "crop_based_on_salience": True,
            "use_face_detection_instead": False,
            "temp": temperature,
        },
    )

    print(output) # Expect to get url to .tar file containing format above, like : https://replicate.delivery/pbxt/xxxx/data.tar
