#
#  Fine-tine images with Stable Diffusion on Replicate
#

import os
import replicate
from dotenv import load_dotenv, find_dotenv

# Function that calls replicate to render an image with the standard SD XL model
def train_model(token,tmpdir, masktarget="a face", captionprefix="a photo of", dreambooth=False):
    """Upload images to Replicate."""

    filename = f'{tmpdir}/{token}.zip'
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Did you run prepare first?")
        return None

    caption_text = f'{captionprefix} {token}'

    print(f'Launching training run for token {token} on Replicate.')
    print(f'Caption: {caption_text}')
    training = replicate.trainings.create(
        version="stability-ai/sdxl:7ca7f0d3a51cd993449541539270971d38a24d9a0d42f073caf25190d41346d7",
        input={
            "input_images": "https://guido.appenzeller.net/wp-content/uploads/tmp/chrltt.zip",
            "caption_prefix": caption_text,
            "mask_target_prompts": masktarget,
            "resolution": 1024,
            "use_face_detection_instead": True,
            "is_lora": not dreambooth,
        },
        destination="appenz/sdxl-chrltt-1-lora"
    )
    print(training.status)

