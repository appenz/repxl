#
#  Render images with Stable Diffusion on Replicate
#

import os
import replicate
from dotenv import load_dotenv, find_dotenv

# list of models
model_sd_xl = "stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2"

# Function that calls replicate to render an image with the standard SD XL model
def render_sdxl(prompt, model=model_sd_xl):
    """Render a new image with the standard SD XL."""
    output = replicate.run(
        model_sd_xl,
        input={"prompt": prompt}
    )
    return output

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    url = render_sdxl("A hamster with a wizard hat")
    print(url[0])