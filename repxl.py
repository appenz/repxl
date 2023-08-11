#
# Stable Diffusion Shell
# Tool to automate SD workflow. For now just re-sizes images.
#
# (c) in 2022 by Guido Appenzeller

import sys
import os

import click
import tqdm
from PIL import Image
from dotenv import load_dotenv, find_dotenv

from imgtools import prep_images
from finetune import train_model

def get_tmp_root(tmproot):
    if tmproot is None:
        tmproot = os.path.join(os.getcwd(),"tmp")
        if not os.path.exists(tmproot):
            os.makedirs(tmproot)
    return tmproot

def get_tmp_dir(tmproot,name):
    tmproot = get_tmp_root(tmproot)
    tmp_dir =  os.path.join(tmproot,name)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        return tmp_dir
    else:
        print(f"Directory {tmp_dir} already exists. Delete or pick a different name with '--name <name>'")
        return None

@click.group()
def repxl():
    """Command line tool to run SDXL on Replicate."""
    pass

@repxl.command()
@click.argument('prompt', type=click.STRING)
@click.option('--model', type=click.STRING, help="Name of the model to use in the format 'username/modelname'")
def render(prompt,model):
    """Render a new image from a trained model."""
    print(prompt)
    pass

@repxl.command()
@click.argument('srcdir', type=click.STRING)
@click.option('--tmpdir', type=click.STRING, default=None, help="Temporary directory for all training, default is ./tmp")
@click.option('--token', type=click.STRING, default="mysdxltoken", help="Token name we use for the training run, default is 'mysdxltoken'")
def prepare(srcdir,tmpdir,token):
    """Prepare images for fine-tuning (crop/convert/zip).
    
    SRCDIR: directory with images to prepare, must be .png./.jpeg/.jpg format.
    By default temporary directory is created in current directory."""

    tmp_root = get_tmp_root(tmpdir) 
    tmp_dir = get_tmp_dir(tmp_root,token)
    prep_images(srcdir, tmp_dir, 1024, 1024, iname=".src.jpg")

    # Run the zip command to creat a zip file of the directory tmpdir    
    os.system(f"zip -j -r {tmp_root}/{token}.zip {tmp_dir}")

@repxl.command()
@click.option('--token', type=click.STRING, default="mysdxltoken", help="Token name we use for the training run, default is 'mysdxltoken'")
@click.option('--tmpdir', type=click.STRING, default=None, help="Temporary directory for all training, default is ./tmp")
@click.option('--masktarget', type=click.STRING, default="a face of a man", help="Mask target for training, default is 'a face'")
@click.option('--captionprefix', type=click.STRING, default="a photo of", help="Prefix before the token, default is a 'a photo of'")
@click.option('--dreambooth', type=click.BOOL, default=False, help="Use dreambooth instead of LoRA")
def train(token,tmpdir, masktarget, captionprefix, dreambooth):
    """Fine-tune SDXL on Replicate.
    
    Training progress can be viewed on https://replicate.com/trainings"""

    tmpdir = get_tmp_root(tmpdir)
    train_model(token,tmpdir,masktarget=masktarget,captionprefix=captionprefix, dreambooth=dreambooth)

repxl.add_command(render)    
repxl.add_command(prepare)    
repxl.add_command(train)

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    repxl()