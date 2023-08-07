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
from finetune import upload_images

def get_tmp_root(tmproot):
    if tmproot is None:
        tmproot = os.path.join(os.getcwd(),"tmp")
        if not os.path.exists(tmproot):
            os.makedirs(tmproot)
    return tmproot

def get_tmp_dir(tmproot,name):
    tmproot = get_tmp_root()
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
def render(prompt):
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
@click.argument('token', type=click.STRING)
@click.option('--tmpdir', type=click.STRING, default=None, help="Temporary directory for all training, default is ./tmp")
@click.option('--temperature', type=click.FLOAT, default=1.0, help="Temperature for training, default is 1.0")
@click.option('--masktarget', type=click.STRING, default="a face of a man", help="Mask target for training, default is 'a face'")
def upload(token,tmpdir,temperature, masktarget):
    """Upload the image set to Replicate.
    
    TOKEN: Token name we use for the training run, default is 'mysdxltoken'
    Outputs the URL where you can view progress."""

    tmpdir = get_tmp_root(tmpdir)
    upload_images(token,tmpdir,masktarget=masktarget,temperature=temperature)

repxl.add_command(render)    
repxl.add_command(prepare)    
repxl.add_command(upload)

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    repxl()