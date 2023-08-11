# replifusion

Command line tool to fine-tune and run SD XL models on Replicate. 

Right now this is work in progress and will require a few manual steps.

## Installation

Tested on macOS and requires python3. Install requirements:

```pip install -r requirements.txt ```

## Fine tuning a model

Sign up for a replicate account and generate an API token. The API token needs to be in the environment variable REPLICATE_API_TOKEN. An easy way to do this is to create an ".env" file in the project root direction that looks like this:

```REPLICATE_API_TOKEN=r8_X...``````

Next create a model on Replicate, [on this page here](https://replicate.com/create?name=my-model&visibility=private&hardware=gpu-a40-large). This can't be done via the API yet (or at least I don't know how yet). As hardware I'd recommend the A40. Add the model name in finetune.py. If your username is "bob123" and the model is "specialfinetune" it should look something like this:

```destination="bob123/specialfinetune"```

Next pick a token you want to use for the new concept that you train on. Pick something that is not a word, e.g. `mysdxltoken` should work.

Now use the tool to create a zip of all the images in the right format:

```python3 repxl.py prepare --token <yourtoken> <directory with your images>```

This should generate a zip file in the `./tmp` directory. You need to host this zip file somewhere on the internet where it is accessible (e.g. and S3 bucket, web setver etc.)

Once that is done, you can start training with:

```python3 repxl.py train --token <yourtoken>```

Training progress can be veiwed at `https://replicate.com/trainings`. Once it is complete, the page has a link to the model and you can download it or use it to generate images on replicate. Just type a prompt with your token, e.g. `photo of man mysdxltoken smiling` and it should get the right result.

## Options

This is WIP, run the command to get a list of supported options.
 
```python3 repxl.py --help```
 
 ## How much does this cost?

 Cost depends on many parameters but expect image generation to cost 5-10 cents/image. Training depends on the number of images but for me it took about 8 minutes for a total cost of $0.75.
 
  
  
