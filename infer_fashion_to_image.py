import argparse
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from pipeline_stable_diffusion_vision import StableDiffusionPipelineVision
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from converter_model import ConverterModel, ConverterConfig
import os
model_path = "train_fashion_to_image_v4_2022-12-23"

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to saved model pipeline",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
    )
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument('inputs', metavar='N', type=str, nargs='+')

    args = parser.parse_args()
    return args


args = parse_args()

generator = torch.Generator(device="cuda").manual_seed(args.seed)

if True:
    converter = ConverterModel.from_pretrained(
        args.model_path,
        config=ConverterConfig(),
        subfolder="converter",
    )
    vision_encoder = CLIPVisionModel.from_pretrained(
        args.model_path,
        subfolder="vision_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path,
        subfolder="unet",
    )
    pipe = StableDiffusionPipelineVision.from_pretrained(
        args.model_path,
        vision_encoder=vision_encoder,
        converter=converter,
        vae=vae,
        unet=unet,
    )
else:
    pipe = StableDiffusionPipelineVision.from_pretrained(
        args.model_path,
    )

pipe.to("cuda")

for f in args.inputs:
    image = pipe(inimg=f, generator=generator).images[0]
    output_path = os.path.join(args.output_dir, os.path.basename(f))
    os.makedirs(args.output_dir, exist_ok=True)
    image.save(output_path)
