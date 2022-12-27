from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from pipeline_stable_diffusion_vision import StableDiffusionPipelineVision
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from converter_model import ConverterModel, ConverterConfig
model_path = "train_fashion_to_image_v4_2022-12-23"

generator = torch.Generator(device="cuda").manual_seed(1)

# Load models and create wrapper for stable diffusion
#tokenizer = CLIPTokenizer.from_pretrained(
#    model_path, subfolder="tokenizer",
#)
#text_encoder = CLIPTextModel.from_pretrained(
#    model_path,
#    subfolder="text_encoder",
#)
converter = ConverterModel.from_pretrained(
    model_path,
    config=ConverterConfig(),
    subfolder="converter",
)
vision_encoder = CLIPVisionModel.from_pretrained(
    model_path,
    subfolder="vision_encoder",
)
vae = AutoencoderKL.from_pretrained(
    model_path,
    subfolder="vae",
)
unet = UNet2DConditionModel.from_pretrained(
    model_path,
    subfolder="unet",
)
#text_encoder.to("cuda")
#vae.to("cuda")
#pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = StableDiffusionPipelineVision.from_pretrained(
    model_path,
    vision_encoder=vision_encoder,
    converter=converter,
    vae=vae,
    unet=unet,
)
pipe.to("cuda")

#hiraoki_path = "/shared/datasets/datasets/fashion/FashionTryOn/v2.0/raw_train_10007_0_hiraoki.jpg"
hiraoki_path = "/shared/datasets/datasets/fashion/FashionTryOn/v2.0/raw_train_21010_1_hiraoki.jpg"
image = pipe(inimg=hiraoki_path, generator=generator).images[0]
#image.save("test.png")
image.save("test2.png")
