import torch
import numpy as np
import os
import argparse
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from typing import List
from huggingface_hub import snapshot_download

# Argument parser
parser = argparse.ArgumentParser(description="Run IDM-VTON with custom human and garment images.")
parser.add_argument("--human_img", type=str, required=True, help="Path to the human image file")
parser.add_argument("--garm_img", type=str, required=True, help="Path to the garment image file")
parser.add_argument("--output_img", type=str, required=True, help="Path and name of output image (without extension as .png will be appended)")
args = parser.parse_args()

output_path = f"{args.output_img}.png"

# Download and cache model locally
model_path = snapshot_download(repo_id="yisol/IDM-VTON", local_dir="saved_models/IDM-VTON")

# Set paths
base_path = 'saved_models/IDM-VTON'

# Load models
unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16)

# Disable gradient computation
for model in [UNet_Encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two]:
    model.requires_grad_(False)

tensor_transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Initialize parsing models
parsing_model = Parsing(0)
openpose_model = OpenPose(0)

# Load Try-On pipeline
pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

# Load images from command-line arguments
human_img_path = args.human_img
garment_img_path = args.garm_img

if not os.path.exists(human_img_path) or not os.path.exists(garment_img_path):
    raise FileNotFoundError("One or both input image paths do not exist!")

human_img = Image.open(human_img_path).convert("RGB").resize((768, 1024))
garm_img = Image.open(garment_img_path).convert("RGB").resize((768, 1024))

# Preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
openpose_model.preprocessor.body_estimation.model.to(device)
pipe.to(device)
pipe.unet_encoder.to(device)

keypoints = openpose_model(human_img.resize((384, 512)))
model_parse, _ = parsing_model(human_img.resize((384, 512)))
mask, mask_gray = get_mask_location("hd", "upper_body", model_parse, keypoints)
mask = mask.resize((768, 1024))

mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

args = apply_net.create_argument_parser().parse_args((
    'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl',
    'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'
))
pose_img = args.func(args, human_img_arg)
pose_img = pose_img[:, :, ::-1]
pose_img = Image.fromarray(pose_img).resize((768, 1024))

# Generate try-on result
with torch.no_grad():
    # Extract the images
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            prompt = "model is wearing Description of garment ex) Short Sleeve Round Neck T-shirts"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                                    
                prompt = "a photo of Description of garment ex) Short Sleeve Round Neck T-shirts"
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(prompt, List):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                generator = torch.Generator(device).manual_seed(42)
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device,torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                    num_inference_steps=30,
                    generator=generator,
                    strength = 1.0,
                    pose_img = pose_img.to(device,torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                    cloth = garm_tensor.to(device,torch.float16),
                    mask_image=mask,
                    image=human_img, 
                    height=1024,
                    width=768,
                    ip_adapter_image = garm_img.resize((768,1024)),
                    guidance_scale=2.0,
                )[0]

    images[0].save(output_path)
    print(f"Generated try-on image saved at: {output_path}")
