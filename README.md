# fasion_to_image
Diffusersを使って、平置き画像を着衣画像に変換

# Setup

```
pip install -r requirements.txt
```

# Train

```
python train_fashion_to_image.py \
       --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
       --use_ema \
       --resolution=224 \
       --center_crop \
       --train_batch_size=64 \
       --gradient_accumulation_steps=1  \
       --num_train_epochs=1000  \
       --learning_rate=1e-05  \
       --max_grad_norm=1 \
       --lr_scheduler="constant" \
       --lr_warmup_steps=0  \
       --output_dir="output" \
       --train_data_dir=/shared/datasets/datasets/fashion/FashionTryOn/v2.0
```

# Eval

```
python infer_fashion_to_image.py \
       --model_path output \
       --resolution 224 \
       /shared/datasets/datasets/fashion/FashionTryOn/v2.0/raw_train_1111*_hiraoki.jpg
```
