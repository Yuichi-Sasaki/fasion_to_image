# fasion_to_image
Diffusersを使って、平置き画像を着衣画像に変換

# Setup

```
conda create -n fashion_diffusion python=3.8
conda activate fashion_diffusion
pip install -r requirements.txt
```

# Train

```
python train_fashion_to_image.py \
--train_data_dir=/shared/datasets/datasets/fashion/FashionTryOn/v2.0 \
--use_ema \
--center_crop \
--num_train_epochs=10000  \
--max_grad_norm=1 \
--lr_scheduler="constant" \
--resolution=224 \
--lr_warmup_steps=0  \
--learning_rate=1e-05  \
--gradient_accumulation_steps=1  \
--train_batch_size=48 \
--train_vision_encoder \
--output_dir="output"
```

```
tensorboard --logdir output --host=0.0.0.0
```

# Eval

```
python infer_fashion_to_image.py \
       --model_path output \
       --resolution 224 \
       /shared/datasets/datasets/fashion/FashionTryOn/v2.0/raw_train_1111*_hiraoki.jpg
```
