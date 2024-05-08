<h3>Code is developed from SAMed: https://github.com/hitachinsk/SAMed</h3>


<h3>1. Download pretrained SAM weights (ViT-B, ViT-L, ViT-H) from https://github.com/facebookresearch/segment-anything and put it in the "checkpoints" folder</h3> 

<h3>2. Train:</h3>

Download the dataset and use your path

```python
python train.py --root_path D:\CrackSAM\dataset\trainingset\  --val_path D:\CrackSAM\dataset\validationset\  --warmup --AdamW --img_size 448  --n_gpu 1  --batch_size 8     --base_lr 0.0004  --warmup_period 300  --tf32  --use_amp --lr_exp 6 --max_epochs 140 --stop_epoch 100   --vit_name vit_h    --delta_type adapter --middle_dim 32 --scaling_factor 0.2 --save_interval 5
```

<h3>3. Test:</h3>

```python
python test.py --volume_path  D:\CrackSAM\dataset\testset\   --is_savenii   --img_size 448     --ckpt D:\CrackSAM\checkpoints\sam_vit_h_4b8939.pth  --vit_name vit_h    --delta_type adapter  --middle_dim 32 --scaling_factor 0.2  --delta_ckpt D:\CrackSAM\checkpoints\CrackSAM_adapter_d32.pth
```
