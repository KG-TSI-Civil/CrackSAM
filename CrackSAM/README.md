<h3>Code is revised from SAMed:</h3>

https://github.com/hitachinsk/SAMed

<h3>Train:</h3>

```python
python train.py --root_path D:\SAM\dataset\trainingset\  --val_path D:\SAM\dataset\validationset\  --warmup --AdamW --img_size 448  --n_gpu 1  --batch_size 8     --base_lr 0.0004  --warmup_period 300  --tf32  --use_amp --lr_exp 6 --max_epochs 140 --stop_epoch 100   --vit_name vit_h    --delta_type adapter --middle_dim 32  --save_interval 5
```

<h3>Test:</h3>

```python
python test.py --volume_path  D:\SAM\dataset\testset\   --is_savenii   --img_size 448     --ckpt D:\CrackSAM\checkpoints\sam_vit_h_4b8939.pth  --vit_name vit_h    --delta_type adapter  --middle_dim 32 --scaling_factor 0.2  --delta_ckpt D:\CrackSAM\checkpoints\CrackSAM_adapter_d32.pth
```
