<h2> Models mainly come from OpenMMLab: </h2>
<h3>https://github.com/open-mmlab/mmsegmentation</h3>

&nbsp;
<h4> DeepLabV3p_ResNet-50: </h4>

https://drive.google.com/file/d/1xg3B2xeyL3JqdWwrosdsayCASNWS3ok3/view?usp=sharing

<h4> DeepLabV3p_ResNet-101: </h4>

https://drive.google.com/file/d/1rqY4-NWAJbk0azkm6ecnlhNL2jnRYyBn/view?usp=sharing

<h4> ResNet_PSPNet: </h4>

https://drive.google.com/file/d/1E0lDWBmXk98QT5dGx8-TdFyEPDZEvIfZ/view?usp=sharing

<h4> SwinUperNet-Tiny: </h4>

https://drive.google.com/file/d/1zqFZygXPSn7Pv_KZDXYuxDXfpOPN4Uuj/view?usp=sharing

<h4> SwinUperNet-Base: </h4>

https://drive.google.com/file/d/1UjiYAv6-9Yj9yr1x9-xUhQxwQ8Clo0wJ/view?usp=drive_link

<h4> SegFormer-MitB5: </h4>

https://drive.google.com/file/d/13vWAUzzd014nkjrDDE13ng1wLbKSBFnd/view?usp=sharing

<h4> ViT-Base: </h4>

https://drive.google.com/file/d/1eO6gLhRgVF-U7DWNyKMo73k30DBYe617/view?usp=sharing

<h4> HRNet_FCN: </h4>

https://drive.google.com/file/d/1cizlUnyeMbDSIqu8Kl5VH4TsoVrjPCgC/view?usp=sharing

<h4> UNet_PSPNet: </h4>

https://drive.google.com/file/d/1AOzeJ_F9sgzOQC1GTWsJnrIKfKSC57IR/view?usp=sharing

<h4> UNet_FCN: </h4>

https://drive.google.com/file/d/1glwMbUP-5IBw9GCy9b_9gvJzTwDpbEbM/view?usp=sharing

<h4> MobileNetV3: </h4>

https://drive.google.com/file/d/1gJ1NDAQeLn8l6NcSaO_o_f1UNOYHJsaZ/view?usp=sharing

<h4> VGG_UNet: </h4>

https://github.com/khanhha/crack_segmentation/blob/master/unet/unet_transfer.py

&nbsp;

<h2> About Knowledge Distillation: </h2>

&nbsp;

Since the model is relatively large, it is not recommended to call the model for inference during the training phase, as this will occupy a lot of memory. 

Therefore, it is necessary to open data augmentation in advance, call the well-trained CrackSAM for inference n times on the dataset to obtain the inferred soft logits, and then save the images, labels, and soft logits to the local storage simultaneously (this requires rewriting the dataloader). <h3>https://github.com/ChaoningZhang/MobileSAM</h3>

When training the student model, directly read the saved images along with their corresponding labels and soft logits.

Here, we adopt Channel-wise distillation: <h3>https://github.com/irfanICMLL/TorchDistiller</h3>.
