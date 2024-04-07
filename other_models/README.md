<h2> Models come from OpenMMLab: </h2>
https://github.com/open-mmlab/mmsegmentation

&nbsp;
<h4> DeepLabV3p_ResNet-50: </h4>

https://drive.google.com/file/d/1xg3B2xeyL3JqdWwrosdsayCASNWS3ok3/view?usp=sharing

<h4> DeepLabV3p_ResNet-101: </h4>

https://drive.google.com/file/d/1rqY4-NWAJbk0azkm6ecnlhNL2jnRYyBn/view?usp=sharing

<h4> SwinUperNet-tiny: </h4>

https://drive.google.com/file/d/1zqFZygXPSn7Pv_KZDXYuxDXfpOPN4Uuj/view?usp=sharing

<h4> SwinUperNet-Base: </h4>

https://drive.google.com/file/d/1UjiYAv6-9Yj9yr1x9-xUhQxwQ8Clo0wJ/view?usp=drive_link

<h4> SegFormer-MitB5: </h4>

https://drive.google.com/file/d/13vWAUzzd014nkjrDDE13ng1wLbKSBFnd/view?usp=sharing

<h4> ViT-Base: </h4>

https://drive.google.com/file/d/1eO6gLhRgVF-U7DWNyKMo73k30DBYe617/view?usp=sharing



&nbsp;

<h2> About Knowledge Distillation: </h2>
https://github.com/irfanICMLL/TorchDistiller

https://github.com/ChaoningZhang/MobileSAM
&nbsp;

Since the model is relatively large, it is not recommended to call the model for inference during the training phase, as this will occupy a lot of memory. 

Therefore, it is necessary to open data augmentation in advance, call the well-trained CrackSAM for inference n times on the dataset to obtain the inferred soft logits, and then save the image, label, and soft logits to the local storage simultaneously (this requires rewriting the dataloader). 

When training the student model, directly read the saved images along with their corresponding labels and soft logits.
