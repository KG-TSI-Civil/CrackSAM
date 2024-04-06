<h2> Models come from OpenMMLab: </h2>
https://github.com/open-mmlab/mmsegmentation

&nbsp;

<h2> About Knowledge Distillation: </h2>
https://github.com/irfanICMLL/TorchDistiller

https://github.com/ChaoningZhang/MobileSAM
&nbsp;

Since the model is relatively large, it is not recommended to call the model for inference during the training phase, as this will occupy a lot of memory. 
Therefore, it is necessary to open data augmentation in advance, call the well-trained CrackSAM for inference n times on the dataset to obtain the inferred soft logits, and then save the image, label, and soft logits to the local storage simultaneously (this requires rewriting the dataloader). 
When training the student model, directly read the saved images along with their corresponding labels and soft logits.
