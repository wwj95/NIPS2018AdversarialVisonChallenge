# NIPS2018AdversarialVisonChallenge
* I am a team member in NIPS2018 Adverdial Vision Challenge.   
* I am foucs on Robust model track. The purpose is to improve the model robustness to attack images.   
* Attack images are generated from original image using both calini attack and fgsm. The attck methods aims to add small perturbation on the clean image and fool the model to make wrong classification.   
* In the defense team, we adopt Ensemble adverial Training methods. We trained VGG16 and VGG19 on clean image from Tiny ImageNet and Use VGG16 and VGG19 to apply Ensemble Adversarial Training on Resnet18.
* We also apllied Gausian Data Augmentation and random resize and padding as a preprocessing layer to increase its generalization capabilities.
* The competition is still under going. We hope tp have some great result in the final submission.
