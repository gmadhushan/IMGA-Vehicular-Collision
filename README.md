# Instance Mask-Guided Attention (IMGA) for Vehicular Collision

We present an Instance Mask-Guided Attention (IMGA) Network for vehicular collision image classification. The Vehicular Collision Image Classification Dataset is created from crowd-sourced dashboard camera videos containing on-road vehicular collision. The dataset consists of images extracted from accident videos forming two classes namely - (i) No Collision - frames extracted from video timestamps before vehicular collision and (ii) Collision - frames extracted from video timestamps on/after vehicular collision. The ability to differentiate between these images based on spatial difference is achieved by guidance of instance masks of vehicles involved in collision. The proposed IMGA network produces an accuracy of 97.94 % in the presented vehicular collision image classification dataset.

**Get access through request to the dataset with the following link:**
LINK

**Generate instance masks of vehicles undergoing collision using DeepMAC_only_collided_vehicles_mask.py**
Download and extract the DeepMac tensorflow object detection model pretrained on COCO dataset from the link 
download.tensorflow.org/models/object_detection/tf2/20210329/deepmac_1024x1024_coco17.tar.gz


