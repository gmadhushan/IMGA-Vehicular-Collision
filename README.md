# Instance Mask-Guided Attention (IMGA) for Vehicular Collision

We present an Instance Mask-Guided Attention (IMGA) Network for vehicular collision image classification. The Vehicular Collision Image Classification Dataset is created from crowd-sourced dashboard camera videos containing on-road vehicular collision. The dataset consists of images extracted from accident videos forming two classes namely - (i) No Collision - frames extracted from video timestamps before vehicular collision and (ii) Collision - frames extracted from video timestamps on/after vehicular collision. The ability to differentiate between these images based on spatial difference is achieved by guidance of instance masks of vehicles involved in collision. The proposed IMGA network produces an accuracy of 97.94 % in the presented vehicular collision image classification dataset.

**[Download the training and validation set](https://drive.google.com/drive/folders/1Pg2qvS2-cnWyK_IqXTnJZ_F8KGxlR5ff?usp=sharing)** of the dataset and arrange the folders in the following order for training the network.
 
    ├── vehicle_collision
    │   ├── train
    │   ├── test
    │   ├── l2_mask
    │   ├── labels.pkl

Here, the train folder contains all training images, test folder contains all the validation set images and l2_mask folder contains the vehicle mask images of all training and validation set images together.
The labels.pkl file contains the ground truth/class labels of all training and validation images in the dataset.

**[Download the test set](https://drive.google.com/drive/folders/19bq0db8jOs7cck7zrwlE0vd5trhY2GgY?usp=sharing)** of the dataset and arrange the folders in the following order to run inference on test data.

    ├── vehicle_collision
    │   ├── test
    │   ├── l2_mask
    │   ├── labels.pkl

Here, the test folder contains all the test images of the dataset and l2_mask folder contains the vehicle mask images of the test set.
The labels.pkl file contains the ground truth/class labels of all test images in the dataset.

The labels for the images are **0 (No collision class)** and **1 (Collision class)**.

# Sample Images 

![Screenshot from 2024-01-22 15-41-43](https://github.com/gmadhushan/IMGA-Vehicular-Collision/assets/62023065/34d0c112-5669-45ab-8990-90b6e68527f2)
![Screenshot from 2024-01-22 16-09-37](https://github.com/gmadhushan/IMGA-Vehicular-Collision/assets/62023065/f0f5ce8e-8884-4b07-bb70-0738dbe696be)
![Screenshot from 2024-01-22 16-11-30](https://github.com/gmadhushan/IMGA-Vehicular-Collision/assets/62023065/cd4352c4-762c-47b3-9347-1756829c0986)
![Screenshot from 2024-01-22 16-14-44](https://github.com/gmadhushan/IMGA-Vehicular-Collision/assets/62023065/78045522-1fb1-4e13-ba08-42f7f4a73911)
