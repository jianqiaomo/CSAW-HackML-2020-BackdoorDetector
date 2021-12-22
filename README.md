# CSAW-HackML-2020

## Report

### Please see our report here: [report.md](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/report/report.md)


----------


```bash
.
├── README.md
├── architecture.py
├── data
│   ├── Multi-trigger Multi-target
│   │   ├── eyebrows_poisoned_data.h5
│   │   ├── lipstick_poisoned_data.h5
│   │   └── sunglasses_poisoned_data.h5
│   ├── anonymous_1_poisoned_data.h5
│   ├── clean_test_data.h5
│   ├── clean_validation_data.h5
│   ├── data.txt
│   ├── sunglasses_poisoned_data.h5
│   └── test_images
│       ├── demo.jpeg
│       ├── demo1.jpeg
│       ├── demo10.jpeg
│       ├── demo2.jpeg
│       ├── demo3.jpeg
│       ├── demo4.jpeg
│       ├── demo5.jpeg
│       ├── demo6.jpeg
│       ├── demo7.jpeg
│       ├── demo8.jpeg
│       └── demo9.jpeg
├── eval.py
├── eval_STRIP              // by Jianqiao Mo, Tianxu Lu
│   ├── README.md
│   ├── eval_strip_anonymous_1.py
│   ├── eval_strip_anonymous_2.py
│   ├── eval_strip_multi.py
│   ├── eval_strip_sunglasses.py
│   ├── repaire.py
│   └── strip.py
├── eval_fine_prune         // by Wenjie Zhu
│   ├── README.md
│   ├── eval_fine_prune_anonymous_1.py
│   ├── eval_fine_prune_anonymous_2.py
│   ├── eval_fine_prune_multi.py
│   ├── eval_fine_prune_sunglasses.py
│   ├── evaluate_example.ipynb
│   ├── fine_prune.py
│   └── fine_prune_project.ipynb
├── lab3
├── models
│   ├── anonymous_1_bd_net.h5
│   ├── anonymous_1_bd_weights.h5
│   ├── anonymous_2_bd_net.h5
│   ├── anonymous_2_bd_weights.h5
│   ├── multi_trigger_multi_target_bd_net.h5
│   ├── multi_trigger_multi_target_bd_weights.h5
│   ├── sunglasses_bd_net.h5
│   └── sunglasses_bd_weights.h5
├── models_G
│   ├── README.md
│   ├── anonymous1_prune_net.h5
│   ├── anonymous2_prune_net.h5
│   ├── anonymous_1_STRIP_net.h5
│   ├── anonymous_2_STRIP_net.h5
│   ├── multi_trigger_multi_target_STRIP_net.h5
│   ├── multi_trigger_multi_target_prune_net.h5
│   ├── sunglasses_STRIP_net.h5
│   └── sunglasses_prune_net.h5
└── report
    ├── 20211222021029.png
    ├── 20211222021035.png
    └── report.md


```

## I. Dependencies
   1. Python 3.6.9
   2. Keras ~~2.3.1~~  2.4.3
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu ~~1.15.2~~ 2.4.1
   
## II. Validation Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5. Similarly, there are other .h5 files with poisoned data that correspond to different BadNets under models directory.

## III. Evaluating the Backdoored Model
   1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 
   2. To evaluate the backdoored model, execute `eval.py` by running:  
      `python3 eval.py <clean validation data directory> <model directory>`.
      
      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.

## IV. Evaluating the Submissions
The teams should submit a single eval.py script for each of the four BadNets provided to you. In other words, your submission should include four eval.py scripts, each corresponding to one of the four BadNets provided. YouTube face dataset has classes in range [0, 1282]. So, your eval.py script should output a class in range [0, 1283] for a test image w.r.t. a specific backdoored model. Here, output label 1283 corresponds to poisoned test image and output label in [0, 1282] corresponds to the model's prediction if the test image is not flagged as poisoned. Effectively, design your eval.py with input: a test image (in png or jpeg format), output: a class in range [0, 1283]. Output 1283 if the test image is poisoned, else, output the class in range [0,1282].

Teams should submit their solutions using GitHub. All your models (and datasets) should be uploaded to the GitHub repository. If your method relies on any dataset with large size, then upload the data to a shareable drive and provide the link to the drive in the GitHub repository. To efficiently evaluate your work, provide a README file with clear instructions on how to run the eval.py script with an example.
For example: `python3 eval_anonymous_2.py data/test_image.png`. Here, eval_anonymous_2.py is designed for anonynous_2_bd_net.h5 model. Output should be either 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned).
