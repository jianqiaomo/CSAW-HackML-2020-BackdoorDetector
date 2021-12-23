# Fine-pruning Exprement Report

#### by _Wenjie Zhu_, 

{wz2140}@nyu.edu

Reference: [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://arxiv.org/abs/1805.12185)

## Fine-Pruning Defense

I inplement Fine-Pruning Defense code in [fine_prune_project.ipynb](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/eval_fine_prune/fine_prune_project.ipynb). 
I inplement evaluation code in [evaluate_example.ipynb](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/eval_fine_prune/evaluate_example.ipynb)

### Setting

The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. 
This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 

### Prune

The `apply_pruning_to_dense` function will `prune` the layer `conv_3`
by using the method (`tfmot.sparsity.keras.prune_low_magnitude`).

### Fine-tuning+Cross Validation

We use `fine-tuning` function to  training `Badnet` on `Clean_dataset`. We use `cross validation`method to validate our results.

### Result

The result is shown in [fine_prune_project.ipynb](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/eval_fine_prune/fine_prune_project.ipynb). 



## How to Evaluate:

To evaluate the repaired model, there are four evaluation python files in eval_fine_prune directory: eval_fine_prune_anonymous_1.py, eval_fine_prune_anonymous_2.py,eval_fine_prune_multi.py, eval_fine_prune_sunglasses.py. for example execute `eval_fine_prune_sunglasses.py` by running: `python3 eval_fine_prune_sunglasses.py <img_path>`. You can use data/test_images as your <img_path>.

  Our evalution results are shown at evaluate_example.ipynb

### Result

The result is shown in [evaluate_example.ipynb](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/eval_fine_prune/evaluate_example.ipynb)

