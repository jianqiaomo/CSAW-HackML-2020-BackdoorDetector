# STRIP

#### by _Jianqiao Mo_, _Tianxu Lu_

{jq.mo, tl3173}@nyu.edu

----------------

Reference: [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531)

## - Backdoor Detection

We inplement our code in [strip.py](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/eval_STRIP/strip.py). 
We have several methods in the `class RepairedNetG()`.

### entropy_layer

According to STRIP algorithm, we need to compute the entropy, which shows the perturbation randomness and indicates 
whether their is backdoor or not.  
The `entropy_layer` function will `sort_samples` of the clean validation set by classes for the coming backdoor detection
and entropy computation (`entropy`).

### _ init _

**Repaire**: We use _BadNet_ to initialize the model with _CleanData_ set and a hyperparameter _boundary_ (we set it to 0.5). 
We `save` this model to `./models_G`.

**Load**: We can load the models from `./models_G` to init() the model. Then you can run 
`predict_img` or `predict`. 

### detect_trojan

According to the algorithm, we first need to run _perturbation_ above the image and compute 
the randomness (entropy). Decision will be made after comparing the overall entropy with 
the threshold (boundary). 

Output should be either 1283 (if test_image.png is poisoned) or one class in 
range [0, 1282] (if test_image.png is not poisoned).

### predict, predict_img

`predict` takes one image (path) as input and return the classification prediction and the 
entropy value.

`predict_img` takes a batch of images (dataset) as input and return the classification prediction list and the 
entropy value list. The result will be used for accuracy or attack success rate (asr) computation.

## - How to run:

### Save Repaire-Model

(You can skip this step since we already have the models in `./models_G`.)

```angular2html
cd ./eval_STRIP
python3 repaire.py
```

The script will use the models from `./models`, and save to `./models_G` 
as "repaired models".

### Evaluate with an image

if you want to evaluate, for example, run:

```angular2html
cd ./eval_STRIP
python3 eval_strip_anonymous_1.py [img_path]
```

We suggest to run the code as above. It will return the predict label on screen.

However, 
we also support and do testing with dataset, for example:

```angular2html
python3 eval_strip_anonymous_1.py --testset ../data/anonymous_1_poisoned_data.h5 None
```

The output "match_rate" ([0, 1]) means the matching between the "prediction labels" 
and the "truth y".
We simply use 100 images of the dataset to run testing by default.

**For example**:

we run `python3 eval_strip_anonymous_1.py --testset ../data/anonymous_1_poisoned_data.h5 None`
(poisoned data). 
The result shows the attack success rate `match_rate:  0.04` which means **4%** attack success rate.

we also run `python3 eval_strip_anonymous_1.py --testset ../data/clean_test_data.h5 None`
(clean data). The result shows the accuracy on clean data set: `match_rate:  0.94` which means **94%** accuracy.


Here are other simple testing examples. 
To save time, we randomly pick 100 images of the dataset and find our the accuracy of clean dataset / asr of 
poisoned dataset.  

|               model              | accuracy(clean_test_data) |                                    asr / (poisoned dataset)                                    |
|:--------------------------------:|:-------------------------:|:----------------------------------------------------------------------------------------------:|
|         anonymous_1_STRIP        |            0.94           |                                   0.04 (anonymous_1_poisoned)                                  |
|         anonymous_2_STRIP        |            0.93           |                                    0 (anonymous_1_poisoned)                                    |
|         sunglasses_STRIP         |            0.92           |                                   0.17 (sunglasses_poisoned)                                   |
| multi_trigger_multi_target_STRIP |            0.89           | 0.25 (Multi-/eyebrows_poisoned) <br> 0.03 (Multi-/lipstick_poisoned) <br> 0 (Multi-/sunglasses_poisoned) |


<div align=center><img src="https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/report/chart.png"/></div>

As the chart shows, we can see the accuracy does not change very much comparing the first two bar. 
It means that the accuracy maintains after reparation.

However, we can see there is a significant drop of the asr in the bar chart. Although there is still some attack 
success in `sunglasses_poisoned` and `eyebrows_poisoned`, the overall result is promising —— STRIP method eliminates 
the backdoor attack effectively.