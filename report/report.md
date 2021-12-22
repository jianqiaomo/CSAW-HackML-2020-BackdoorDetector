# ML for cybersecurity Project Report

* Team member and distribution: 
  * Jianqiao Mo: STRIP and report
  * Tianxu Lu: STRIP and report
  * Wenjie Zhu: Fine-pruning
* Github Repo: https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector

## Project Goal and Introduction

The project is to design a backdoor detector for BadNets trained on the YouTube Face dataset. An output G should be created as a "repaired" BadNet which has N+1 classes. 

1. Output the correct class if the test input is clean. The correct class will be in [1,N].
2. Output class N+1 if the input is backdoored.

To achieve this Goal,  we tried two different methods from different aspects. 

1. By using Fine-Pruning method, we will prune neurons in the model and retrain using clean validation dataset to get a new model B' and new model G. 
2. By using STRIP algorithm, we can fix the output without changing the structure of BadNet, instead we keep the output when we detect the image is clean, and we change the output when we detect that the image is poisoned. 

## Option 1: Fine-Pruning

Please see [README.md](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/eval_fine_prune/README.md) 
to run the script.

The first method we tried is Fine-Pruning method which is introduced in class and can be found more in the paper "Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks". 

Like what we have done in lab3 but one step further, we first prune the network poisoned by sunglasses and then fine-tune the model via cross validation using clean validation dataset. While we are fine-tuning, we use clean test dataset and sunglasses poisoned dataset to evaluate the model each 2 epochs. We can get the result as below:

```
Epochs=2 (clean data) - pruned test accuracy: 0.9372564554214478
Epochs=2 (poisoned data) - attack success rate: 0.6111457347869873
Epochs=4 (clean data) - pruned test accuracy: 0.913250207901001
Epochs=4 (poisoned data) - attack success rate: 0.6320343017578125
Epochs=6 (clean data) - pruned test accuracy: 0.9178487658500671
Epochs=6 (poisoned data) - attack success rate: 0.1637568175792694
Epochs=8 (clean data) - pruned test accuracy: 0.8950116634368896
Epochs=8 (poisoned data) - attack success rate: 0.1191738098859787
Epochs=10 (clean data) - pruned test accuracy: 0.9208885431289673
Epochs=10 (poisoned data) - attack success rate: 0.05151987448334694
```

*Please refer to [fine_prune_project.ipynb](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/eval_fine_prune/fine_prune_project.ipynb)
for the fine-pruning detail.

We can see from the result above that, we can reduce the attack success rate epoch by epoch and get a clean classification accuracy at the end of training.

However, when we tried to use this method to fix other BadNet which poisoned by other datasets. We cannot get the attack success rate decrease near to 0, with the clean classification accuracy larger than 90%. But if we fine-tune the model with more epochs, we may get the attack success rate nearly to 0 with clean classification rate keeping larger than 80%. Therefore, when the situation become more complicated, the model is attacked by multiple triggers, maybe there are other ways to reach the goal better than just using fine pruning method. Therefore, we tried another algorithm called STRIP. 

## Option 2: STRIP

`detect_trojan` in `strip.py` can detect whether the input is poisoned or not for single input and a batch of inputs respectively. If the input is poisoned, label N will be returned. If the input is clean, label between 0 to (N-1) will be returned.

The `eval_strip_[badnet_name].py` is script to evaluate.

Please see [README.md](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/eval_STRIP/README.md) 
to run the script.



### 1 Principle

After we analyze the data in the different datasets, we can get the conclusion that the poisoned data is all poisoned by a little trigger as sunglasses, little marks. The attacker's aim is to just add a little mark to the images, any image we input will be classified as a class we already have. Although the mark is small, this little perturbation can be stronger than any other perturbations. 

As we have the conclusion above, we can know that an abnormal and suspicious behavior that the predictions of all poisoned inputs will always falling into the attacker's targeted class can be detected. If the inputs are not poisoned, when we add an big perturbation to the input, the prediction can be different from before. By using this suspicious behavior, we can add one strong perturbations ourselves to detect if the input is poisoned. If the input is poisoned, our perturbations will not be useful to change the prediction. Instead if the input is not poisoned, our perturbations will change the prediction of the model.

Therefore, in the STRIP algorithm, we choose to blend clean input with other clean inputs. This can be a very strong perturbation to the input. If the input is clean, the prediction of the model will be very different from its original prediction. But if the input is poisoned, the perturbation is not strong enough to change the prediction of the model. 

### 2 Algorithm

#### 2.1 Entropy

The Shannon Entropy can be very useful to represent the randomness of the predicted classes of all perturbed inputs corresponding to the given incoming input x. The n_th perturbed input's entropy can be written as below:
$$
H_n = -\sum^M_{i=1}y_i\times\log_2{y_i}
$$
We can get the sum of all N perturbed inputs as below:
$$
H_{sum} = \sum^N_{n=1}H_n
$$
If the sum is higher, the probability that the input x is a poisoned input will be higher. We can normalize the entropy as below:
$$
H = \frac{1}{N}H_{sum}
$$
The H is regarded as the entropy of an incoming input x. It serves as an indicator whether the incoming input x is poisoned or not. 

#### 2.2 Algorithm

To using STRIP in practice, the principle above can be transformed into the algorithm below:

![](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/report/20211222021029.png)

* x is the input
* D_test is the dataset we have
* F_theta() is the BadNet model. 
* z is the prediction of the input using model
* D_p is the perturbation set which is formed by determining whether input x is trojaned or not based on the observation on predicted classes to all N perturbed inputs. 
* The judgement will be based on the entropy which can be used to measure the randomness of the prediction. The entropy formulas is introduced above. 
* Here is a figure of the whole process of the STRIP algorithm. 

![](https://github.com/jianqiaomo/CSAW-HackML-2020-BackdoorDetector/blob/master/report/20211222021035.png)

# References

1. Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg, Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks, https://arxiv.org/abs/1805.12185
2. Yansong Gao, Chang Xu, Derui Wang, Shiping Chen, Damith C.Ranasinghe, Surya Nepal: STRIP: a defense against trojan attacks on deep neural networks, https://arxiv.org/abs/1610.02391
