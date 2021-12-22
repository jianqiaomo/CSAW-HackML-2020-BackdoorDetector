# STRIP

by _Jianqiao Mo_, _Tianxu Lu_

----------------

Reference: [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531)

## How to run:

### Save Repaire-Model

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
we also support and do testing with data_set, for example:

```angular2html
python3 eval_strip_anonymous_1.py --testset ../data/anonymous_1_poisoned_data.h5 None
```

The output "match_rate" ([0, 1]) means the matching between the "prediction labels" 
and the "truth y".
We simply use 100 images of the dataset to run testing by default.

**For example**:

we run `python3 eval_strip_anonymous_1.py --testset ../data/anonymous_1_poisoned_data.h5 None`
(poisoned data). The result shows the attack success rate `match_rate:  0.04` which means **4%** attack success rate.

we also run `python3 eval_strip_anonymous_1.py --testset ../data/clean_test_data.h5 None`
(clean data). The result shows the accuracy on clean data set: `match_rate:  0.94` which means **94%** accuracy.
The result is promising.