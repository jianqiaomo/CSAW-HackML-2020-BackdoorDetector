## ML security fine prune部分
1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 
2. To evaluate the repaired model, for example execute `eval_fine_prune_sunglasses.py` by running:  
   `python3 eval_fine_prune_sunglasses.py <clean validation data directory> `.

