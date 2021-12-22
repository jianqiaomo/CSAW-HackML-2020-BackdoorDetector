import fine_prune
import sys

if __name__ == '__main__':
    
    model_path = '../models_G/multi_trigger_multi_target_prune_net.h5'
    img_path = str(sys.argv[1])
    
    fine_prune.eval(model_path, img_path)
    
    pass
