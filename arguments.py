from omegaconf import OmegaConf
import argparse
import os 
import yaml 
def parser():
    parser = argparse.ArgumentParser(description='Active Learning - Benchmark')
    parser.add_argument('--default_setting', type=str, default=None, help='default config file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
        )
    
    # load default config 
    args = parser.parse_args()
    cfg = OmegaConf.load(args.default_setting)
    
    # Update experiment name 
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        try:
            OmegaConf.update(cfg, k, eval(v), merge=True)
        except:
            OmegaConf.update(cfg, k, v, merge=True)
            
    print(OmegaConf.to_yaml(cfg))
    
    # Update output save dir 
    save_dir = os.path.join(cfg.BASE.save_dir,cfg.DATA.dataset_name,cfg.DATA.class_name,cfg.BASE.exp_name)
    os.makedirs(save_dir,exist_ok=True)
    cfg.BASE.save_dir = save_dir    
    return cfg 