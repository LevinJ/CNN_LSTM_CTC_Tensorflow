import os
import re
import argparse


g_modellogdir = '.'

class RunAllCheckpoints(object):
    def __init__(self):
        
        
        return
    
    
    
    def get_all_checkpoints(self,checkpoint_path):
        
        with open("{}/{}/checkpoint".format(g_modellogdir, self.checkpoint_path)) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        checkpoints = []
        for line in content:
            m = re.search('all_model_checkpoint_paths:(.*)ocr-model-(.*)"', line)
            if m:
                num = m.group(2)
                checkpoints.append(num)
        min_step = 0
        step = 100
        last_step = min_step
        sel_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint = int(checkpoint)
            if checkpoint < min_step:
                continue
            if checkpoint == int(checkpoints[-1]):
                #the last checkpoint always get selected
                sel_checkpoints.append(checkpoint)
                continue
            if checkpoint >= last_step:
                sel_checkpoints.append(checkpoint)
                last_step = last_step + step
        if self.check_only_latest:
            #if we only want to evluate the latest checkpoints
            sel_checkpoints = [sel_checkpoints[-1]]
        return sel_checkpoints
    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-l', '--latest',  help='evaluate only the latest checkpoints',  action='store_true')
        parser.add_argument('-c', '--checkpoint_path',  help='which checkpoint(directory) to use',  default="checkpoint")
        args = parser.parse_args()
        
        self.checkpoint_path = args.checkpoint_path
        self.check_only_latest = args.latest
            
        return
    def run_all_checkpoints(self):
        
        sel_checkpoints = self.get_all_checkpoints(self.checkpoint_path)
        #for tine tuning checkpoint path, we can skip the first chckpoint since it's already calcuated 
        if self.checkpoint_path == 'logs/finetune' and (not self.check_only_latest):
            sel_checkpoints = sel_checkpoints[1:]
                
        
        for checkpoint in sel_checkpoints:
            for split_name in ["train", "eval"]:
                
                checkpoint_file ="{}/{}/ocr-model-{}".format(g_modellogdir,  self.checkpoint_path, checkpoint)
#                 print("checkpoint {}, {} data".format(checkpoint_file, split_name))
                
                cmd_str = "python ./eval_model.py "
                
                cmd_str = '{} -s "{}" -c "{}"'.format(cmd_str, split_name, checkpoint_file)
                os.system(cmd_str)
            
        return
    
    
    def run(self):
        self.parse_param()
        self.run_all_checkpoints()
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= RunAllCheckpoints()
    obj.run()