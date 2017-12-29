import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import cnn_lstm_otc_ocr
import utils
import helper
from preparedata import PrepareData
FLAGS = utils.FLAGS
import math
import argparse



class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        return
    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--split_name',  help='which split of dataset to use',  default="val")
        parser.add_argument('-c', '--checkpoint_path',  help='which checkpoint to use',  default= "./checkpoint/")
        args = parser.parse_args()
        
       
        self.checkpoint_path = args.checkpoint_path
        self.split_name = args.split_name
            
        return
    def eval_model(self):
        model = cnn_lstm_otc_ocr.LSTMOCR('eval')
        model.build_graph()
    
        print('loading validation data, please wait---------------------')
        val_feeder, num_samples = self.input_batch_generator(self.split_name, is_training=False, batch_size = FLAGS.batch_size)
        print('get image: ', num_samples)
    
       
        num_batches_per_epoch = int(math.ceil(num_samples / float(FLAGS.batch_size)))
       
      
    
        
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
    
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            eval_writer = tf.summary.FileWriter(FLAGS.log_dir + '/eval', sess.graph)
            
            
            if tf.gfile.IsDirectory(self.checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
            else:
                checkpoint_file = self.checkpoint_path
            print('Evaluating checkpoint_path={}, split={}'.format(checkpoint_file, self.split_name))
           
            saver.restore(sess, checkpoint_file)
           
    

            

            for i in range(num_batches_per_epoch):
                inputs, labels, _ = next(val_feeder)
                feed = {model.inputs: inputs,
                            model.labels: labels}
                start = time.time()
                _ = sess.run(model.names_to_updates, feed)
                elapsed = time.time()
                elapsed = elapsed - start
                print('{}/{}, {:.5f} seconds.'.format(i, num_batches_per_epoch, elapsed))
                    
                # print the decode result
                
            summary_str, step = sess.run([model.merged_summay, model.global_step])
            eval_writer.add_summary(summary_str, step)
            return
    def run(self):
        self.parse_param()
        self.eval_model()
        return

       




if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.run()
