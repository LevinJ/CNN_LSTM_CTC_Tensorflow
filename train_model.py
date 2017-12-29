import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import cnn_lstm_otc_ocr
import helper
from preparedata import PrepareData
import math


data_prep = PrepareData()
num_epochs = 25
save_epochs = 5 # save every save_epochs epochs
validation_steps = 500
checkpoint_dir = './checkpoint'
batch_size = 40
log_dir = './log'
restore = False
def train():
    model = cnn_lstm_otc_ocr.LSTMOCR('train')
    model.build_graph()
    train_feeder, num_train_samples = data_prep.input_batch_generator('train', is_training=True, batch_size = batch_size)
    print('get image: ', num_train_samples)
   
    num_batches_per_epoch = int(math.ceil(num_train_samples / float(batch_size)))
    
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        if restore:
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        for cur_epoch in range(num_epochs):
            # the tracing part
            for cur_batch in range(num_batches_per_epoch):
               
                batch_time = time.time()
                batch_inputs, batch_labels, _ = next(train_feeder)
                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels}

                loss, step, _ = sess.run([model.cost, model.global_step, model.train_op], feed)
                
                if step % 100 == 0:
                    print('{}/{}:{},loss={}, time={}'.format(step, cur_epoch, num_epochs, loss, time.time() - batch_time))

                # monitor trainig process
                if step % validation_steps == 0 or (cur_epoch == num_epochs-1 and cur_batch == num_batches_per_epoch-1):
                    
                    batch_inputs, batch_labels, _ = next(train_feeder)
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels}
                    summary_str = sess.run(model.merged_summay,feed)
                    train_writer.add_summary(summary_str, step)
                    
            # save the checkpoint once very few epoochs
            if (cur_epoch % save_epochs == 0) or (cur_epoch == num_epochs-1):
                if not os.path.isdir(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                print('save the checkpoint of step {}'.format(step))
                saver.save(sess, os.path.join(checkpoint_dir, 'ocr-model'), global_step=step)

                    


if __name__ == '__main__':
    train()
