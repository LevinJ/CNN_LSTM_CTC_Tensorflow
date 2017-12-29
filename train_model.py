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

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)

data_prep = PrepareData()
def train(train_dir=None, val_dir=None, mode='train'):
    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()
    train_feeder, num_train_samples = data_prep.input_batch_generator('train', is_training=True, batch_size = FLAGS.batch_size)
    print('get image: ', num_train_samples)
   
    num_batches_per_epoch = int(math.ceil(num_train_samples / float(FLAGS.batch_size)))

    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        num_epochs = FLAGS.num_epochs
        for cur_epoch in range(num_epochs):
            # the tracing part
            for cur_batch in range(num_batches_per_epoch):
               
                batch_time = time.time()
                batch_inputs, batch_labels, _ = next(train_feeder)
                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels}

                loss, step, _ = sess.run([model.cost, model.global_step, model.train_op], feed)
                
                if (cur_batch + 1) % 100 == 0:
                    print('{}/{}:{},loss={}, time={}'.format(step, cur_epoch, num_epochs, loss, time.time() - batch_time))

                # save the checkpoint
                if step % FLAGS.save_steps == 1 or (cur_epoch == num_epochs-1 and cur_batch == num_batches_per_epoch):
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    print('save the checkpoint of step {}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                               global_step=step)

                # do validation
                if step % FLAGS.validation_steps == 0 or (cur_epoch == num_epochs-1 and cur_batch == num_batches_per_epoch):
                    
                    batch_inputs, batch_labels, _ = next(train_feeder)
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels}
                    summary_str = sess.run(model.merged_summay,feed)
                    train_writer.add_summary(summary_str, step)

                    

def main(_):
   
    train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
