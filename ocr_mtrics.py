import tensorflow as tf
from tensorflow.contrib import slim


from tensorflow.python.ops import math_ops

def character_accuracy(predictions, labels):
    """predictions and labels are of shape Batches x NUM_Digits
    """
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
      
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    acc = tf.reduce_sum(is_correct)
    return acc
def word_accuracy(predictions, labels):
    """predictions and labels are of shape Batches x NUM_Digits
    """
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    num_digits = tf.shape(labels)[1]  
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    is_correct = tf.reduce_sum(is_correct, axis = -1)
    
    is_correct = tf.equal(is_correct, num_digits)
    acc = tf.reduce_sum(math_ops.to_float(is_correct))
    return acc

def streaming_character_accuracy(predictions, labels):
    """predictions and labels are of shape Batches x NUM_Digits
    """
    return slim.metrics.streaming_accuracy(predictions, labels,name='character_acc')

def streaming_word_accuracy(predictions, labels):
    """predictions and labels are of shape Batches x NUM_Digits
    """
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    num_digits = tf.shape(labels)[1]   
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    is_correct = tf.reduce_sum(is_correct, axis = -1)
    is_correct = tf.equal(is_correct, num_digits)
    is_correct = math_ops.to_float(is_correct)
    
    return  tf.metrics.mean(is_correct,name='word_acc')