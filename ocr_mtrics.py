import tensorflow as tf
from tensorflow.contrib import slim


from tensorflow.python.ops import math_ops
def pad_pred_label(predictions, labels):
    num_digit_predictions = tf.shape(predictions)[-1]
    num_digit_labels = tf.shape(labels)[-1]
    
    paddings_mask = tf.constant([[0,0], [0,1]])
    paddings = tf.fill([2,2], tf.abs(num_digit_predictions-num_digit_labels))
    paddings  = paddings * paddings_mask
    # paddings = tf.constant([[0, 0,], [0, tf.abs(num_digit_predictions-num_digit_predictions)]])
    
    predictions = tf.cond(num_digit_predictions< num_digit_labels, lambda: tf.pad(predictions, paddings, constant_values=-1), lambda: tf.identity(predictions))
    labels = tf.cond(num_digit_labels< num_digit_predictions, lambda: tf.pad(labels, paddings, constant_values=-1), lambda: tf.identity(labels))
    return predictions, labels

def character_accuracy(predictions, labels):
    """predictions and labels are of shape Batches x NUM_Digits_Pred and Batches x NUM_Digits_Label
    """
    predictions, labels = pad_pred_label(predictions, labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
      
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    acc = tf.reduce_mean(is_correct)
    return acc
def word_accuracy(predictions, labels):
    """predictions and labels are of shape Batches x NUM_Digits_Pred and Batches x NUM_Digits_Label
    """
    predictions, labels = pad_pred_label(predictions, labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    num_digits = math_ops.to_float((tf.shape(labels)[1])) 
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    is_correct = tf.reduce_sum(is_correct, axis = -1)
    
    is_correct = tf.equal(is_correct, num_digits)
    acc = tf.reduce_mean(math_ops.to_float(is_correct))
    return acc

def streaming_character_accuracy(predictions, labels):
    """predictions and labels are of shape Batches x NUM_Digits_Pred and Batches x NUM_Digits_Label
    """
    predictions, labels = pad_pred_label(predictions, labels)
    return slim.metrics.streaming_accuracy(predictions, labels,name='character_acc')

def streaming_word_accuracy(predictions, labels):
    """predictions and labels are of shape Batches x NUM_Digits_Pred and Batches x NUM_Digits_Label
    """
    predictions, labels = pad_pred_label(predictions, labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    num_digits = math_ops.to_float((tf.shape(labels)[1]))
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    is_correct = tf.reduce_sum(is_correct, axis = -1)
    is_correct = tf.equal(is_correct, num_digits)
    is_correct = math_ops.to_float(is_correct)
    
    return  tf.metrics.mean(is_correct,name='word_acc')