import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.utils import shuffle
from ocr_datasets import OCRDatasets
import cv2
import utils
import math

FLAGS = utils.FLAGS




class PrepareData(OCRDatasets):
    def __init__(self):
        OCRDatasets.__init__(self)
        
        return
    def sparse_tuple_from_label(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []
    
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)
    
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    
        return indices, values, shape
    def __preprocess_samples(self, samples):
        batch_inputs = []
        batch_labels = []
        for sample in samples:
            image_name = sample
            im = cv2.imread(image_name, 0).astype(np.float32)/255.
            # resize to same height, different width will consume time on padding
            # im = cv2.resize(im, (image_width, image_height))
            im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
            batch_inputs.append(im)
    
            # batch_inputs is named as /.../<folder>/00000_abcd.png
            code = image_name.split('/')[-1].split('_')[1].split('.')[0]
            code = [utils.SPACE_INDEX if code == utils.SPACE_TOKEN else utils.encode_maps[c] for c in list(code)]
            batch_labels.append(code)
        batch_labels_sparse = self.sparse_tuple_from_label(batch_labels)
            
        return batch_inputs, batch_labels_sparse, batch_labels
    
    def __generator(self, samples, batch_size):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            samples = shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                yield self.__preprocess_samples(batch_samples)

    
    def input_batch_generator(self, split_name, is_training=True, batch_size=32):
        samples = self.get_samples(split_name) 
        
        gen = self.__generator(samples, batch_size)
       
        return gen, len(samples)
   
    def run(self):
        batch_size = 32
        split_name = 'val'
        generator, dataset_size = self.input_batch_generator(split_name, is_training=False, batch_size=batch_size)
        num_batches_per_epoch = int(math.ceil(dataset_size / float(batch_size)))
        for _ in range(num_batches_per_epoch):
            images, labels, _ = next(generator)
            print(labels)
        return
    
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()