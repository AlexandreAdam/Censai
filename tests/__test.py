import tensorflow as tf
from censai.utils import _int64_feature
import glob, os

# Test the logic of how we build the dataset from multiple source, and make sure each iterations through the dataset is different

for i in range(20):
    data = tf.data.Dataset.range(i * 1000, i * 1000 + 1000)
    filename = f"../data/test{i}.tfrecords"
    with tf.io.TFRecordWriter(filename, ) as writer:
        for x in data.as_numpy_iterator():
            features = {
                "x": _int64_feature(x),
            }
            record = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
            writer.write(record)


files = glob.glob(os.path.join("../data/", "*.tfrecords"))
datasets = []
for i in range(len(files)//2):
    _files = files[i*2:i*2+2]
    # Read concurrently from multiple records
    _files = tf.data.Dataset.from_tensor_slices(_files).shuffle(len(files), reshuffle_each_iteration=True)
    dataset = _files.interleave(lambda x: tf.data.TFRecordDataset(x), block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    datasets.append(dataset.shuffle(1000, reshuffle_each_iteration=True))
dataset = tf.data.experimental.sample_from_datasets(datasets, weights=None)

def decode(record_bytes):
    example = tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        features={
            'x': tf.io.FixedLenFeature([], tf.int64),
        })
    return example['x']

dataset = dataset.map(decode)
recovered_data = []
for x in dataset.as_numpy_iterator():
    recovered_data.append(x)
a = set(recovered_data)
true = set(range(20000)) # should be empty set if we did everything right
print(true.difference(a))

recovered_data2 = []
for x in dataset.as_numpy_iterator():
    recovered_data2.append(x)

diff = [a - b for a, b in zip(recovered_data, recovered_data2)]
print(diff) # should be different from zero most of the time since reshuffling occur at each calls
