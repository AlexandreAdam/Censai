import tensorflow as tf
def run(data_folder, output_list, input_dict):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(data_folder+'epoch0.ckpt.meta')
        new_saver.restore(sess, data_folder+'epoch0.ckpt')

        result = sess.run(output_list, input_dict)

    sess.close()
    return result