from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

from my_model import MyModel, DataLoader
import time
import logging
import sys
import math

import tensorflow as tf 

logging.basicConfig(format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
					level = logging.DEBUG, 
					stream = sys.stdout)

MODEL_PATH = './model/model.ckpt'
SUMMARY_PATH = './logs/'

NUM_CLASSES = ord('z') - ord('a') + 1 + 1 + 1
NUM_EPOCHS = 200
LST_NUM_HIDDEN = [100, 100, 200, 200]
NUM_LAYERS = 4
BATCH_SIZE = 4
INITIAL_LEARNING_RATE = 1e-2
MOMENTUM = 0.9

def main(argv):
	data_loader = DataLoader('../data/only_folder', BATCH_SIZE)

	with tf.device('/cpu:0'):
		config = tf.ConfigProto()
		graph = tf.Graph()

		with graph.as_default():
			logging.debug('Starting new TensorFlow graph.')
			#======================TODO========================
			inputs_placeholder = tf.placeholder(tf.int32, [None, None, NUM_FEATURES])
			labels = tf.sparse_placeholder(tf.int32)
			seq_len = tf.placeholder(tf.int32, [None])
			phase_placeholder = tf.placeholder(tf.string)
			#==============================================

			model = MyModel(LST_NUM_HIDDEN, NUM_CLASSES, BATCH_SIZE)

			logits = model.forward(inpt, phase_placeholder)
			cost = model.ctc_loss(labels, logits, seq_len)

			optim = tf.train.MomentumOptimizer(INITIAL_LEARNING_RATE, 0.9).minimize(cost)
			decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(logits, seq_len)

			label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

		with tf.Session(config = config, graph = graph) as sess:
			logging.debug('Starting TensorFlow session.')

			saver = tf.train.Saver()

			merged_summary = tf.summary.merge_all()
			summary_writer = tf.summary.FileWriter(SUMMARY_PATH, tf.get_default_graph())

			tf.global_variables_initializer().run()

			train_num = data_loader.get_train_size()
			validation_num = data_loader.get_valid_size()

			if train_num <= 0:
				logging.error('There are no trainig example')
				return

			num_batches_per_epoch = math.ceil(train_num/BATCH_SIZE)

			for phase in ['normal', 'sparse']:
				for current_epoch in range(NUMBER_EPOCHS):
					start_time = time.time()

					train_cost = 0
					train_label_error_rate = 0

					for step in range(num_batches_per_epoch):
						batch = data_loader.get_train_batch()
						feed = {inputs : batch[0],
								labels : batch[1],
								seq_len : batch[2],
								phase : phase}

						batch_cost, _, summary = sess.run([cost, optim, merged_summary], feed)
						
						train_cost += batch_cost * BATCH_SIZE
						train_label_error_rate += sess.run(label_error_rate, feed) * BATCH_SIZE

						summary_writer.add_summary(summary, current_epoch * num_batches_per_epoch + step)

				train_cost /= train_num
				train_label_error_rate /= train_num

				valid = data_loader.get_valid()
				valid_feed = {inputs : valid[0],
							  labels : valid[1],
							  seq_len : valid[2]}

				validation_cost, validation_label_error_rate = sess.run([cost, label_error_rate], valid_feed)
				validation_cost /= validation_num
				validation_label_error_rate /= validation_num

				logging.info('Epoch %d/%d (time: %.3f s)', current_epoch + 1, NUM_EPOCHS, time.time() - start_time)
				logging.info('Train cost: %.3f, train label error rate: %.3f', train_cost, train_label_error_rate)
				logging.info('Validation cost: %.3f, validation label error rate: %.3f', train_cost, train_label_error_rate)


		# test_feed = {inputs : test_inputs,
		# 			 seq_len : test_seq_len}

		# decoded_outputs = sess.run(decoded[0], test_feed)
		# dense_decoded = tf.sparse_tuples_from_sequences(decoded_outputs, default_value = -1).eval(session = sess)
		# test_num = test_texts.shape[0]

		# for i, sequence in enumerate(dense_decoded):
		# 	sequence = [s for s in sequence if s != -1]
		# 	decoded_text = utils.sequence_decoder(sequence)

		# 	logging.info('Sequence %d/%d', i + 1, test_num)
		# 	logging.info('Original: \n%s', test_texts[i])
		# 	logging.info('Decoded: \n %s', decoded_text)

		save_path = saver.save(sess, MODEL_PATH)
		logging.info('Model saved in file: %s', save_path)

if __name__ == '__main__':
	tf.app.run()



