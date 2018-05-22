import tensorflow as tf
import os
import random
import skimage.io as io
import skimage.transform as transform
import numpy as np




class MyModel():
	def __init__(self, lst_num_hid, num_classes, batch_size, k):
		self.lst_num_hid = lst_num_hid
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.k = k
		self.weight = {}
		self.bias = {}
		self.init_weights_bias()

	def init_weights_bias():
		for layer_number in range(1, 5):
			self.weight['w_fw_{}'.format(layer_number)] = tf.Variable(tf.truncated_normal([num_hidden,num_hidden], sttdev = 0.1), name = 'w_fw_{}'.format(layer_number))
			self.weight['w_bw_{}'.format(layer_number)] = tf.Variable(tf.truncated_normal([num_hidden,num_hidden], sttdev = 0.1), name = 'w_bw_{}'.format(layer_number))
			self.bias['b{}'.format(layer_number)] = tf.Variable(tf.constant(0., shape = [num_hidden]), name = 'b{}'.format(layer_number))

	def lstm_cell(self, num_hidden):
		return tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)

	def layer(self, inpt, layer_number, phase):
		num_hidden = self.lst_num_hid[layer_number - 1]
		output_fw_bw, _ = tf.bidirectional_dynamic_rnn(self.lstm_cell(num_hidden), self.lstm_cell(num_hidden), inpt)
		
		fw_matrix = self.weight['w_fw_{}'.format(layer_number)]
		bw_matrix = self.weight['w_bw_{}'.format(layer_number)]
		
		if phase == 'sparse':
			abs_fw_matrix = tf.map_fn(lambda x : abs(x), fw_matrix)
			abs_bw_matrix = tf.map_fn(lambda x : abs(x), bw_matrix)

			lambda_fw = tf.nn.top_k(abs_fw_matrix, self.k)[-1]
			lambda_bw = tf.nn.top_k(abs_bw_matrix, self.k)[-1]

			mask_fw = tf.map_fn(lambda x : 1 if x > lambda_fw else 0, abs_fw_matrix)
			mask_bw = tf.map_fn(lambda x : 1 if x > lambda_bw else 0, abs_bw_matrix)

			fw_matrix = tf.matmul(fw_matrix, mask_fw)
			bw_matrix = tf.matmul(bw_matrix, mask_bw)


		output = tf.matmul(output_fw_bw[0], fw_matrix) + tf.matmul(output_fw_bw[1], bw_matrix) + self.bias['b{}'.format(layer_number)]

		return output

	def residual_block(self, inpt, block_number):
		layer_number = block_number * 2
		F_y = self.layer(inpt, layer_number)
		output = F_y + inpt

		return output

	def affine_transform(self, inpt):
		num_hidden = self.lst_num_hid[-1]
		self.weights['w_a'] = tf.Variable(tf.truncated_normal([num_hidden, self.num_classes], sttdev = 0.1), name = 'w_a')
		self.bias['b_a'] = tf.Variable(tf.constant(0., shape = [self.num_classes]), name = 'b_a')

		logits = tf.matmul(inpt, self.weights['w_a']) + self.bias['b_a']
		logits = tf.reshape(logits, [self.batch_size, -1, self.num_classes])
		logits = tf.transpose(logits, (1, 0, 2)) #TODO

		return logits

	def ctc_loss(self, labels, logits, seq_len):
		with tf.name_scope('loss'):
			loss = tf.nn.ctc_loss(labels, logits, seq_len)
			cost = tf.reduce_mean(loss)
			tf.summary.scalar('loss', cost)

			return cost

	def forward(inpt, phase):
		layer1 = self.layer(inpt,   1, phase)
		layer2 = self.residual_block(layer1, 1, phase)
		layer3 = self.layer(layer2, 3, phase)
		layer4 = self.residual_block(layer3, 2, phase)

		return layer4

class DataLoader():
	def __init__(self, path_to_data, batch_size, valid_size = 0.3):
		self.path_to_data = path_to_data
		self.batch_size = batch_size
		self.last_batch_index = 0
		
		self.lst_pathes = []
		self.fill_lst()

		self.train_path = self.lst_pathes[:int(len(self.lst_pathes)*(1-valid_size))]
		self.valid_path = self.lst_pathes[int(len(self.lst_pathes)*(1- valid_size)):]

	def fill_lst(self):
# 		paths = []
# 		for path, _, files in os.walk(self.path_to_data):
# 			for img in files:
# 				try:
# 					height = io.imread(path + '/' + img).shape[0] 
# 					if height > 25 and height < 100:
# 						paths.append((path,img))
# 				except:
# 					continue
		
# 		random.shuffle(self.lst_pathes)
		
# 		labels = []
# 		with open('words.txt', 'r') as f:
# 			for i,line in enumerate(f):
# 				if i >= 18:
# 					lb = line.split()
# 					labels.append((lb[0], lb[-1].lower()))
		
# 		for p in paths:
# 			file_name = p[1].split('.')[0]
# 			for l in labels:
# 				if file_name == l[0]:
# 					self.lst_pathes.append((p[0]+'/' + p[1], l[1]))
#                     break
		with open('path_label.txt', 'r') as file:
			for line in file:
				self.lst_pathes.append(line)
		
	def get_train_batch(self):
		if self.last_batch_index + self.batch_size < len(self.train_path):
			indexes = self.train_path[self.last_batch_index : self.last_batch_index + self.batch_size]
		else:
			indexes = self.train_path[self.last_batch_index : ]
		self.last_batch_index = self.last_batch_index + self.batch_size
		X = []
		y = []
		seq_len = []
		for ind in indexes:
			path = ind.split()[0]
			label = ind.split()[1]
			# TODO preprocessing of image
			image = io.imread(path, as_gray = True)
			seq_len.append(image.shape[1])
			image = transform.resize(image, (100,480, 3))
			X.append(np.expand_dims(image, 0))
			lst = list(label)
			alphabet = list("n/&b:t*l0ezg;yr-c5.3dvpow72qmxj4f+986#)a,h!?uki1s")
			num_lst = []
			for el in lst:
				num_lst.append(alphabet.index(el))
			y_out = np.zeros((len(lst),len(alphabet)))
			y_out[np.arange(len(lst)),num_lst] = 1
			y.append(y_out)
			
		return np.vstack(X), y, seq_len

	def get_valid(self):

		X = []
		y = []

		for ind in self.valid_path:
			path = ind[0]
			label = ind[1]
			image = io.imread(path, as_gray = True)
			sequence_length = image.shape[1]
			image = transform.resize(image, (100,480, 3))
			X.append(np.expand_dims(image, 0))
			y.append(label)

		return np.vstack(X), np.asarray(y), sequence_length

	def get_train_size(self):
		return len(self.train_path)

	def get_valid_size(self):
		return len(self.valid_path)