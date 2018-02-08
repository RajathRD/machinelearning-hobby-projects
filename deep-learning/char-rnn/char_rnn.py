import tensorflow as tf


import numpy as np
import sys 

# data I/O
data = open(sys.argv[1], 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

lstm_size = 100
number_of_layers = 2
seq_length = 10
batch_size = 1


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def rnn(stacked_lstm,inputs,initial_state):
	outputs = []
	states = []

	state = initial_state
	for input_ in inputs:
		output, state = stacked_lstm(input_, state)
		outputs.append(output)
		states.append(state)

	return outputs,states

def sample(state, seed_ix, n):
	""" 
	sample a sequence of integers from the model 
	state is memory state, seed_ix is seed letter for first time step
	"""
	ixes = []
	tf.get_variable_scope().reuse_variables()
	embeds = tf.nn.embedding_lookup(embedding, [[seed_ix]])

	for t in xrange(n):
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(embeds,1, 1)]
		outputs, states = rnn(stacked_lstm, inputs, state)
		output = tf.reshape(tf.concat(outputs,1), [-1, lstm_size])
		logits = tf.nn.xw_plus_b(output,tf.get_variable("softmax_w", [lstm_size, vocab_size]),tf.get_variable("softmax_b", [vocab_size]))
		probs = tf.nn.softmax(logits)

		p = sess.run([probs])[0]

		ix = np.random.choice(range(vocab_size), p=p.ravel())
		ixes.append(ix)

		embeds = tf.nn.embedding_lookup(embedding,[[ix]])
		state = states[-1]
	return ixes


input_data = tf.placeholder(tf.int32, [batch_size,seq_length])
targets= tf.placeholder(tf.int32, [batch_size, seq_length])

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers, state_is_tuple=False)

initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)



embedding = tf.get_variable("embedding", [vocab_size, lstm_size])
inputs = tf.nn.embedding_lookup(embedding, input_data)

inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, seq_length, 1)]

outputs,states = rnn(stacked_lstm,inputs,state)

output = tf.reshape(tf.concat(outputs,1), [-1, lstm_size])
logits = tf.nn.xw_plus_b(output,tf.get_variable("softmax_w", [lstm_size, vocab_size]),tf.get_variable("softmax_b", [vocab_size]))


loss = tf.reduce_mean(tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[tf.reshape(targets, [-1])],[tf.ones([batch_size * seq_length])],vocab_size))

#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.control_dependencies([loss]):
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train_step = optimizer.apply_gradients(zip(grads, tvars))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n,p = 0,0

while True:
	# prepare inputs (we're sweeping from left to right in steps seq_length long)
	if p+seq_length+1 >= len(data) or n == 0: 
		initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
		p = 0 # go from start of data
	
	input_data_ = np.array([char_to_ix[ch] for ch in data[p:p+seq_length]]).reshape(1,-1)
	targets_ = np.array([char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]).reshape(1,-1)

	loss_val,_ = sess.run([loss,train_step],feed_dict={input_data: input_data_, targets: targets_})	

	if n%1000 == 0:
		print 'Iteration completed: ',n,'Loss: ',loss_val
		print '-'*80
		state = stacked_lstm.zero_state(batch_size, tf.float32)
		ixes = sample(state,input_data_[0,0],200)
		outStr = ''.join([ix_to_char[ix] for ix in ixes])
		print outStr


	p += seq_length # move data pointer
	n += 1 # iteration counter 




