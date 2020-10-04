import tensorflow as tf

slim = tf.contrib.slim

def create_network(state,inputs, is_training, scope="win37_dep9", reuse=False):


	with tf.variable_scope(scope, reuse=reuse):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
			normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):


			for i in state:
				if i[0]=='none':
					state.remove(i)
			f=state.pop(0)
			if f[0]=='conv2d':
				net=slim.conv2d(inputs,64,[f[3],f[3]],padding=f[2],scope='conv1')
			elif f[0]=='batch':
				net=slim.batch_norm(inputs,is_training=is_training)
			else:
				raise ValueError('not in category1')
			for i in state:
				if i[0] == 'conv2d':
					net = slim.conv2d(net, i[1], [i[3], i[3]], padding=i[2])
				elif i[0] == 'batch':
					net = slim.batch_norm(net, is_training=is_training)
				elif i[0]=='none':
					pass
				else:
					raise ValueError('not in category')

			net = slim.batch_norm(net, is_training=is_training)	

	return net



