import tensorflow as tf

slim = tf.contrib.slim

def create_network(state,inputs, is_training, scope="win37_dep9", reuse=False):
	num_maps = 64
	kw = 5
	kh = 5

	with tf.variable_scope(scope, reuse=reuse):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
			normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):

			
			'''net = slim.conv2d(inputs, 32, [kh, kw], scope='conv_bn_relu1')

			
			net = slim.conv2d(net, 32, [kh, kw], scope='conv_bn_relu2')
			net = slim.repeat(net, 6, slim.conv2d, num_maps, [kh, kw], scope='conv_bn_relu3_8')'''
			'''f=state.popleft()
			if f[0]=='conv2d':
				net=slim.conv2d(inputs,f[1],[kh,kw],padding=f[2],scope='0')
			else:
				net = slim.conv2d(inputs, f[1], [kh, kw],padding=f[2] ,scope='0')
				print('else')
			for i in state:
				if i[0] == 'conv2d':
					net = slim.conv2d(net, i[1], [kh, kw],padding=i[2] ,scope=str(state.index(i)))
				else:
					net = slim.conv2d(net, i[1], [kh, kw],padding=i[2], scope=str(state.index(i)))
					print('else')'''
			'''tensors=[]
			number=0
			net = slim.conv2d(inputs, 32, [kh, kw], scope='conv_bn_relu1')
			tensors.append(net)
			for i in state:
				if i[0]=='conv2d':
					net=slim.conv2d(net,state[1],[kh,kw],padding=state[2],scope=str(number))
				elif i[0]=='batch':
					net=slim.batch_norm(net,is_training=is_training)
				else:pass





			net = slim.conv2d(net, num_maps, [kh, kw], scope='conv9', activation_fn=None, 
					normalizer_fn=None)'''
			print(type(state))
			for i in state:
				if i[0]=='none':
					state.remove(i)
			print(state)

			f=state.pop(0)
			if f[0]=='conv2d':
				net=slim.conv2d(inputs,64,[f[3],f[3]],padding=f[2],scope='conv1')
				print(net.shape)
			elif f[0]=='batch':
				net=slim.batch_norm(inputs,is_training=is_training)
				print(net.shape)
			else:
				print(f)
				raise ValueError('not in category1')
			for i in state:
				if i[0] == 'conv2d':
					net = slim.conv2d(net, i[1], [i[3], i[3]], padding=i[2])
				elif i[0] == 'batch':
					net = slim.batch_norm(net, is_training=is_training)
				elif i[0]=='none':
					pass
				else:
					print(i)
					raise ValueError('not in category')
				print(net.shape)

			net = slim.batch_norm(net, is_training=is_training)	

	return net



