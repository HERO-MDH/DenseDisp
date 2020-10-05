import tensorflow as tf
import os
import models.net_factory as nf
import numpy as np
from data_handler import Data_handler
from simanneal import Annealer
import random
import sys
from termcolor import  colored
import sqlite3
import datetime
import time
now=datetime.datetime.now()

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('num_iter', 5000, 'Total training iterations')
flags.DEFINE_string('model_dir', 'C:\\Users\\moham\\PycharmProjects\\DenseDisp\\output', 'Trained network dir')
flags.DEFINE_string('data_version', 'kitti2015', 'kitti2012 or kitti2015')
flags.DEFINE_string('data_root', 'C:\\Users\\moham\\Desktop\\Projects\\data_scene_flow\\training', 'training dataset dir')
flags.DEFINE_string('util_root', 'C:\\Users\\moham\\Desktop\\Projects\\data_scene_flow\\save_dir', 'Binary training files dir')
flags.DEFINE_string('net_type', 'win37_dep9', 'Network type: win37_dep9 pr win19_dep9')

flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
flags.DEFINE_integer('num_val_img', 40, 'number of evaluation images')
flags.DEFINE_integer('patch_size', 37, 'training patch size')
flags.DEFINE_integer('num_val_loc', 50, 'number of validation locations')
flags.DEFINE_integer('disp_range', 201, 'disparity range')
flags.DEFINE_string('phase', 'train', 'train or evaluate')

FLAGS = flags.FLAGS

np.random.seed(123)

dhandler = Data_handler(data_version=FLAGS.data_version,
                        data_root=FLAGS.data_root,
                        util_root=FLAGS.util_root,
                        num_tr_img=FLAGS.num_tr_img,
                        num_val_img=FLAGS.num_val_img,
                        num_val_loc=FLAGS.num_val_loc,
                        batch_size=FLAGS.batch_size,
                        patch_size=FLAGS.patch_size,
                        disp_range=FLAGS.disp_range)

if FLAGS.data_version == 'kitti2012':
    num_channels = 1
elif FLAGS.data_version == 'kitti2015':
    num_channels = 3
else:
    sys.exit('data_version should be either kitti2012 or kitti2015')


class SimulatedAnnealer(Annealer):
    def __init__(self, state):
        self.num = 0
        self.best=1
        self.time=0
        path=FLAGS.model_dir
        conn=sqlite3.connect(path+'\\bests.db')
        c=conn.cursor()
        c.execute('''CREATE TABLE bestss
                     (num int, arc text, acc real, t_flops real, energy real)''')
        conn.commit()
        conn.close()
        super(SimulatedAnnealer, self).__init__(state)
    #Mutation mechanism
    def move(self):


        valids=[]
        others=[]
        #The list of layers' condidates except convolutional layers with valid padding
        layers=[['conv2d',32,'same',3],['conv2d',32,'same',5],['conv2d',32,'same',7],['conv2d',32,'same',11],['batch',0,'none',0]]
        #Extract covolutional layers with valid padding
        #As we have different mutation mechanism for Convolutional valid layers and others, then we need to find Convolutional layers with
        #valid layers place before mutation
        for i in self.state:
            if i[0]=='conv2d':
                if i[2]=='valid':
                    if i[3]>3:
                        valids.append(self.state.index(i))
                    else:
                        pass
                else:
                    others.append(self.state.index(i))
            else:
                others.append(self.state.index(i))

        #As if we ingnore the convolutional layers with valid padding which have less than 3*3 filter size,
        #then there is some networks with Zero size valids list. So we need to define that as a condition
        if len(valids)!=0:
            #Valid layers mutation
            if random.random()<0.2:
                a=random.choice(valids)
                b=random.randrange(3,self.state[a][3],2)
                temp=self.state[a][3]
                self.state[a][3]=b
                self.state.append(['conv2d',32,'valid',(temp-b)+1])

            #None-Valid layers mutation
            elif random.random()>=0.2 and random.random()<0.7:
                a=random.choice(others)
                be=random.choice(layers)
                self.state.remove(self.state[a])
                self.state.append(be)
            else:
                #Swape two layers
                a=random.randint(0,len(self.state)-1)
                b=random.randint(0,len(self.state)-1)
                temp1=self.state[a]
                self.state[a]=self.state[b]
                self.state[b]=temp1
        else:
            #The condition that we don't have any valid layer
            if random.random()<0.5:
                a = random.choice(others)
                be = random.choice(layers)
                self.state.remove(self.state[a])
                self.state.append(be)
            else:
                a = random.randint(0, len(self.state) - 1)
                b = random.randint(0, len(self.state) - 1)
                temp1 = self.state[a]
                self.state[a] = self.state[b]
                self.state[b] = temp1
        kernel_sum=0
        num_node=0
        #Check that the size of valid layers are consistant with the constraints
        for i in self.state:
            if i[0]=='conv2d':
                if i[2]=='valid':
                    kernel_sum+=i[3]
                    num_node+=1
        ex=kernel_sum-num_node
        if ex!=36:
            self.state.append(['conv2d',32,'valid',(36-ex)+1])
        #Call the energy mechanism
        return self.energy()
    #Computing model energy which related to the FLOPs and Accuracy
    def energy(self):
        kernel_sum=0
        num_node=0
        #Again checking the consistency to make sure
        for i in self.state:
            if i[0]=='conv2d':
                if i[2]=='valid':
                    kernel_sum+=i[3]
                    num_node+=1
        ex=kernel_sum-num_node
        if ex!=36:
            print('2',self.state)
            raise ValueError('this is not appropriante')

        #Computing FLOPs for the model
        t_flops=train(self.state, self.num)
        #Estimating the accuracy
        acc = evaluate(self.state,self.num)
        #The mechanism of Computing Energy by Flops and accuarcy
        acc=acc/100
        flops=(18700000-t_flops)/18700000
        e=0.44*(1-acc)+0.56*(1-flops)
        if acc>0.5:
            e+=0.1
        statea=str(self.state)
        #Save all the provided models with thier accuracy and FLOPs and Energy in a database table
        if e<self.best:
            conn = sqlite3.connect(FLAGS.model_dir+'\\bests.db')
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?,?)''',[self.num,statea,acc,t_flops,e])
            conn.commit()
            conn.close()
            self.best=e
        self.num = self.num + 1
        return e
#Estimating the accuracy of each model by training for some epochs
def train(state, number):
    #save all the half-trained models in separated directories
    path = FLAGS.model_dir + '/' + str(number)
    if not os.path.exists(path):
        os.makedirs(path)
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    g = tf.Graph()
    with tf.device('/gpu:0'):
        with g.as_default():

            limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
            rimage = tf.placeholder(tf.float32,
                                    [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels],
                                    name='rimage')
            targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

            snet = nf.create(limage, rimage, targets, state, FLAGS.net_type)

            loss = snet['loss']
            train_step = snet['train_step']
            session = tf.InteractiveSession()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            acc_loss = tf.placeholder(tf.float32, shape=())
            loss_summary = tf.summary.scalar('loss', acc_loss)
            train_writer = tf.summary.FileWriter(path + '/training', g)

            saver = tf.train.Saver(max_to_keep=1)
            losses = []
            summary_index = 1
            lrate = 1e-2
            time=0


            for it in range(1, FLAGS.num_iter):
                lpatch, rpatch, patch_targets = dhandler.next_batch()

                train_dict = {limage: lpatch, rimage: rpatch, targets: patch_targets,
                              snet['is_training']: True, snet['lrate']: lrate}
                _, mini_loss = session.run([train_step, loss], feed_dict=train_dict)
                losses.append(mini_loss)
                #update the trained weights each 100 iterations
                if it % 10 == 0:
                    print('Loss at step: %d: %.6f' % (it, mini_loss))
                    saver.save(session, os.path.join(path, 'model.ckpt'), global_step=snet['global_step'])
                    train_summary = session.run(loss_summary,
                                                feed_dict={acc_loss: np.mean(losses)})
                    train_writer.add_summary(train_summary, summary_index)
                    summary_index += 1
                    train_writer.flush()
                    losses = []

                if it == 24000:
                    lrate = lrate / 5.
                elif it > 24000 and (it - 24000) % 8000 == 0:
                    lrate = lrate / 5.
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops is not None:
            t_flops=flops.total_float_ops
    return t_flops

#Compute accuracy by evaluating saved models
def evaluate(state,number):
    lpatch, rpatch, patch_targets = dhandler.evaluate()
    labels = np.argmax(patch_targets, axis=1)
    path = FLAGS.model_dir + '/' + str(number)

    with tf.Session() as session:
        limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
        rimage = tf.placeholder(tf.float32,
                                [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels],
                                name='rimage')
        targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

        snet = nf.create(limage, rimage, targets, state,FLAGS.net_type)
        prod = snet['inner_product']
        predicted = tf.argmax(prod, axis=1)
        acc_count = 0

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(path))

        for i in range(0, lpatch.shape[0], FLAGS.eval_size):
            eval_dict = {limage: lpatch[i: i + FLAGS.eval_size],
                         rimage: rpatch[i: i + FLAGS.eval_size], snet['is_training']: False}
            pred = session.run([predicted], feed_dict=eval_dict)
            acc_count += np.sum(np.abs(pred - labels[i: i + FLAGS.eval_size]) <= 3)
            print('iter. %d finished, with %d correct (3-pixel error)' % (i + 1, acc_count))

        print('accuracy: %.3f' % ((acc_count / lpatch.shape[0]) * 100))
    return ((acc_count / lpatch.shape[0]) * 100)


if __name__ == '__main__':
    #Initializing the process by random network
    init = [['conv2d', 32, 'same',5],
            ['conv2d', 64, 'same',5],
            ['none',0,'none',0],
            ['conv2d', 64, 'same',5],
            ['none',0,'none',0],
            ['conv2d', 64, 'same',5],
            ['conv2d', 64, 'same',5],
            ['none',0,'none',0],
            ['conv2d', 64, 'same',5],
            ['conv2d', 64, 'same',5],
            ['conv2d', 64, 'valid',37]]
    #Initializing Simulated Annealing hyper parameters
    tsp = SimulatedAnnealer(init)
    tsp.set_schedule(tsp.auto(0.01,10))
    tsp.copy_strategy = "slice"
    state, e = tsp.anneal()
    print()
    print("%i mile rout:" % e)
    print("state {}".format(state))
