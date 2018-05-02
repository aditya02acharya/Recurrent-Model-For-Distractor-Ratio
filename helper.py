import numpy as np
import random
import tensorflow as tf
import itertools

def processState(state):
    return np.reshape(state,[36]) 

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars/2].eval(session=sess)
    #if a.all() == b.all():
     #   print "Target Set Success"
    #else:
     #   print "Target Set Failed"


def huber_loss(y_true, y_pred, clip_value):

   x = y_true - y_pred

   condition = tf.abs(x) < clip_value
   squared_loss = .5 * tf.square(x)
   linear_loss = clip_value * (tf.abs(x) - .5 * clip_value)

   if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
   else:
        return tf.where(condition, squared_loss, linear_loss) # condition, true, false

