
"""
# Authors: Xiaoyang Chen, Shengquan Chen
# Created Time : Thu 30 Sep 2021 21:45:00 PM CST
# File Name: EpiAnno.py
# Description: 
"""

import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed
dtype = np.float32
def EpiAnno(D0,D,n_classes,sample_shape):
    
    mu = [ed.Normal(loc=tf.zeros([D0], dtype),scale=tf.ones([D0], dtype),name='mu%d'%i) for i in range(n_classes)]
    sigma = ed.InverseGamma(concentration=tf.ones([D0], dtype=dtype),
            rate=tf.ones([D0], dtype=dtype),name='sigma') 
    z = [ed.MultivariateNormalDiag(loc=mu[i],sample_shape=sample_shape[i],
            scale_diag=sigma,name='z%d'%i)for i in range(n_classes)]
    
    h = tf.concat(z,axis = 0)
    beta = ed.Normal(tf.zeros([D0, D],dtype = dtype), 1., name="beta")
    alpha = ed.Normal(tf.zeros([D],dtype = dtype), 1., name="alpha")
    
    output = tf.nn.leaky_relu(h @ beta + alpha,alpha = 0.5)
    noise = ed.InverseGamma(concentration=tf.ones([1], dtype=dtype),
                            rate=tf.ones([1], dtype=dtype),name='noise')
    x = ed.Normal(loc = output, scale = noise, name = 'x')
    return mu,sigma,z,(beta,alpha),noise,x

def Q(D0,D,n_classes,sample_shape):
    qmu_loc = [tf.Variable(np.zeros([D0]), dtype=dtype) for i in range(n_classes)]
    qmu_scale = [tf.nn.softplus(0.1 * tf.Variable(np.ones([D0]), dtype=dtype)) for i in range(n_classes)]
    qmu = [ed.Normal(loc=qmu_loc[i], scale=qmu_scale[i], name='qmu%d'%i) for i in range(n_classes)]
    
    qsigma_alpha = tf.nn.softplus(0.5 * tf.Variable(np.ones([D0]), dtype=dtype))
    qsigma_beta = tf.nn.softplus(0.5 * tf.Variable(np.ones([D0]), dtype=dtype))
    qsigma = ed.InverseGamma(concentration=qsigma_alpha, rate=qsigma_beta,name='qsigma')
    
    qz = [ed.MultivariateNormalDiag(loc=qmu[i],sample_shape=sample_shape[i],
            scale_diag=qsigma,name='qz%d'%i)for i in range(n_classes)]
    
    qbeta_loc = tf.Variable(tf.zeros([D0, D], dtype=dtype), name="qbeta_loc")
    qbeta_scale = tf.math.softplus(tf.Variable(tf.ones([D0, D], dtype=dtype), name="qbeta_scale"))
    qbeta = ed.Normal(qbeta_loc, qbeta_scale, name="qbeta")

    qalpha_loc = tf.Variable(tf.zeros([D], dtype=dtype), name="qalpha_loc")
    qalpha_scale = tf.math.softplus(tf.Variable(tf.ones([D], dtype=dtype), name="qalpha_scale"))
    qalpha = ed.Normal(qalpha_loc, qalpha_scale, name="qalpha")
    
    qnoise_alpha = tf.nn.softplus(0.5 * tf.Variable(np.ones([1]), dtype=dtype))
    qnoise_beta = tf.nn.softplus(0.5 * tf.Variable(np.ones([1]), dtype=dtype))
    qnoise = ed.InverseGamma(concentration=qnoise_alpha, rate=qnoise_beta,name='qnoise')

    return qmu,qsigma,qz,(qbeta,qalpha),qnoise

def set_values(**model_kwargs):
    """Creates a value-setting interceptor."""
    def interceptor(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]
        else:
            print(f"set_values not interested in {name}.")
        return ed.interceptable(f)(*args, **kwargs)
    return interceptor

def ELBO(mu,sigma,z,w,noise,x,qmu,qsigma,qz,qw,qnoise,D,N):
    energy = tf.reduce_sum([tf.reduce_sum(v.distribution.log_prob(v.value)) for v in mu]) + \
        tf.reduce_sum(sigma.distribution.log_prob(sigma.value)) + \
        tf.reduce_sum([tf.reduce_sum(v.distribution.log_prob(v.value)) for v in w]) +\
        tf.reduce_sum([tf.reduce_sum(v.distribution.log_prob(v.value)) for v in z]) +\
        D*N*tf.reduce_sum(noise.distribution.log_prob(noise.value)) + \
        tf.reduce_sum(x.distribution.log_prob(x.value))
    entropy = tf.reduce_sum([tf.reduce_sum(v.distribution.log_prob(v.value)) for v in qmu])+\
        tf.reduce_sum(qsigma.distribution.log_prob(qsigma.value)) + \
        tf.reduce_sum([tf.reduce_sum(v.distribution.log_prob(v.value)) for v in qw]) +\
        tf.reduce_sum([tf.reduce_sum(v.distribution.log_prob(v.value)) for v in qz]) +\
        D*N*tf.reduce_sum(qnoise.distribution.log_prob(qnoise.value))
    elbo = energy - entropy
    return elbo

def predict_z(posterior_qw,x):
    x_tmp = np.copy(x)
    x_tmp_1 = np.copy(x)
    x_tmp[x_tmp>0]=0
    x_tmp_1 = x_tmp_1+x_tmp
    z = (x_tmp_1-posterior_qw[1])@(np.mat(posterior_qw[0]).I)
    return z

def predict_target(posterior_mu, posterior_sigma,test_data):
    K = len(posterior_mu)
    model = [ed.MultivariateNormalDiag(posterior_mu[i], posterior_sigma) for i in range (K)]
    prob = []
    with tf.Session() as sess:
        for i in range(K):
            label_prob = model[i].distribution.log_prob(test_data).eval()
            prob.append(label_prob)
    pred_target = np.argmax(prob,0) 
    return pred_target

def train(qmu,qsigma,qw,elbo,sess,learning_rate = 0.15,num_epochs = 50000,verbose = True):
    posterior_mu = []
    posterior_sigma = []
    post_qw = []
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(-elbo)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_epochs+1):
        sess.run(train)
        if i>=num_epochs-1000:
            posterior_mu.append(sess.run(qmu))
            posterior_sigma.append(sess.run(qsigma))
            post_qw.append(sess.run(qw))
        if verbose:
            if i % 10 == 0: print(".", end="", flush=True)
            if i % 100 == 0:
                str_elbo = str(sess.run(elbo) / 1000) + " k"
                print("\n" + str(i) + " epochs\t" + str_elbo, end="", flush=True)
    return posterior_mu,posterior_sigma,post_qw
