import EpiAnno
import os
import tensorflow as tf
import hickle as hkl
from tensorflow_probability import edward2 as ed
from data_processing import data_processing
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import KFold
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='EpiAnno: Single-cell epigenomic data annotation via supervised non-linear embedding')
    parser.add_argument('--train_data', '-d', type=str, help='input train data path')
    parser.add_argument('--train_cell_type', '-c', type=str, help='input train cell-type path')
    parser.add_argument('--test_data', '-t', type=str, help='input test data path')
    #parser.add_argument('--test_cell_type', '-tc', type=str, help='input test cell-type path')
    parser.add_argument('--outdir', '-o', type=str, default=os.path.dirname(os.getcwd())+'/output/crossdataset-projection', help='Output path')
    parser.add_argument('--verbose', type = bool, default = True, help='Print loss of training process')
    parser.add_argument('--gpu', '-g', default='0', type=str, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for repeat results')
    parser.add_argument('--latent_dim', '-l',type=int, default=10, help='latent dim')
    parser.add_argument('--peak_rate', '-r', type=float, default=0.03, help='Remove low ratio peaks')
    parser.add_argument('--epoch', '-e', type=int, default=50000, help='Epochs for training(50000 for enough training)')
    parser.add_argument('--learning_rate','-lr', type=float, default=0.15,help='Learning rate for training(0.15 as a better choice)')
    parser.add_argument('--save_model','-m', type = bool, default = True, help='Save parameters of EpiAnno model')
    parser.add_argument('--save_result','-p', type = bool, default = True, help='Save test labels and predicted labels')

    args = parser.parse_args()
    num_epochs = args.epoch
    verbose = args.verbose
    peak_rate = args.peak_rate
    latent_dim = args.latent_dim
    learning_rate = args.learning_rate
    outdir = args.outdir
    # Load data and cell type
    train_data = hkl.load(args.train_data)
    train_label = hkl.load(args.train_cell_type)
    test_data = hkl.load(args.test_data)
    test_label = np.array(range(test_data.shape[0]))# create fake label for processing
    train_data,train_label,test_data,test_label = data_processing(train_data,train_label,test_data,test_label,peak_rate)
    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if tf.test.is_gpu_available()==True:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth=True
        tf.Session(config=tf_config)
    # LabelEncoder and label to target
    le = LabelEncoder()
    train_target = le.fit_transform(train_label)
    # create data_train for training
    dtype = np.float32
    n_classes = max(train_target)+1
    Data = {i: np.array(train_data[train_target == i,:])  for i in range(n_classes)}
    sample_shape = {i: sum(train_target == i)  for i in range(n_classes)}# sample_shape for each class
    data_train = Data[0]
    for i in range(n_classes-1):
        data_train=np.vstack((data_train,Data[i+1]))
    # get Variables of EpiAnno
    qmu , qsigma,qz,qw,qnoise = EpiAnno.Q(latent_dim,data_train.shape[1],n_classes,sample_shape)
    qmu_dict = {v.distribution.name.split("_")[0].split("/")[0][1:]: v for v in qmu}
    qw_dict = {v.distribution.name.split("_")[0].split("/")[0][1:]: v for v in qw}
    qz_dict = {v.distribution.name.split("_")[0].split("/")[0][1:]: v for v in qz}
    # set Variables to EpiAnnp, then get the ELBO
    with ed.interception(EpiAnno.set_values(**qmu_dict,sigma=qsigma,**qw_dict,x = data_train,\
                                          **qz_dict,noise = qnoise)):
        pmu,psigma,pz,pw,pnoise,px = EpiAnno.EpiAnno(10,data_train.shape[1],n_classes,sample_shape)
    elbo = EpiAnno.ELBO(pmu,psigma,pz,pw,pnoise,px,qmu , qsigma,qz,qw,qnoise,data_train.shape[1],data_train.shape[0])
    # train the EpiAnno model, get the last 1000 parameters of the model from training
    with tf.Session(config=tf_config) as sess:
        posterior_mu,posterior_sigma,posterior_qw = EpiAnno.train(qmu,qsigma,qw,elbo,sess,learning_rate,num_epochs,verbose)
    # random select 10 parameters from 1000 parameters
    select = np.random.randint(0,1000,10)
    pred_target_test = []
    for i in range(10):
        pred_z_test = EpiAnno.predict_z(posterior_qw[select[i]],test_data)
        pred_target_test.append(EpiAnno.predict_target(posterior_mu[select[i]],posterior_sigma[select[i]],pred_z_test))
    pred_target_test = np.array(pred_target_test)
    pred_target = []
    for i in range(pred_target_test.shape[1]):
        b = np.bincount(pred_target_test[:,i])
        pred_target.append(np.argmax(b))
    pred_label = le.inverse_transform(np.array(pred_target))
    if args.save_model:
        # save 10 parameters we selected
        parameter = {
            'posterior_mu':np.array(posterior_mu)[select],
            'posterior_sigma':np.array(posterior_sigma)[select],
            'posterior_qw':np.array(posterior_qw)[select]
        }
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        np.save(outdir+'/model_peak_rate_%.3f_latent_%d.npy'%(peak_rate,latent_dim),parameter)
    if args.save_result:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        result = {
            'pred_label':pred_label
        }
        np.save(outdir+'/result_peak_rate_%.3f_latent_%d.npy'%(peak_rate,latent_dim),result)
        