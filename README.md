# EpiAnno
#### Single-cell epigenomic data annotation via supervised non-linear embedding
The recent advances in profiling epigenetic landscape of thousands of individual cells increase the demand for automatic annotation of cells, given that the conventional cell annotation method is cumbersome and time-consuming, and several supervised computational methods can hardly characterize the high-dimensional sparse single-cell epigenomic data. Here we proposed EpiAnno, a probabilistic generative model integrated with a Bayesian neural network, to annotate single-cell epigenomic data in a supervised manner.  

<div align=center>
<img src = "inst/Fig1-01.jpg" width = 40% height = 40%>
</div>  

## Installation  

```  
Requiements:  
1. Python 3.5 or greater version  
2. Packages:  
    numpy (>=1.15.1)  
    tensorflow_probability (0.7.0)  
    tensorflow(-gpu) (1.15.2)  
    hickle (>=3.4)
  
Package installation:
  
pip install -U numpy  
pip install tensorflow-probability==0.7.0  
pip install tensorflow-gpu==1.15.2 #pip install tensorflow==1.15.2  
pip install -U hickle  
```
## Tutorial  
### self-projection   
Two input files(.hkl) are required: 1) a samples-by-peaks Array (samples * peaks) 2) a list(vector) of cell-type labels.  
  
The dataset we used is available on github. First unzip the datasets:  
```  
$ tar -xzvf data/self_projection.tar.gz
```
Then run the self-projection subprogram:
```   
$ python code/run_self_projection.py -d data_count.hkl -c cell_type.hkl -o outdir -g gpu
```
For exsample:
```
$ python code/run_self_projection.py -d data/self-projection/InSilico_count.hkl -c data/self-projection/InSilico_cell_type.hkl
```
Or you can get help in this way:
```  
$ python code/run_self_projection.py -h
usage: run_self_projection.py [-h] [--data DATA] [--cell_type CELL_TYPE]
                              [--outdir OUTDIR] [--verbose VERBOSE]
                              [--gpu GPU] [--seed SEED] [--state STATE]
                              [--latent_dim LATENT_DIM]
                              [--peak_rate PEAK_RATE] [--epoch EPOCH]
                              [--learning_rate LEARNING_RATE]
                              [--n_splits N_SPLITS] [--save_model SAVE_MODEL]
                              [--save_result SAVE_RESULT]

EpiAnno: Single-cell epigenomic data annotation via supervised non-linear
embedding

optional arguments:
  -h, --help            show this help message and exit
  --data DATA, -d DATA  input data path
  --cell_type CELL_TYPE, -c CELL_TYPE
                        input cell-type path
  --outdir OUTDIR, -o OUTDIR
                        Output path
  --verbose VERBOSE     Print loss of training process
  --gpu GPU, -g GPU     Select gpu device number when training
  --seed SEED           Random seed for repeat results
  --state STATE, -s STATE
                        Random state for KFold
  --latent_dim LATENT_DIM, -l LATENT_DIM
                        latent dim
  --peak_rate PEAK_RATE, -r PEAK_RATE
                        Remove low ratio peaks
  --epoch EPOCH, -e EPOCH
                        Epochs for training(50000 for enough training)
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate for training(0.15 as a better choice)
  --n_splits N_SPLITS, -n N_SPLITS
                        Number of folds
  --save_model SAVE_MODEL, -m SAVE_MODEL
                        Save parameters of EpiAnno model
  --save_result SAVE_RESULT, -p SAVE_RESULT
                        Save test labels and predicted labels
```

### crossdataset-projection

