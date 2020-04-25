# EpiAnno
##### Single-cell epigenomic data annotation via supervised non-linear embedding
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
