#Mcnet

This is an implementation ofMcnet in TensorFlow for Classification Carassius auratus of diploid (2N = 100) and triploid fishes (3N = 162) image .

###Model Description

Mcnet is developsed a discriminative framework to classify 2N and 3N in the goldfish (C. auratus complex)C. auratusC. auratus. Our system, call Mcnet, provides a novel three pathways (i.e. two HSV channels and one RGB channel) based convolutional neural network, which enable it to combine the advantages of HSV and RGB, and fuse them together to enhance the final feature representation.   Through carefully design in terms of specific turning the channels and convolu-tional  levels,  configuration of different convolution kernel size, GPU based approach to reduce the training time with millions of parameters, we optimize both the structure and performance of  Mcnet.

###Requirements
TensorFlow needs to be installed before running the scripts. TensorFlow v1.1.0 is supported; for TensorFlow v0.12 please refer to this branch; for TensorFlow v0.11 please refer to this branch. Note that those branches may not have the same functional as the current master.

To install the required python packages (except TensorFlow), run

```
pip install -r requirements.txt
```

or for a local installation

```
pip install -user -r requirements.txt
```

###Dataset and Training
Our original dataset consists of 1154 diploid and triploid crucian carp, collected in Dianchi Lake ( 25°02'11 "N, 102°42'31" E, 1800 meters above sea level) ranging from 2009 to April 2016. 

 To train the model McNet , we can run train.py script:
 

    python train.py  --batch-size {Number of images sent to the network in one step.} --train-data {Path to the training trefcords file.} --test-data {Path to the test trefcords file.} --mode-path {Where restore model parameters from.} --log-dir {save training log path.}
 
    
