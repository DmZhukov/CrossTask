# Cross-task weakly supervised learning from instructional videos

## About
This is an implementation of the paper "Cross-task weakly supervised learning from instructional videos" by D. Zhukov, J.-B. Alayrac, R. G. Cinbis, D. Fouhey, I. Laptev and J. Sivic [[arXiv](https://arxiv.org/abs/1903.08225)]

Please, consider siting the paper, if you use our code or data:
> @INPROCEEDINGS{Zhukov2019,
>     author      = {Zhukov, Dimitri and Alayrac, Jean-Baptiste and Cinbis, Ramazan Gokberk and Fouhey, David and Laptev, Ivan and Sivic, Josef},
>     title       = {Cross-task weakly supervised learning from instructional videos},
>     booktitle   = CVPR,
>     year        = {2019},
> }

## CrossTask dataset
CrossTask dataset contains instructional videos, collected for 83 different tasks.
For each task we provide an ordered list of steps with manual descriptions.
The dataset is divided in two parts: 18 primary and 65 related tasks.
Videos for the primary tasks are collected manually and provided with annotations for temporal step boundaries.
Videos for the related tasks are collected automatically and don't have annotations.

Tasks, video URLs and annotations are provided [here](https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip). See readme.txt for details.

Features are available [here](https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip) (30Gb). Features for each video are provided in a NumPy array with one 3200-dimensional feature per second. The feature vector is a concatenation of RGB I3D features (columns 0-1023), Resnet-152 (columns 1024-3071) and audio VGG features (columns 3072-3199).

Temporal constraints, extracted from narration are available [here](https://www.di.ens.fr/~dzhukov/crosstask/crosstask_constraints.zip).

**Update 30/05/2019:** added videos_val.csv with validation set from the paper, removed extra lines from the constraints.

## Code
Provided code can be used to train and evaluate the component model, proposed in the paper, on CrossTask dataset. 
It was tested with Python 3.7, PyTorch 1.0, NumPy 1.16 and Cython 0.29. 

 1. Clone the repository
```bash
git clone https://github.com/DmZhukov/CrossTask.git
cd CrossTask
```
 2. Download and unpack the dataset
```bash
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_constraints.zip
unzip '*.zip'
```
 3. Compile Cython code
```bash
python setup.py build_ext --inplace
```
 4. Run training
```bash
python train.py
```
