# Cross-task weakly supervised learning from instructional videos
## CrossTask dataset
CrossTask dataset contains instructional videos, collected for 83 different tasks.
For each task we provide an ordered list of steps with manual descriptions.
The dataset is divided in two parts: 18 primary and 65 related tasks.
Videos for the primary tasks are collected manually and provided with annotation for temporal step boundaries.
Videos for the related tasks are collected automatically and have no annotation.

Tasks, video URLs and annotations are provided [here](https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip). See readme.txt for details.

Features are available [here](https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip) (30Gb). Features for each video are provided in a NumPy array with one 3200-dimensional feature per second. The feature vector is a concatenation of RGB I3D features (columns 0-1023), Resnet-152 (columns 1024-3071) and audio VGG features (columns 3072-3199).

Temporal constrains, extracted from narration are available [here](https://www.di.ens.fr/~dzhukov/crosstask/crosstask_constraints.zip).

Please, consider siting our paper if you use this data:
> @INPROCEEDINGS{Zhukov2019,
>     author      = {Zhukov, Dimitri and Alayrac, Jean-Baptiste and Cinbis, Ramazan Gokberk and Fouhey, David and Laptev, Ivan and Sivic, Josef},
>     title       = {Cross-task weakly supervised learning from instructional videos},
>     booktitle   = CVPR,
>     year        = {2019},
> }
