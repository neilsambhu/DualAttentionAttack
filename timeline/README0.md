7/18/2022 1:22 PM: setup github at https://github.com/neilsambhu/DualAttentionAttack  
7/18/2022 1:39 PM: unzip "masks" and "phy_attack" into src/data/  
7/18/2022 2:15 PM: 
```
(dualattentionattack) nsambhu@SAMBHU19:~/github/DualAttentionAttack$ conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: \ 
Found conflicts! Looking for incompatible packages.
This can take several minutes.  Press CTRL-C to abort.
failed                                                                          

UnsatisfiableError: The following specifications were found
to be incompatible with the existing python installation in your environment:

Specifications:

  - torchvision==0.5.0 -> python[version='>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=3.8,<3.9.0a0|>=3.7,<3.8.0a0']

Your python: python=3.9

If python is on the left-most side of the chain, that's the version you've asked for.
When python appears to the right, that indicates that the thing on the left is somehow
not available for the python version you are constrained to. Note that conda will not
change your python version to a different minor version unless you explicitly specify
that.

The following specifications were found to be incompatible with each other:

Output in format: Requested package -> Available versions

Package cudatoolkit conflicts for:
torchvision==0.5.0 -> cudatoolkit[version='>=10.0,<10.1|>=9.2,<9.3|>=10.1,<10.2']
torchvision==0.5.0 -> pytorch==1.4.0 -> cudatoolkit[version='>=10.0.130,<10.1.0a0|>=10.1.243,<10.2.0a0|>=9.2,<9.3.0a0']

Package _openmp_mutex conflicts for:
pytorch==1.4.0 -> libgcc-ng[version='>=7.3.0'] -> _openmp_mutex[version='>=4.5']
python=3.9 -> libgcc-ng[version='>=7.5.0'] -> _openmp_mutex[version='>=4.5']The following specifications were found to be incompatible with your system:

  - feature:/linux-64::__glibc==2.27=0
  - feature:|@/linux-64::__glibc==2.27=0
  - python=3.9 -> libgcc-ng[version='>=7.5.0'] -> __glibc[version='>=2.17']

Your installed version is: 2.27


```
Downgrade from Python 3.9 to Python 3.7.  
7/18/2022 2:30 PM: get training working.  
7/18/2022 2:47 PM: "import cv2":opencv won't install. Restart environment.
```
conda create -n dualattentionattack python=3.7
```
7/18/2022 3:24 PM: restart environment.  
7/18/2022 5:14 PM: need to find what's wrong with train.py:"texture_origin"  
7/18/2022 5:50 PM: train.py:run_cam:dataset  
7/18/2022 6:16 PM: train.py:run_cam:cal_texture  
7/18/2022 6:25 PM: need to learn torch.nn.Tanh()  