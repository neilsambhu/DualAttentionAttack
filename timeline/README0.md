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
7/19/2022 1:25 PM: train.py:run_cam:cal_texture:stop texture_param.type() conversion 
from torch.cuda.FloatTensor to torch.cuda.ByteTensor  
7/20/2022 1:01 PM:  
torch.nn.Tanh() solved.  
train.py:run_cam:cal_texture  
# SSD Red Hat Enterprise Linux 8
9/19/2022 6:49:15 PM: setup conda environment
```
conda create -n dualattentionattack python=3.7
conda activate dualattentionattack
bash run_custom_setup/install_requirements.sh
```
10/8/2022 10:58:28 PM: get dimensions for assessing broadcasting
```
(dualattentionattack) [nsambhu@localhost DualAttentionAttack]$ ./run_custom/train.sh
Neil src/train.py:56
obj_file: src/audi_et_te.obj, texture_size: 6
Neil src/train.py:60
Neil src/train.py:92
args.content: src/contents/smile.jpg
args.content: src/contents/smile.jpg exists
Neil src/train.py:102
args.canny: src/contents/smile_edge.jpg
args.canny: src/contents/smile_edge.jpg exists
Neil src/train.py:111
Neil src/train.py:114
Neil src/train.py:133
Neil src/train.py:177
Neil src/train.py:180
type(texture_param): <class 'numpy.ndarray'>
Neil src/train.py:184
type(texture_param): <class 'torch.Tensor'>
texture_param.type(): torch.cuda.FloatTensor
Neil src/train.py:190
type(textures): <class 'torch.Tensor'>
Neil src/train.py:197
Neil src/train.py:200
args.faces: src/all_faces.txt
Neil src/train.py:209
logs/epoch-1+batch_size-1+lr-0.01+model-resnet50+loss_func-loss_midu+loss_content+loss_smooth+lamb-0.0001+D1-0.9+D2-0.1+T-0.0001+
Neil src/train.py:352
Neil src/train.py:381
type(texture_param): <class 'torch.Tensor'>
texture_param.type(): torch.cuda.FloatTensor
src/data/phy_attack/train/
Neil src/train.py:253
12500
Neil /home/nsambhu/github/DualAttentionAttack/src/data_loader.py:47
type(faces): <class 'torch.Tensor'>
Neil /home/nsambhu/github/DualAttentionAttack/src/data_loader.py:52
Neil /home/nsambhu/github/DualAttentionAttack/src/data_loader.py:56
Neil src/train.py:256
Neil src/train.py:269
Neil src/train.py:218
type(texture_param): <class 'torch.Tensor'>
texture_param.type(): torch.cuda.FloatTensor
Neil src/train.py:224
Neil src/train.py:227
texture_origin.size(): torch.Size([1, 23145, 6, 6, 6, 3])
texture_mask.size(): torch.Size([1, 23145, 6, 6, 6, 3])
textures.size(): torch.Size([647, 646, 3])
(1 - texture_mask).size(): torch.Size([1, 23145, 6, 6, 6, 3])
(texture_origin * (1 - texture_mask)).size(): torch.Size([1, 23145, 6, 6, 6, 3])
Traceback (most recent call last):
  File "src/train.py", line 386, in <module>
    run_cam(train_dir, EPOCH)
  File "src/train.py", line 270, in run_cam
    textures = cal_texture()
  File "src/train.py", line 245, in cal_texture
    print(f'(texture_mask * textures).size(): {(texture_mask * textures).size()}')
RuntimeError: The size of tensor a (6) must match the size of tensor b (646) at non-singleton dimension 4
```
