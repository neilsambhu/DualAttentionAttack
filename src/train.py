import os
import sys
from PIL import Image
import numpy as np
import tqdm
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")

# sys.path.append("../renderer/")

import nmr_test as nmr
import neural_renderer

from torchvision.transforms import Resize
from data_loader import MyDataset
from torch.utils.data import Dataset, DataLoader
from grad_cam import CAM

import torch.nn.functional as F
import random
from functools import reduce
import argparse
from inspect import currentframe, getframeinfo # 11/17/2022 3:15:57 PM: Neil added

torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)


parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--lamb", type=float, default=1e-4)
parser.add_argument("--d1", type=float, default=0.9)
parser.add_argument("--d2", type=float, default=0.1)
parser.add_argument("--t", type=float, default=0.0001)

parser.add_argument("--obj", type=str, default='audi_et_te.obj')
parser.add_argument("--faces", type=str, default='./all_faces.txt')
parser.add_argument("--datapath", type=str)
parser.add_argument("--content", type=str)
parser.add_argument("--canny", type=str)

args = parser.parse_args()


obj_file =args.obj
texture_size = 6
vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)


mask_dir = os.path.join(args.datapath, 'masks/')


torch.autograd.set_detect_anomaly(True)

log_dir = ""

def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name



T = args.t
D1 = args.d1
D2 = args.d2
lamb = args.lamb
LR = args.lr
BATCH_SIZE = args.batchsize
EPOCH = args.epoch


# 11/5/2022 7:46:03 PM: Neil debugging pickle error: start
bVerbose = True
if bVerbose:
    print(f'args.content: {args.content}')
    # args.content: contents/smile.jpg
    # import sys;sys.exit()
# 11/5/2022 7:46:03 PM: Neil debugging pickle error: end
# texture_content = torch.from_numpy(np.load(args.content,allow_pickle=True)).cuda(device=0) # 11/5/2022 7:50:37 PM: Neil commented out
# 11/5/2022 7:50:59 PM: Neil debugging pickle error: start
texture_content = torch.from_numpy(np.asarray(Image.open(args.content))).cuda(device=0) # 11/5/2022 7:52:04 PM: Neil new image load technique
# 11/5/2022 7:50:59 PM: Neil debugging pickle error: end
# texture_canny = torch.from_numpy(np.load(args.canny)).cuda(device=0) # 11/5/2022 7:53:35 PM: Neil commented out
# 11/5/2022 7:53:44 PM: Neil duplicate pickle solution from texture_content to texture_canny: start
texture_canny = torch.from_numpy(np.asarray(Image.open(args.canny))).cuda(device=0)
# 11/5/2022 7:53:44 PM: Neil duplicate pickle solution from texture_content to texture_canny: end
texture_canny = (texture_canny >= 1).int()
def loss_content_diff(tex):
    return  D1 * torch.sum(texture_canny * torch.pow(tex - texture_content, 2)) + D2 * torch.sum((1 - texture_canny) * torch.pow(tex - texture_content, 2)) 

def loss_smooth(img, mask):
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    mask = mask[:, :-1, :-1]
    
    mask = mask.unsqueeze(1)
    # print(mask.size())
    # print(s1.size())
    return T * torch.sum(mask * (s1 + s2))

cam_edge = 7

vis = np.zeros((cam_edge, cam_edge))

def dfs(x1, x, y, points):
    points.append(x1[x][y])
    global vis
    vis[x][y] = 1
    n = 1
    # print(x, y)
    if x+1 < cam_edge and x1[x+1][y] > 0 and not  vis[x+1][y]:
        n += dfs(x1, x+1, y, points)
    if x-1 >= 0 and x1[x-1][y] > 0 and not  vis[x-1][y]:
        n += dfs(x1, x-1, y, points)
    if y+1 < cam_edge and x1[x][y+1] > 0 and not  vis[x][y+1]:
        n += dfs(x1, x, y+1, points)
    if y-1 >= 0 and x1[x][y-1] > 0 and not  vis[x][y-1]:
        n += dfs(x1, x, y-1, points)
    return n
        
    
def loss_midu(x1):
    # print(torch.gt(x1, torch.ones_like(x1) * 0.1).float())
    
    x1 = torch.tanh(x1)
    global vis
    vis = np.zeros((cam_edge, cam_edge))
    
    loss = []
    # print(x1)
    for i in range(cam_edge):
        for j in range(cam_edge):
            if x1[i][j] > 0 and not vis[i][j]:
                point = []
                n = dfs(x1, i, j, point)
                # print(n)
                # print(point)
                loss.append( reduce(lambda x, y: x + y, point) / (cam_edge * cam_edge + 1 - n) )
    # print(vis)
    if len(loss) == 0:
        return torch.zeros(1).cuda()
    return reduce(lambda x, y: x + y, loss) / len(loss)

# Textures

texture_param = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32') * -0.9# test 0
if bVerbose:
    print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: texture_param.shape: {texture_param.shape}')
texture_param = torch.autograd.Variable(torch.from_numpy(texture_param).cuda(device=0), requires_grad=True)
if bVerbose:
    print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: texture_param.size(): {texture_param.size()}')

# texture_origin = torch.from_numpy(textures[None, :, :, :, :, :]).cuda(device=0) # 11/5/2022 7:59:13 PM: Neil commented out
# 11/5/2022 7:59:39 PM: Neil texture_origin: start
# print(f'textures.size(): {textures.size()}') # textures.size(): torch.Size([23145, 6, 6, 6, 3])
# from PIL import Image;im=Image.fromarray(textures.cpu().numpy());im.save("textures.png");import sys;sys.exit()
texture_origin = torch.from_numpy(np.asarray(textures[None, :, :, :, :, :].cpu())).cuda(device=0)
# 11/5/2022 7:59:39 PM: Neil texture_origin: end

texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
with open(args.faces, 'r') as f:
    face_ids = f.readlines()
    # print(face_ids)
    # 11/9/2022 1:42:28 PM: Neil texture_mask debug: start
    print(f'np.shape(texture_mask): {np.shape(texture_mask)}')
    print(f'type(face_ids): {type(face_ids)}')
    print(f'len(face_ids): {len(face_ids)}')
    # import sys;sys.exit()
    # 11/9/2022 1:42:28 PM: Neil texture_mask debug: end
    for face_id in face_ids:
        if face_id != '\n':
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
            # 11/11/2022 5:39:05 PM: Neil texture_mask and face_id pring: start
            # print(f'face_id.strip(): {face_id.strip()}\tnp.shape(texture_mask): {np.shape(texture_mask)}')
            # import sys;sys.exit()
            # 11/11/2022 5:39:05 PM: Neil texture_mask and face_id pring: end
texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)

# 11/11/2022 3:30:12 PM: convert torch.Tensor to png on hard disk: start 
def TorchTensorToPng(tensor, sOutputFilename):
    arrayNumpy = tensor.cpu().detach().numpy().astype(np.uint8)
    # print(f'arrayNumpy: {arrayNumpy}')
    from PIL import Image
    Image.fromarray(arrayNumpy).save(sOutputFilename)
# 11/11/2022 3:30:12 PM: convert torch.Tensor to png on hard disk: end
def cal_texture(CONTENT=False):
    # 11/5/2022 9:44:33 PM: textures: start
    textures = None
    # 11/5/2022 9:44:33 PM: textures: end
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1) 
    else:
        # 11/7/2022 2:51:54 PM: texture_param to textures: start
        print(f'texture_param.size(): {texture_param.size()}') # texture_param.size(): torch.Size([647, 646, 3])
        # 11/7/2022 2:51:54 PM: texture_param to textures: end
        # 11/10/2022 1:22:09 PM: texture_param viewing: start
        # a = texture_param.cpu()
        # print(f'type(a): {type(a)}')
        # b = a.detach()
        # print(f'type(b): {type(b)}')
        # c = b.numpy()
        # print(f'type(c): {type(c)}')
        # print(f'c.shape: {c.shape}')
        # print(f'c.dtype: {c.dtype}')
        # print(f'c.dtype==np.float32: {c.dtype==np.float32}')
        # d = (c*255).astype(np.uint8)
        # from PIL import Image;im=Image.fromarray(d);im.save("texture_param.png");import sys;sys.exit()
        # 11/10/2022 1:22:09 PM: texture_param viewing: end
        # 11/11/2022 3:33:43 PM: call TorchTensorToPng: start
        # TorchTensorToPng(texture_param,'texture_param.png')
        # import sys;sys.exit();
        # 11/11/2022 3:33:43 PM: call TorchTensorToPng: end
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    # return textures
    return texture_origin * (1 - texture_mask) + texture_mask * textures # 11/5/2022 9:26:24 PM: Neil commented out # 11/17/2022 3:43:44 PM: restored
    # 11/5/2022 9:39:20 PM: cal_texture debug: start
    # return texture_origin @ (1 - texture_mask) + texture_mask * textures # 11/5/2022 9:27:08 PM: error
    # return texture_origin * (1 - texture_mask) + texture_mask @ textures # 11/5/2022 9:38:14 PM: error
    # return texture_origin @ (1 - texture_mask) + texture_mask @ textures # 11/5/2022 9:38:14 PM: error
    # a = texture_origin * (1 - texture_mask) # 11/5/2022 9:42:28 PM: successful
    # b = texture_mask * textures # 11/5/2022 9:42:34 PM: error is here
    # print(f'texture_mask: {texture_mask}\ntextures: {textures}');import sys;sys.exit()
    print(f'texture_origin.size(): {texture_origin.size()}') # texture_origin.size(): torch.Size([1, 23145, 6, 6, 6, 3]) # 14,997,960 elements
    print(f'texture_mask.size(): {texture_mask.size()}') # texture_mask.size(): torch.Size([1, 23145, 6, 6, 6, 3])
    print(f'textures.size(): {textures.size()}') # textures.size(): torch.Size([647, 646, 3]) # 1,253,886 elements
    a = texture_origin * (1 - texture_mask) # each term has the same dimension
    print(f'a.size(): {a.size()}')
    b = texture_mask * textures
    # 11/5/2022 9:39:20 PM: cal_texture debug: end
   
         
def run_cam(data_dir, epoch, train=True, batch_size=BATCH_SIZE):
    print(data_dir)
    print(data_dir, texture_size, faces.shape, vertices.shape, batch_size);quit()
    dataset = MyDataset(data_dir, 224, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)
    loader = DataLoader(
        dataset=dataset,     
        batch_size=batch_size,  
        shuffle=False,        
        # num_workers=2,     
        num_workers=2,     # 11/27/2022 12:36:33 PM: Neil added
    )
    
    optim = torch.optim.Adam([texture_param], lr = LR)
    
    Cam = CAM()

    textures = cal_texture()

    dataset.set_textures(textures)
    print(torch.cuda.is_available(),torch.cuda.device_count())
    for e in range(epoch):
        print('Epoch: ', e, '/', epoch)
        print(len(loader))
        count = 0
        # tqdm_loader = tqdm.tqdm(loader) # 11/27/2022 12:10:21 PM: Neil commented out
        if bVerbose:
            # print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: tqdm_loader: {tqdm_loader}')
            # print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: len(tqdm_loader): {len(tqdm_loader)}')
            # print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: dataset: {dataset}')
            print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: type(loader): {type(loader)}')
            print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: loader: {loader}')
            # print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: enumerate(tqdm_loader): {enumerate(tqdm_loader)}')
            # print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: list(enumerate(tqdm_loader)): {list(enumerate(tqdm_loader))}')
        # 11/26/2022 4:47:59 PM: install cupy: "conda install -c anaconda cupy"
        # for i, (index, total_img, texture_img, masks) in enumerate(tqdm_loader): # 11/27/2022 12:10:56 PM: Neil commented out
        for i,j in enumerate(loader):
            print(i)
        continue
        for i, (index, total_img, texture_img, masks) in enumerate(loader):
            if bVerbose:
                print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: Neil reached here')
            index = int(index[0])
            if bVerbose:
                print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: Neil reached here')
            
            
            total_img_np = total_img.data.cpu().numpy()[0]
            if bVerbose:
                print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: Neil reached here')
            # print(total_img_np.shape)
            total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))
            if bVerbose:
                print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: Neil reached here')

            total_img_np.save(os.path.join(log_dir, 'test_total.jpg')) 
            if bVerbose:
                print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: Neil reached here')
            # print(texture_img.size())
            # print(torch.max(texture_img))
            Image.fromarray((255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, 'texture2.png'))
            if bVerbose:
                print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: Neil reached here')
            Image.fromarray((255 * masks).data.cpu().numpy()[0].astype('uint8')).save(os.path.join(log_dir, 'mask.png'))
            if bVerbose:
                print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: Neil reached here')
            # scipy.misc.imsave(os.path.join(log_dir, 'mask.png'), (255*masks).data.cpu().numpy()[0])
            
            #######
            # CAM #
            #######
            pred = 0
            
            mask, pred = Cam(total_img, index, log_dir)
                
            
            ###########
            #   LOSS  #
            ###########
            
            
            loss = loss_midu(mask) + lamb * loss_content_diff(texture_param) + loss_smooth(texture_img, masks)
            
            
            with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
            
                tqdm_loader.set_description('Loss %.8f, Prob %.8f' % (loss.data.cpu().numpy(), pred))
                f.write('Loss %.8f, Prob %.8f\n' % (loss.data.cpu().numpy(), pred))
                

            ############
            # backward #
            ############
            if train and loss != 0:
                # print(loss.data.cpu().numpy())
                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()
            # print(texture_param)

            textures = cal_texture()
            dataset.set_textures(textures)

            
if __name__ == '__main__':
    logs = {
        'epoch': EPOCH,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'model': 'resnet50',
        'loss_func': 'loss_midu+loss_content+loss_smooth',
        'lamb': lamb,
        'D1': D1,
        'D2': D2,
        'T': T,  
    }
    
    make_log_dir(logs)
    
    train_dir = os.path.join(args.datapath, 'phy_attack/train/')
    test_dir = os.path.join(args.datapath, 'phy_attack/test/')

    # texture_param = torch.autograd.Variable(torch.from_numpy(np.load(args.content)).cuda(device=0), requires_grad=True) # 11/5/2022 8:16:39 PM: Neil commented out
    # 11/5/2022 8:16:48 PM: texture_param debug: start
    texture_param_ofSize_647_646_3 = torch.autograd.Variable(torch.from_numpy(np.asarray(Image.open(args.content), dtype=np.float32)).cuda(device=0), requires_grad=True)
    if bVerbose:
        print(f'{getframeinfo(currentframe()).filename}:{getframeinfo(currentframe()).lineno}: texture_param_ofSize_647_646_3.size(): {texture_param_ofSize_647_646_3.size()}')
    # 11/5/2022 8:16:48 PM: texture_param debug: end
    
    # run_cam(train_dir, EPOCH)
    run_cam(test_dir, EPOCH)
    
    # np.save(os.path.join(log_dir, 'texture.npy'), texture_param_ofSize_647_646_3.data.cpu().numpy())