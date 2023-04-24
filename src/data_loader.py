import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append("../renderer/")

import nmr_test as nmr
import neural_renderer

bVerbose = True
bCuda = False
class MyDataset(Dataset):
    def __init__(self, data_dir, img_size, texture_size, faces, vertices, distence=None, mask_dir='', ret_mask=False):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for file in files:
            if distence is None:
                self.files.append(file)
            else:
                data = np.load(os.path.join(self.data_dir, file))
                veh_trans = data['veh_trans']
                cam_trans = data['cam_trans']

                cam_trans[0][0] = cam_trans[0][0] + veh_trans[0][0]
                cam_trans[0][1] = cam_trans[0][1] + veh_trans[0][1]
                cam_trans[0][2] = cam_trans[0][2] + veh_trans[0][2]

                veh_trans[0][2] = veh_trans[0][2] + 0.2

                dis = (cam_trans - veh_trans)[0, :]
                dis = np.sum(dis ** 2)
                # print(dis)
                if dis <= distence:
                    self.files.append(file)
        print(f'len(self.files): {len(self.files)}')
        print(f'faces.shape[0]: {faces.shape[0]}')
        self.img_size = img_size
        textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        if bCuda:
            self.textures = torch.from_numpy(textures).cuda(device=0)
        # self.faces_var = torch.from_numpy(faces[None, :, :]).cuda(device=0) # 11/5/2022 9:17:13 PM: Neil commented out
        # 11/5/2022 9:17:23 PM: Neil self.faces_var debug: start
            self.faces_var = torch.from_numpy(np.asarray(faces[None, :, :].cpu())).cuda(device=0)
        # 11/5/2022 9:17:23 PM: Neil self.faces_var debug: end
        # self.vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0) # 11/5/2022 9:20:16 PM: Neil commented out
        # 11/5/2022 9:20:41 PM: Neil self.vertices_var debug: start
            self.vertices_var = torch.from_numpy(np.asarray(vertices[None, :, :].cpu())).cuda(device=0)
        # 11/5/2022 9:20:41 PM: Neil self.vertices_var debug: end
        elif not bCuda:
            self.textures = torch.from_numpy(textures)
            self.faces_var = torch.from_numpy(np.asarray(faces[None, :, :].cpu()))
            self.vertices_var = torch.from_numpy(np.asarray(vertices[None, :, :].cpu()))
            
        # 11/29/2022 11:52:43 AM: Neil added shapes: start
        if bVerbose:
            print(f'self.textures.size(): {self.textures.size()}\nself.vertices_var.size(): {self.vertices_var.size()}\nself.faces_var.size(): {self.faces_var.size()}')
            # import sys;sys.exit()
        # 11/29/2022 11:52:49 AM: Neil added shapes: end
        self.mask_renderer = nmr.NeuralRenderer(img_size=self.img_size).cuda() # 11/27/2022 1:06:06 PM: Neil commented out
        # if bCuda:
        #     self.mask_renderer = nmr.NeuralRenderer(img_size=self.img_size).cuda(device=0) # 11/27/2022 1:06:15 PM: Neil added
        # elif not bCuda:
        #     self.mask_renderer = nmr.NeuralRenderer(img_size=self.img_size)
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        # print(self.files)
    
    def set_textures(self, textures):
        self.textures = textures
    
    def __getitem__(self, index):
        index = 5
        
        print(index)
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file)
        img = data['img']
        veh_trans = data['veh_trans']
        cam_trans = data['cam_trans']
        cam_trans[0][0] = cam_trans[0][0] + veh_trans[0][0]
        cam_trans[0][1] = cam_trans[0][1] + veh_trans[0][1]
        cam_trans[0][2] = cam_trans[0][2] + veh_trans[0][2]

        veh_trans[0][2] = veh_trans[0][2] + 0.2

        eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)
        
        self.mask_renderer.renderer.renderer.eye = eye
        self.mask_renderer.renderer.renderer.camera_direction = camera_direction
        self.mask_renderer.renderer.renderer.camera_up = camera_up 

        imgs_pred = self.mask_renderer(self.vertices_var, self.faces_var, self.textures)
        # masks = imgs_pred[:, 0, :, :] | imgs_pred[:, 1, :, :] | imgs_pred[:, 2, :, :]
        # print(masks.size())
        
        img = img[:, :, ::-1]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0)
        # print(img.size())
        # print(imgs_pred.size())
        imgs_pred = imgs_pred / torch.max(imgs_pred)
        
        
        
        # if self.ret_mask:
        mask_file = os.path.join(self.mask_dir, self.files[index][:-4] + '.png')
        mask = cv2.imread(mask_file)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        # mask = torch.from_numpy(mask.astype('float32')).cuda() # 11/27/2022 1:06:57 PM: Neil commented out
        if bCuda:
            mask = torch.from_numpy(mask.astype('float32')).cuda(device=0) # 11/27/2022 1:07:06 PM: Neil added
        elif not bCuda:
            mask = torch.from_numpy(mask.astype('float32'))
        # print(mask.size())
        # print(torch.max(mask))

        total_img = img * (1-mask) + 255 * imgs_pred * mask

        return index, total_img.squeeze(0) , imgs_pred.squeeze(0), mask
        # return index, total_img.squeeze(0) , imgs_pred.squeeze(0)'''
    
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    obj_file = 'audi_et_te.obj'
    vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, load_texture=True)
    # dataset = MyDataset('../data/phy_attack/train/', 608, 4, faces, vertices) # 11/22/2022 2:41:52 PM: Neil commented out
    # 11/22/2022 2:42:20 PM: dataset directory: start
    dataset = MyDataset('data/phy_attack/train/', 608, 4, faces, vertices)
    # 11/22/2022 2:42:20 PM: dataset directory: end
    loader = DataLoader(
        dataset=dataset,   
        batch_size=3,     
        shuffle=True,            
        #num_workers=2,              
    )
    # 11/26/2022 5:02:33 PM: DataLoader object debug: start
    if bVerbose:
        print(f'loader: {loader}')
        # print(f'loader[0]: {loader[0]}')
        # print(f'loader.__getitem__(0): {loader.__getitem__(0)}')
        # import sys;sys.exit()
    # 11/26/2022 5:02:33 PM: DataLoader object debug: end
    for img, car_box in loader:
        print(img.size(), car_box.size())
# ß 11/5/2022 8:26:34 PM: Neil commented out