import os
from glob import glob

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import neural_renderer as nr
import src.nmr_test as nmr

bVerbose = True

def overlay_morph(image: torch.Tensor, rendered: torch.Tensor, mask: torch.Tensor, image_size: int):
    image = T.functional.to_pil_image(image.permute((2,0,1)))
    image = T.functional.resize(image, (image_size, image_size))
    image = T.functional.to_tensor(image)

    mask = T.functional.to_pil_image(mask.permute((2,0,1)))
    mask = T.functional.resize(mask, (image_size, image_size))
    mask = T.functional.to_tensor(mask)
    
    return image * (1 - mask) + rendered * mask


class DualAttentionDataset(Dataset):
    def __init__(self, root: str, distance: int = -1, train: bool = True):
        # if bVerbose:
        #     print(f'root before: {root}')
        self.root = root = f'{root}/train' if train else f'{root}/test'
        # if bVerbose:
        #     print(f'root after: {root}')
        #     print(f'self.root: {self.root}')
        
        assert os.path.exists(root)
        
        self.items = [] # actual data points/files
        
        for filename in os.listdir(root):
            if distance < 0:
                self.items.append(filename)
                print(f'{filename}')
            
            else:
                data = np.load(os.path.join(root, filename))
                if self.camera_to_vehicle_distance(data['cam_trans'], data['veh_trans']) < distance:
                    self.items.append(filename)
    
    def __len__(self):
        return len(self.items)
    
    def camera_to_vehicle_distance(self, camera_t: np.ndarray, vehicle_t: np.ndarray) -> float:
        new_camera_t = camera_t[0] + vehicle_t[0]
        vehicle_t[0, 2] += 0.2
        return ((new_camera_t - vehicle_t)[0] ** 2).sum()
    
    def __getitem__(self, idx: int):
        rgb_filename = os.path.join(self.root, self.items[idx])
        
        mask_filename = os.path.join('/'.join(self.root.split('/')[:-2]), 'masks', f'{self.items[idx][:-4]}.png')
        mask_img = torch.from_numpy(cv2.imread(mask_filename))
        
        data = np.load(rgb_filename)
        
        image, camera_t, vehicle_t = data['img'], data['cam_trans'].astype(np.float32), data['veh_trans']
        camera_t[0] += vehicle_t[0]
        vehicle_t[0,2] += 0.2
        eye, look_at, camera_up = nmr.get_params(camera_t, vehicle_t)
        
        return image, mask_img, (eye, look_at, camera_up)

    @staticmethod
    def dual_attn_collate(items):
        images, masks = [], []
        eyes, look_ats, camera_ups = [], [], []
        
        for img, mask, (eye, look_at, camera_up) in items:
            images.append(torch.from_numpy(img).unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            eyes.append(torch.Tensor(eye).unsqueeze(0))
            look_ats.append(torch.Tensor(look_at).unsqueeze(0))
            camera_ups.append(torch.Tensor(camera_up).unsqueeze(0))
        
        return torch.cat(images), torch.cat(masks), (torch.cat(eyes), torch.cat(look_ats), torch.cat(camera_ups))


class DASTrainingSet(Dataset):
    def __init__(self, root, image_size):
        self.image_size = image_size
        
        mask_folder = os.path.join(root, 'masks')
        rendering_folder = os.path.join(root, 'rendering')
        mash_folder = os.path.join(root, 'mash')
        assert os.path.exists(mask_folder) and os.path.exists(rendering_folder) and os.path.exists(mash_folder)
        
        self.mask_files = glob(os.path.join(mask_folder, '**.png'))
        self.rendering_files = glob(os.path.join(rendering_folder, '**.png'))
        self.mash_files = glob(os.path.join(mash_folder, '**.png'))
        
    
    def __len__(self):
        return len(self.rendering_files) # 12/14/2022 1:46:29 PM: Neil added
        
    
    def __getitem__(self, index: int):
        mask_filename = self.mask_files[index]
        rendering_filename = self.rendering_files[index]
        mash_filename = self.mash_files[index]
        
        label = int(mask_filename.split('/')[-1].split('.')[0].replace('data', ''))
        
        mask_img = TF.to_tensor(
            TF.resize(TF.to_pil_image
                      (torch.from_numpy(cv2.imread(mask_filename)).permute((1, 2, 0))
                       ), (self.image_size,self.image_size))).permute((1,2,0))
        rendering_img = torch.from_numpy(cv2.imread(rendering_filename))
        mash_img = torch.from_numpy(cv2.imread(mash_filename))
        
        return (mask_img, rendering_img, mash_img), label
    
    @staticmethod
    def collate_train_set(items):
        masks, renderings, mashes, labels = [], [], [], []
        
        for (mask_img, rendering_img, mash_img), label in items:
            masks.append(mask_img.unsqueeze(0))
            renderings.append(rendering_img.unsqueeze(0))
            mashes.append(mash_img.unsqueeze(0))
            labels.append(label)
        
        return (torch.cat(masks), torch.cat(renderings).float(), torch.cat(mashes).float()), torch.Tensor(labels)


def prepare_dataset_for_training(root, output, vehicle_obj, batch_size, 
                                 image_size, texture_size, distance):
    os.makedirs(os.path.join(output, 'mash'), exist_ok=True)
    os.makedirs(os.path.join(output, 'rendering'), exist_ok=True)
    
    dataset = DualAttentionDataset(os.path.join(root, 'phy_attack'), distance, train=True)

    dl = DataLoader(dataset, batch_size, collate_fn=DualAttentionDataset.dual_attn_collate) # 5/1/2023 5:53:50 PM: Neil to TVS: Why are we using the PyTorch DataLoader instead of the src/data_loader.py?

    vertices, faces, _ = nr.load_obj(filename_obj=vehicle_obj, texture_size=texture_size, load_texture=True)
    # fully white texture
    textures = torch.ones((batch_size, faces.size(0), texture_size, texture_size, texture_size, 3), device=vertices.device)

    vertices.unsqueeze_(0)
    faces.unsqueeze_(0)

    render = nmr.NeuralRenderer(image_size)
    # render.renderer.renderer.camera_mode = 'look_at'
    idx_select = 10
    for idx, (imgs, masks, (eye, look_at, camera_up)) in enumerate(dl):
        if idx < idx_select:
            continue
        elif idx == idx_select:
            pass
        elif idx > idx_select:
            quit()
        print(f'Image {idx+1} of {len(dataset)}')
        # print(f'vertices:\n{vertices}\n{vertices.shape}\nfaces:\n{faces}\n{faces.shape}\ntextures:\n{textures}\n{textures.shape}')
        if bVerbose:
            pass
            # print(f'eye: {eye}')
            # print(f'type(eye): {type(eye)}')
        render.renderer.renderer.eye = eye.to(vertices.device)
        render.renderer.renderer.camera_direction = look_at.to(vertices.device)
        render.renderer.renderer.camera_up = camera_up.to(vertices.device)
        if bVerbose:
            pass
            # print(f'render.renderer.renderer.eye: {render.renderer.renderer.eye}')
            # print(f'type(render.renderer.renderer.eye): {type(render.renderer.renderer.eye)}')
            # print(f'render.renderer.renderer.camera_direction: {render.renderer.renderer.camera_direction}')
            # print(f'type(render.renderer.renderer.camera_direction): {type(render.renderer.renderer.camera_direction)}')
            # print(f'render.renderer.renderer.camera_up: {render.renderer.renderer.camera_up}')
            # print(f'type(render.renderer.renderer.camera_up): {type(render.renderer.renderer.camera_up)}')
        # 4/30/2023 10:25:07 PM: multiply eye, camera_direction, and camera_up by a constant: start
        # render.renderer.renderer.eye = torch.mul(render.renderer.renderer.eye, .5)
        # render.renderer.renderer.camera_direction = torch.mul(render.renderer.renderer.camera_direction, -1)
        # render.renderer.renderer.camera_up = torch.mul(render.renderer.renderer.camera_up, -2)
        # vertices = torch.mul(vertices,2)
        # faces = torch.mul(faces,.9)
        # textures = torch.mul(textures,10)
        # 4/30/2023 10:25:07 PM: multiply eye, camera_direction, and camera_up by a constant: end
        if bVerbose:
            print('1')
        renderings = render(vertices, faces, textures)
        if bVerbose:
            print('2')
        renderings /= renderings.max()
        
        result = overlay_morph(imgs[0], renderings[0].cpu(), masks[0], image_size) # result.shape = [3, image_size, image_size]
        
        dst = os.path.join(output, 'mash', f'data{idx}.png')
        T.functional.to_pil_image(result).save(dst)
        
        dst = os.path.join(output, 'rendering', f'data{idx}.png')
        T.functional.to_pil_image(renderings[0].cpu()).save(dst)
