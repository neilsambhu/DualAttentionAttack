import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.neil.dataset import DASTrainingSet
from src.neil.losses import SmoothLoss, ContentDiffLoss, LossMIDU
from src.neil.model import DASModel


def train(args):
    # Dataset
    train_set = DASTrainingSet(args.masks_dir, args.dataset, image_size=args.image_size)
    dataloader = DataLoader(train_set, args.batch_size, collate_fn=DASTrainingSet.collate_train_set)

    # Model
    model = DASModel(args.vehicle_object, args.texture_size).cuda()

    # Optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    # Loss
    def loss(x, texture_params, texture_img, gt_mask):
        l1 = LossMIDU(cam_edge=args.cam_edge)
        l2 = ContentDiffLoss(args.d1, args.d2, content_src=args.content_src, canny_src=args.canny_src)
        l3 = SmoothLoss(t=args.t)
        
        # return l1(x) + 0 * l2(texture_params) + l3(texture_img, gt_mask)
        return l1(x) + l3(texture_img, gt_mask)

    # Training
    for e in range(args.epochs):
        print(f'Epoch {e+1} of {args.epochs}')
        
        # batch-wise training
        progressbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, ((masks, renderings, mashes), labels) in progressbar:
            predictions = model(mashes.permute((0,3,1,2)).cuda(), labels.cuda())
            loss_value = loss(predictions, model.texture_param, renderings, masks)
            # progressbar.set_description(f'Batch {idx} of {len(dataloader)}; Loss: {loss_value.item()}')
            
            if loss_value.item() != 0:
                optimiser.zero_grad()
                loss_value.backward(retain_graph=True)
                optimiser.step()

                if (idx+1) % args.save_every == 0 or idx == len(dataloader):
                    torch.save(model.state_dict(), os.path.join(args.model_dst, f'ckpt_{idx+1}.pt'))
