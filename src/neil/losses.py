from functools import reduce

import cv2
import numpy as np
import torch


class LossMIDU(torch.nn.Module):
    def __init__(self, cam_edge):
        super(LossMIDU, self).__init__()
        self.cam_edge = cam_edge
        
    def forward(self, x1):
        # print(torch.gt(x1, torch.ones_like(x1) * 0.1).float())
        
        x1 = torch.tanh(x1)
        self.vis = np.zeros((self.cam_edge, self.cam_edge))
        
        loss = []
        # print(x1)
        for i in range(self.cam_edge):
            for j in range(self.cam_edge):
                if x1[i][j] > 0 and not self.vis[i][j]:
                    point = []
                    n = self.dfs(x1, i, j, point)
                    # print(n)
                    # print(point)
                    loss.append( reduce(lambda x, y: x + y, point) / (self.cam_edge * self.cam_edge + 1 - n) )
        # print(vis)
        if len(loss) == 0:
            return torch.zeros(1).cuda()
        return reduce(lambda x, y: x + y, loss) / len(loss)

    def dfs(self, x1, x, y, points):
        points.append(x1[x][y])
        self.vis[x][y] = 1
        n = 1
        # print(x, y)
        if x+1 < self.cam_edge and x1[x+1][y] > 0 and not  self.vis[x+1][y]:
            n += self.dfs(x1, x+1, y, points)
        if x-1 >= 0 and x1[x-1][y] > 0 and not  self.vis[x-1][y]:
            n += self.dfs(x1, x-1, y, points)
        if y+1 < self.cam_edge and x1[x][y+1] > 0 and not  self.vis[x][y+1]:
            n += self.dfs(x1, x, y+1, points)
        if y-1 >= 0 and x1[x][y-1] > 0 and not  self.vis[x][y-1]:
            n += self.dfs(x1, x, y-1, points)
        return n


class ContentDiffLoss(torch.nn.Module):
    def __init__(self, d1, d2, content_src: str, canny_src: str):
        super(ContentDiffLoss, self).__init__()
        
        self.d1 = d1
        self.d2 = d2
        self.content = torch.from_numpy(cv2.imread(content_src))
        self.canny = (torch.from_numpy(cv2.imread(canny_src)) >= 1).int()
    
    def forward(self, x):
        return  self.d1 * torch.sum(self.canny * torch.pow(x - self.content, 2)) + self.d2 * torch.sum((1 - self.canny) * torch.pow(x - self.content, 2))


class SmoothLoss(torch.nn.Module):
    def __init__(self, t):
        super(SmoothLoss, self).__init__()
        
        self.t = t
            
    def forward(self, img, mask):
        s1 = torch.pow(img[:, 1:, :-1, :] - img[:, :-1, :-1, :], 2)
        s2 = torch.pow(img[:, :-1, 1:, :] - img[:, :-1, :-1, :], 2)
        mask = mask[:, :-1, :-1]
        
        mask = mask.unsqueeze(1)
        return self.t * torch.sum(mask * (s1 + s2))
