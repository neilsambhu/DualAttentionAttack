import torch
import neural_renderer

from ..grad_cam import CAM


class DASModel(torch.nn.Module):
    def __init__(self, obj_file: str, texture_size: int):
        super(DASModel, self).__init__()
        
        _, faces, _ = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)
        self.texture_param = torch.nn.parameter.Parameter(
            torch.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3)) * -0.9
        )
        self.camera = CAM()
    
    def forward(self, x, labels):
        predictions, _ = self.camera(x, labels, log_dir=None)
        
        return predictions
