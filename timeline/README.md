# Dual Attention Attack
4/27/2023 11:21:21 AM: prepare dataset command
```
python main.py prepare --dataset "src/data" --output "~/data1/DAS_output" --vehicle-object "assets/object_files/audi/audi_et_te.obj" --batch-size 1 --image-size 800 --texture-size 6 --distance -1 |& tee out.txt
```
4/27/2023 12:28:06 PM: associated functions:  
dataset.py: `renderings = render(vertices, faces, textures)`  
nmr_test.py: `def forward(self, vertices, faces, textures=None):`  
4/28/2023 1:23:15 PM: load mash image at specified index in sandbox.ipynb
4/30/2023 5:44:12 PM: new prepare dataset command
```
python -u main.py prepare --dataset "src/data" --output "~/data1/DAS_output" --vehicle-object "assets/object_files/audi/audi_et_te.obj" --batch-size 1 --image-size 800 --texture-size 6 --distance -1 |& tee out.txt
```
4/30/2023 10:30:31 PM: `eye` multiplied by 2, 10, and 100 don't change the output of any of the images in sandbox.ipynb.  
5/1/2023 12:18:20 PM: change from image at index 10 to index 0. `eye` still has no effect.  
5/1/2023 12:24:13 PM: TODO: change something in dataset.py to verify sandbox.ipynb is reading updated values.  
5/1/2023 12:29:13 PM: assume link between files is accurate. Delete files in directory to ensure new files are created.  
5/1/2023 2:02:42 PM: new prepare dataset command
```
python -u main.py prepare --dataset "src/data" --output "/home/nsambhu/data1/DAS_output" --vehicle-object "assets/object_files/audi/audi_et_te.obj" --batch-size 1 --image-size 800 --texture-size 6 --distance -1 |& tee out.txt
```
5/1/2023 2:09:13 PM: link between files is not accurate. `dataset.py` did not properly connect to `sandbox.ipynb`. TODO: redo changes to `eye` and other variables to see which variables affect rotation of the rendering.  
5/1/2023 2:13:11 PM: `eye` affects the zoom in or out from the object. Larger values result in a smaller object.  
5/1/2023 2:16:27 PM: `camera_direction` does not affect the object. Negative values make the object disappear.  Same for `camera_up`.  
5/1/2023 2:24:04 PM: `vertices` affects zoom like `eye`.  
5/1/2023 2:26:35 PM: `faces` affects the quality of the curves of the object. Smaller values result in a lower quality render.  
5/1/2023 2:38:27 PM: `textures` does not affect the render.  
5/1/2023 2:47:13 PM: `self.renderer.perspective=False` makes the render larger.  
5/1/2023 6:45:32 PM: train command
```
python -u main.py train --vehicle-object "assets/object_files/audi/audi_et_te.obj" --texture-size 6 --epochs 1 --batch-size 1 --masks_dir "./src/data" --dataset "/home/nsambhu/data1/DAS_output" --content-src "assets/contents/smile.jpg" --canny-src "assets/contents/smile_edge.jpg" --image-size 800 --cam-edge 7 --d1 0.9 --d2 0.1 --t 0.0001 --save-every 1000 --model-dst "models"
```