# Dual Attention Attack
4/27/2023 11:21:21 AM: prepare dataset command
```
python main.py prepare --dataset "src/data" --output "~/data1/DAS_output" --vehicle-object "assets/object_files/audi/audi_et_te.obj" --batch-size 1 --image-size 800 --texture-size 6 --distance -1 |& tee out.txt
```
4/27/2023 12:28:06 PM: associated functions:  
dataset.py: `renderings = render(vertices, faces, textures)`  
nmr_test.py: `def forward(self, vertices, faces, textures=None):`