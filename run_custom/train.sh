# python train.py \
# 	--datapath=[path to dataset] \
# 	--content=[path to seed content] \
# 	--canny=[path to edge mask]
python src/train.py \
	--datapath="src/data" \
	--content="src/contents/smile.jpg" \
	--canny="src/contents/smile_edge.jpg" \
	--obj="src/audi_et_te.obj" \
	--faces="src/all_faces.txt"