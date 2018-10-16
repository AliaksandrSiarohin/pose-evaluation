## Evaluation

Evaluation is based on several pretrained models for faces and human
poses. We evaluate, if the movement is the same as in target video and
if the identity is preserved.

```
git clone --recursive https://github.com/AliaksandrSiarohin/pose-evaluation
```

### Faces

For extracting face keypoints we use: https://github.com/1adrianb/face-alignment

```
cd face-aligment
pip install -r requirements.txt
python setup.py install
```

```
python extract.py --in_folder mtm/data/nemo/test --out_file nemo_pose_gt.pkl --is_video --type face_pose
python extract.py --in_folder log/nemo/reconstruction --out_file nemo_pose_gen.pkl --is_video --column 2 --type face_pose
```

For extracting identity embedding we use: https://github.com/thnkim/OpenFacePytorch

```
python extract.py --in_folder mtm/data/nemo/test --out_file nemo_id_gt.pkl --is_video --type face_id
python extract.py --in_folder log/nemo/reconstruction --out_file nemo_id_gen.pkl --is_video --column 2 --type face_id
```

### Poses

For extracting pose keypoints we use: https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation

Download model https://yadi.sk/d/0L-PgAaGRKgkJA
```
mkdir pytorch_Realtime_Multi-Person_Pose_Estimation/network/weight
mv pose_model.pth pytorch_Realtime_Multi-Person_Pose_Estimation/network/weight/pose_model.pth
```

```
python extract.py --in_folder mtm/data/taichi/test --out_file taichi_pose_gt.pkl --is_video --type body_pose
python extract.py --in_folder log/taichi/reconstruction --out_file taichi_pose_gen.pkl --is_video --column 2 --type body_pose
```

For extracting identity embedding we use: https://github.com/layumi/Person_reID_baseline_pytorch

Download model https://yadi.sk/d/jAPhvFEFp6qzIw
```
python extract.py --in_folder mtm/data/taichi/test --out_file taichi_id_gt.pkl --is_video --type body_id
python extract.py --in_folder log/taichi/reconstruction --out_file taichi_id_gen.pkl --is_video --column 2 --type body_id
```
