## Evaluation

Evaluation is based on several pretrained models for faces and human
poses. We evaluate, if the movement is the same as in target video and
if the identity is preserved.

```
git clone --recursive https://github.com/AliaksandrSiarohin/pose-evaluation
```

The general pipeline first compute the statistics for real data and generated using the script ```extract.py```. Then use script ```cmp.py``` to measure the final score.

### Faces

For extracting face keypoints (AKD) we use: https://github.com/1adrianb/face-alignment

```
cd face-aligment
pip install -r requirements.txt
python setup.py install
```

```
python extract.py --in_folder /path/to/test --out_file pose_gt.pkl --is_video --type face_pose --image_shape 256,256
python extract.py --in_folder /path/to/generated/png --out_file pose_gen.pkl --is_video --type face_pose --image_shape 256,256
python cmp_kp.py pose_gt.pkl pose_gen.pkl
```

For extracting identity embedding (AED) we use: https://github.com/thnkim/OpenFacePytorch

```
python extract.py --in_folder /path/to/test --out_file id_gt.pkl --is_video --type face_id --image_shape 256,256
python extract.py --in_folder /path/to/generated/png --out_file id_gen.pkl --is_video --type face_id --image_shape 256,256
python cmp.py id_gt.pkl id_gen.pkl
```

### Poses

For extracting pose keypoints (AKD) we use: https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation

Download model https://yadi.sk/d/0L-PgAaGRKgkJA
```
mkdir pytorch_Realtime_Multi-Person_Pose_Estimation/network/weight
mv pose_model.pth pytorch_Realtime_Multi-Person_Pose_Estimation/network/weight/pose_model.pth
```

```
python extract.py --in_folder /path/to/test --out_file pose_gt.pkl --is_video --type body_pose --image_shape 256,256
python extract.py --in_folder /path/to/generated/png --out_file pose_gen.pkl --is_video --type body_pose --image_shape 256,256
python cmp_with_missing.py pose_gt.pkl pose_gen.pkl
```

For extracting identity embedding (AED) we use: https://github.com/layumi/Person_reID_baseline_pytorch

Download model https://yadi.sk/d/jAPhvFEFp6qzIw
```
python extract.py --in_folder /path/to/test --out_file id_gt.pkl --is_video --type body_id --image_shape 256,256
python extract.py --in_folder /path/to/generated/png --out_file id_gen.pkl --is_video --type body_id --image_shape 256,256
python cmp.py id_gt.pkl id_gen.pkl
```
