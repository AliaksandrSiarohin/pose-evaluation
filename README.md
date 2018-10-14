## Evaluation

Evaluation is based on several pretrained models for faces and human
poses. We evaluate, if the movement is the same as in target video and
if the identity is preserved.

### Faces

For extracting face keypoints we use: https://github.com/1adrianb/face-alignment

```
cd face-aligment
pip install -r requirments.txt
python setup.py install
```


For extracting identity embedding we use: https://github.com/thnkim/OpenFacePytorch

```
git clone https://github.com/thnkim/OpenFacePytorch
```

### Poses

For extracting pose keypoints we use: https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation
```
git clone https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation
```
Download model https://yadi.sk/d/0L-PgAaGRKgkJA
```
mkdir pytorch_Realtime_Multi-Person_Pose_Estimation/network/weight
mv pose_model.pth pytorch_Realtime_Multi-Person_Pose_Estimation/network/weight/pose_model.pth
```
Optional:
```
git clone https://github.com/cocodataset/cocoapi
cd cocoapi/PythonAPI/
python setup.py install

mv pytorch_Realtime_Multi-Person_Pose_Estimation pose_estimation
```


For extracting identity embedding we use: https://github.com/layumi/Person_reID_baseline_pytorch
