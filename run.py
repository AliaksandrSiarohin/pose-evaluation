from imageio import mimread, imread, mimsave
import numpy as np
import warnings
import pandas as pd
import os
from tqdm import tqdm
from skimage.transform import resize


def extract_facial_landmarks(in_folder, image_shape, column):
    import face_alignment
    from skimage import io

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), image_shape, column)
        for i, frame in enumerate(video):
            kp = fa.get_landmarks(frame)[0]
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(kp)

    return pd.DataFrame(out_df)


def extract_face_id_embeddings(in_folder, image_shape, column):
    from evaluation.OpenFacePytorch.loadOpenFace import prepareOpenFace
    from torch.autograd import Variable
    import torch

    net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), image_shape, column)
        for i, frame in enumerate(video):
            frame = frame[..., ::-1]
            frame = resize(frame, (96, 96))
            frame = np.transpose(frame, (2, 0, 1))
            with torch.no_grad():
                frame = Variable(torch.Tensor(frame)).cuda()
                frame = frame.unsqueeze(0)
                id_vec = net(frame)[0].data.cpu().numpy()

            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(id_vec)

    return pd.DataFrame(out_df)


def extract_pose(in_folder, image_shape, column):
    from evaluation.pose_estimation.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
    import torch
    from evaluation.pose_estimation.network.rtpose_vgg import get_model
    from evaluation.pose_estimation.network.post import decode_pose

    weight_name = 'pose_estimation/network/weight/pose_model.pth'

    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_name))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), image_shape, column)
        for i, frame in tqdm(enumerate(video)):
            frame = frame[..., ::-1]# B,G,R order
            multiplier = get_multiplier(frame)

            with torch.no_grad():
                orig_paf, orig_heat = get_outputs(multiplier, frame, model, 'rtpose')

                # Get results of flipped image
                swapped_img = frame[:, ::-1, :]
                flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, model, 'rtpose')

                # compute averaged heatmap and paf
                paf, heatmap = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)

            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            _, _, joint_list, _ = decode_pose(frame, param, heatmap, paf)
            joint_list = np.array(joint_list)
            tmp = -np.ones((18, 2))
            tmp[joint_list[:, -1].astype(int)] = joint_list[:, :2]
            joint_list = tmp

            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(joint_list)


    return pd.DataFrame(out_df)


def extract_pose_embeddings(in_folder, image_shape, column):
    from evaluation.Person_reID_baseline_pytorch.model import ft_net_dense
    from torch.autograd import Variable
    import torch

    net = ft_net_dense(751)
    net.load_state_dict('')

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), image_shape, column)
        for i, frame in enumerate(video):
            frame = frame[..., ::-1]
            frame = resize(frame, (96, 96))
            frame = np.transpose(frame, (2, 0, 1))
            with torch.no_grad():
                frame = Variable(torch.Tensor(frame)).cuda()
                frame = frame.unsqueeze(0)
                id_vec = net(frame)[0].data.cpu().numpy()

            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(id_vec)

    return pd.DataFrame(out_df)



if __name__ == "__main__":
    df = extract_pose('test', (64, 64), 0)
    df.sort_values(by=['file_name', 'frame_number'])


