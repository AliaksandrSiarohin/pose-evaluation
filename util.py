from imageio import mimread, imread, mimsave
import numpy as np
import warnings


def frames2array(file, is_video, image_shape=None, column=0):
    if is_video:
        if file.endswith('.png') or file.endswith('.jpg'):
            ### Frames is stacked (e.g taichi ground truth)
            image = imread(file)
            if image.shape[2] == 4:
                image = image[..., :3]

            video = np.moveaxis(image, 1, 0)
            video = video.reshape((-1, ) + image_shape + (3, ))
            video = np.moveaxis(video, 1, 2)
        elif file.endswith('.gif') or file.endswith('.mp4'):
            video = np.array(mimread(file))
        else:
            warnings.warn("Unknown file extensions  %s" % file, Warning)
            return []
    else:
        ## Image is given, interpret it as video with one frame
        image = imread(file)
        if image.shape[2] == 4:
            image = image[..., :3]
        video = video[np.newaxis]

    if image_shape is None:
        return video
    else:
        ### Several images stacked together select one based on column number
        return video[:, :, (image_shape[1] * column):(image_shape[1] * (column + 1))]


def draw_video_with_kp(video, kp_array):
    from skimage.draw import circle
    video_array = np.copy(video)
    for i in range(len(video_array)):
        for kp_ind, kp in enumerate(kp_array[i]):
            rr, cc = circle(kp[1], kp[0], 2, shape=video_array.shape[1:3])
            video_array[i][rr, cc] = (255, 255, 255)
    return video_array
