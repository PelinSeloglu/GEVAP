import shutil
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse
import torch
from torch import nn
import pretrainedmodels
from pretrainedmodels import utils
import cv2


C, H, W = 3, 224, 224


def extract_frames(video, dst):
    cap = cv2.VideoCapture(video)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (400, 300))
        cv2.imwrite(dst+ '/'+ 'kang' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % dir_fc)
    #video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))
    video_list = ['data/train-video/video00.mp4', 'data/train-video/video01.mp4', 'data/train-video/video02.mp4']
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        #dst = params['model'] + '_' + video_id

        new_fold = dir_fc + '/' + video_id
        if not os.path.isdir(new_fold):
            os.mkdir(new_fold)

        dst = 'data/images/' + video_id
        extract_frames(video, dst)

        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        samples = np.round(np.linspace(0, len(image_list) - 1, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]

        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        outfile = os.path.join(dir_fc, video_id + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        shutil.rmtree(dst)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/images', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/train-video', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)

    model = model.cuda()
    extract_feats(params, model, load_image_fn)
