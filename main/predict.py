import os
import shutil
import glob
from pretrainedmodels import resnet152
import numpy as np
import pretrainedmodels.utils as util
import torch
import torch.nn as nn
import opts
from models import EncoderRNN, DecoderRNN, S2VTAttModel
import mix.utils as utils
import json
import cv2
from language import *

C, H, W = 3, 224, 224


def extract_frames(video, dst):
    cap = cv2.VideoCapture(video)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (400, 300))
        cv2.imwrite(dst + '/' + 'kang' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def extract_feats(video_path):
    C, H, W = 3, 224, 224
    model = resnet152(pretrained='imagenet')
    load_image_fn = util.LoadTransformImage(model)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model.last_linear = util.Identity()
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    dir_fc = '../data/sample'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % dir_fc)

    video_id = video_path.split("/")[-1].split(".")[0]
    print(video_id)

    os.mkdir(dir_fc + '/' + video_id)
    dst = '../data/sample/' + video_id
    extract_frames(video_path, dst)

    image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
    samples = np.round(np.linspace(0, len(image_list) - 1, 40))
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
    return img_feats


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    opt['saved_model'] = '../data/save/model_10.pth'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], bidirectional=bool(opt["bidirectional"]),
                         # opt["bidirectional"] eski hali
                         input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
    decoder = DecoderRNN(16860, opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                         input_dropout_p=opt["input_dropout_p"],
                         rnn_dropout_p=opt["rnn_dropout_p"],
                         bidirectional=bool(opt["bidirectional"]))  # opt["bidirectional"] eski hali

    model = S2VTAttModel(encoder, decoder).cuda()
    model.load_state_dict(torch.load(opt['saved_model']))
    model.eval()

    video_path = '../data/sample_video/video00.mp4'

    image_feats = extract_feats(video_path)

    fc_feat = torch.from_numpy(image_feats).type(torch.FloatTensor)
    fc_feat = torch.unsqueeze(fc_feat, 0).cuda()

    with torch.no_grad():
        seq_probs, seq_preds = model(fc_feat,  mode='inference', opt=opt)

    vocab = json.load(open('../data/info.json'))['ix_to_word']

    prediction = utils.decode_sequence(vocab, seq_preds)
    speak_main(prediction)
    print('Tahmin: ', prediction[0])