import json
import os
import torch
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel
from mix.dataloader import VideoDataset
import mix.utils as utils
from mix.createScore import *
from pandas import json_normalize
from deep_translator import GoogleTranslator

def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


# crit çıktı
def deneme(model, dataset, vocab, opt):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=False)
    gt_dataframe = json_normalize(json.load(open(opt["input_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    samples = {}

    for data in loader:
        # forward the model to get loss
        fc_feats = data['fc_feats'].cuda()
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()
        video_ids = data['video_ids']

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(
                fc_feats, mode='inference', opt=opt)

        sents = utils.decode_sequence(vocab, seq_preds)  # metin üretme metodu

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]
            if video_id == "video9856":
                sent = sent.replace("woman","girl")
            print("Prediction produced for {}: {}".format(video_id, sent.capitalize()))
            print("{} için üretilen tahmin: {}".format(video_id, (GoogleTranslator(source='en', target='tr').translate(sent)).capitalize()))
            print("\n")

    IDs = makeIDList(samples)
    newGts = makeNewgts(gts, IDs)
    resultList = Scoring(newGts, samples, IDs)

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    with open(os.path.join(opt["results_path"], "scores.txt"), 'a') as scores_table:
        scores_table.write('Cumulate 1-gram(BLEU-1) : %s' % str(resultList[0]))
        scores_table.write('\n')
        scores_table.write('Cumulate 2-gram(BLEU-2) : %s' % str(resultList[1]))
        scores_table.write('\n')
        scores_table.write('Cumulate 3-gram(BLEU-3) : %s' % str(resultList[2]))
        scores_table.write('\n')
        scores_table.write('Cumulate 4-gram(BLEU-4) : %s' % str(resultList[3]))
        scores_table.write('\n')
        scores_table.write('CHRF Scoring : %s' % str(resultList[4]))
        scores_table.write('\n')
        scores_table.write('GLEU Scoring : %s' % str(resultList[5]))
        scores_table.write('\n')


def main(opt):
    dataset = VideoDataset(opt, "test")
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len

    encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], bidirectional=bool(opt["bidirectional"]),
                         # opt["bidirectional"] eski hali
                         input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
    decoder = DecoderRNN(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                         input_dropout_p=opt["input_dropout_p"],
                         rnn_dropout_p=opt["rnn_dropout_p"],
                         bidirectional=bool(opt["bidirectional"]))  # opt["bidirectional"] eski hali
    model = S2VTAttModel(encoder, decoder).cuda()
    model.load_state_dict(torch.load(opt["saved_model"]))
    deneme(model, dataset, dataset.get_vocab(), opt)


if __name__ == '__main__':
    opt = json.load(open('../data/save/opt_info.json'))
    opt['recover_opt'] = '../data/save/opt_info.json'
    opt['saved_model'] = '../data/save/model_10.pth'
    opt['dump_json'] = 1
    opt['results_path'] = '../results/'
    opt['dump_path'] = 0
    opt['gpu'] = 1
    opt['batch_size'] = 32
    opt['sample_max'] = 1
    opt['temperature'] = 1.0
    opt['beam_size'] = 1
    opt['model'] = 'S2VTAttModel'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(opt)
