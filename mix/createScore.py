from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.gleu_score import sentence_gleu


def splitDic(dic1, dic2, i): #finding captions in dictionaries
    ref = []
    for j in dic1[i]:
        str1 = j['caption']
        ref.append(str1.split())
    str2 = dic2[i][0]['caption']
    sam = str2.split()
    return ref, sam


def makeIDList(dic2): #finding ids of videos from samples
    temp = []
    for key, value in dic2.items():
        temp.append(key)
    return temp


def makeNewgts(dic1, IDs): #creating new gts from old gts according to the ids of the videos
    newGts = {x: dic1[x] for x in IDs}
    return newGts


def Scoring(dic1, dic2, IDs):# performing score operations between gts and sample and adding them to the list
    result = []
    counter = 0
    bleu1_top = 0
    bleu2_top = 0
    bleu3_top = 0
    bleu4_top = 0
    gleu_top = 0
    chrf_top = 0

    for i in IDs:
        ref, sam = splitDic(dic1, dic2, i)
        counter += 1

        #bleu scoring
        bleu1 = sentence_bleu(ref, sam, weights=(1, 0, 0, 0))
        bleu1_top += bleu1

        bleu2 = sentence_bleu(ref, sam, weights=(0.5, 0.5, 0, 0))
        bleu2_top += bleu2

        bleu3 = sentence_bleu(ref, sam, weights=(0.33, 0.33, 0.33, 0))
        bleu3_top += bleu3

        bleu4 = sentence_bleu(ref, sam, weights=(0.25, 0.25, 0.25, 0.25))
        bleu4_top += bleu4

        #gleu and chrf scoring
        gleu = sentence_gleu(ref, sam)
        gleu_top += gleu

        counterCHRF = 0
        for sentence in ref:
            counterCHRF += 1
            chrf = sentence_chrf(sentence, sam)
            chrf_top += chrf
        chrf_top = chrf_top/counterCHRF

    ort_Bleu1 = bleu1_top / counter
    ort_Bleu2 = bleu2_top / counter
    ort_Bleu3 = bleu3_top / counter
    ort_Bleu4 = bleu4_top / counter
    ort_chrf = chrf_top / counter
    ort_gleu = gleu_top / counter

    result.append(ort_Bleu1)
    result.append(ort_Bleu2)
    result.append(ort_Bleu3)
    result.append(ort_Bleu4)
    result.append(ort_chrf)
    result.append(ort_gleu)

    print('Cumulate 1-gram :%f' % ort_Bleu1)
    print('Cumulate 2-gram :%f' % ort_Bleu2)
    print('Cumulate 3-gram :%f' % ort_Bleu3)
    print('Cumulate 4-gram :%f' % ort_Bleu4)
    print('CHRF Scoring : %f' % ort_chrf)
    print('GLEU Scoring : %f' % ort_gleu)
    return result

'''Fonkisyonların kullanılma önceliği:
IDs = makeIDList(samples)
newGts = makeNewgts(gts, IDs)
resultList = Scoring(newGts, samples, IDs)
'''