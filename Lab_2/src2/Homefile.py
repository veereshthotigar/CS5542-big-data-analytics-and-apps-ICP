from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os

import tensorflow as tf

from medium_show_and_tell_caption_generator.caption_generator import CaptionGenerator
from medium_show_and_tell_caption_generator.model import ShowAndTellModel
from medium_show_and_tell_caption_generator.vocabulary import Vocabulary

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab

import json
from json import encoder

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_path", "/home/vthotigar/gitrepo/CS5542-big-data-analytics-and-apps-ICP/Lab_2/src2/model/show-and-tell.pb", "Model graph def path")
tf.flags.DEFINE_string("vocab_file", "/home/vthotigar/gitrepo/CS5542-big-data-analytics-and-apps-ICP/Lab_2/src2/etc/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "/gitrepo/CS5542-big-data-analytics-and-apps-ICP/Lab_2/src/data/*.jpg",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main(_):
    model = ShowAndTellModel(FLAGS.model_path)
    vocab = Vocabulary(FLAGS.vocab_file)
    filenames = _load_filenames()

    generator = CaptionGenerator(model, vocab)
    gts={}
    res={}

    #for filename in filenames:
    file = open("/home/vthotigar/gitrepo/CS5542-big-data-analytics-and-apps-ICP/Lab_2/src2/captions.txt", "r")
    id=0
    preimg=""
    dic = []
    res_dic=[]
    for line in file:
        url="/home/vthotigar/gitrepo/CS5542-big-data-analytics-and-apps-ICP/Lab_2/src/data/"
        Image = line[0: line.index(".jpg") + 4];
        data = line[line.index(".jpg") + 7:len(line)]
        dic.append({'image_id' : id , 'id' : id ,'caption' : data})
        if preimg != Image:
            preimg=Image
            filename = url+Image
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
                captions = generator.beam_search(image)
                print("Captions for image %s:" % os.path.basename(filename))
                for i, caption in enumerate(captions):
                    # Ignore begin and end tokens <S> and </S>.
                    sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
                    sentence = " ".join(sentence)
                    print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                    res_dic.append({'image_id' : id , 'id' : id ,'caption' : sentence})
            if preimg != "":
                gts[id]=dic
                dic=[]
                res[id]=res_dic
                res_dic=[]
                id = id + 1
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    # set up file names and pathes
    dataDir = '.'
    dataType = 'val2014'
    algName = 'fakecap'
    annFile = '%s/annotations/captions_%s.json' % (dataDir, dataType)
    subtypes = ['results', 'evalImgs', 'eval']
    [resFile, evalImgsFile, evalFile] = \
        ['%s/results/captions_%s_%s_%s.json' % (dataDir, dataType, algName, subtype) for subtype in subtypes]

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    print(gts)
    print(res)
    #gts = {391895: [{'image_id': 391895, 'id': 770337, 'caption': 'A man with a red helmet on a small moped on a dirt road. '}, {'image_id': 391895, 'id': 771687, 'caption': 'Man riding a motor bike on a dirt road on the countryside.'}, {'image_id': 391895, 'id': 772707, 'caption': 'A man riding on the back of a motorcycle.'}, {'image_id': 391895, 'id': 776154, 'caption': 'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. '}, {'image_id': 391895, 'id': 781998, 'caption': 'A man in a red shirt and a red hat is on a motorcycle on a hill side.'}]};
    #res = {391895: [{'image_id': 391895, 'caption': 'man holding a red umbrella in the rain', 'id': 410}, {'image_id': 391895, 'caption': 'A man with a red helmet on a small moped on a dirt road.', 'id': 411}]}
    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate(gts, res)

    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))



def _load_filenames():
    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    logger.info("Running caption generation on %d files matching %s",
                len(filenames), FLAGS.input_files)
    return filenames


if __name__ == "__main__":
    tf.app.run()

#gts = {391895: [{'image_id': 391895, 'id': 770337, 'caption': 'A man with a red helmet on a small moped on a dirt road. '}, {'image_id': 391895, 'id': 771687, 'caption': 'Man riding a motor bike on a dirt road on the countryside.'}, {'image_id': 391895, 'id': 772707, 'caption': 'A man riding on the back of a motorcycle.'}, {'image_id': 391895, 'id': 776154, 'caption': 'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. '}, {'image_id': 391895, 'id': 781998, 'caption': 'A man in a red shirt and a red hat is on a motorcycle on a hill side.'}]};
#res = {391895: [{'image_id': 391895, 'caption': 'man holding a red umbrella in the rain', 'id': 410}, {'image_id': 391895, 'caption': 'A man with a red helmet on a small moped on a dirt road.', 'id': 411}]}


pylab.rcParams['figure.figsize'] = (10.0, 8.0)


encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='.'
dataType='val2014'
algName = 'fakecap'
annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= \
['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]

coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching

