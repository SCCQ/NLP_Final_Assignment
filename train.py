import os
import numpy as np
import re
from tqdm import tqdm
from argparse import ArgumentParser
from configparser import ConfigParser

from TNet.utils.embeddings import Glove
from TNet.utils.data import Batch
from TNet.utils import get_normalized_batch
from TNet.model import TNet


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--mode', type=str, default='as')
    arg_parser.add_argument('--train_data_fp', type=str, default='datasets/train.txt')
    arg_parser.add_argument('--test_data_fp', type=str, default='datasets/test1.txt')
    arg_parser.add_argument('--embedding_fp', type=str, default='embeddings/TAChinese.txt')
    arg_parser.add_argument('--hparams_fp', type=str, default='./hparams.ini')
    arg_parser.add_argument('--model_name', type=str, default='tnet')
    args = arg_parser.parse_args()

    return vars(args)

def load_hparams(fp):
    conf_parser = ConfigParser()
    conf_parser.read(fp)

    return conf_parser

def clean_log():
    os.system(
        'rm -rf log/*'
    )

def main(**kwargs):
    training_data_fname = kwargs.get('train_data_fp')
    testing_data_fname = kwargs.get('test_data_fp')
    embeddings_fname = kwargs.get('embedding_fp')
    TEST_LABEL = kwargs.get('test_label')

    # rm log
    clean_log()

    hparams = load_hparams(kwargs.get('hparams_fp'))

    # load data
    embeddings = Glove(embeddings_fname)
    batch_generator = Batch(training_data_fname, hparams, mode=kwargs.get('mode'), shuffle=True)
    test_batch_generator = Batch(testing_data_fname, hparams, mode=kwargs.get('mode'))
    model = TNet(hparams, **kwargs)
    epoch = tqdm(range(0, int(hparams['global']['num_epochs'])), desc='epoch')
    highest_acc = 0

    result_data_fname = "./datasets/result.txt"
    with open(testing_data_fname,"r") as fin:
        testing_data = fin.readlines()

    for _ in epoch:
        acc_list = []
        for batch in batch_generator():
            feed_batch = get_normalized_batch(batch, embeddings)
            model.train_on_batch(**feed_batch)

        batchsize = int(hparams["global"]["batch_size"])
        tempcount = 0
        tempresult = []
        for batch in test_batch_generator():
            feed_test_batch = get_normalized_batch(batch, embeddings)
            test_acc = model.test_acc(**feed_test_batch)
            acc_list.append(test_acc)
            pred = model.pred(**feed_test_batch)
            for tempindex in range(batchsize):
                realindex = tempindex+batchsize*tempcount
                temppredindex = pred[tempindex].tolist().index(max(pred[tempindex]))
                if temppredindex == 0:
                    tempresult.append(re.sub(r'/[p0nt]', r'/p', testing_data[realindex]))
                elif temppredindex == 1:
                    tempresult.append(re.sub(r'/[p0nt]', r'/n', testing_data[realindex]))
                else:
                    tempresult.append(re.sub(r'/[p0nt]', r'/0', testing_data[realindex]))
            tempcount+=1
        print("Every batch score:", acc_list)
        if np.mean(acc_list) > highest_acc:
                model.save_model()
                highest_acc = np.mean(acc_list)
                epoch.set_description('highest test acc: {acc:.3f}'.format(acc=highest_acc))
        with open(result_data_fname,"w") as fout:
            fout.writelines(tempresult)

if __name__ == "__main__":
    args = get_args()
    main(**args)