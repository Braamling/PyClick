#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import argparse
import sys

import time

from pyclick.click_models.Evaluation import LogLikelihood, Perplexity
from pyclick.click_models.UBM import UBM
from pyclick.click_models.DBN import DBN
from pyclick.click_models.SDBN import SDBN
from pyclick.click_models.DCM import DCM
from pyclick.click_models.CCM import CCM
from pyclick.click_models.CTR import DCTR, RCTR, GCTR
from pyclick.click_models.CM import CM
from pyclick.click_models.PBM import PBM
from pyclick.click_models.RelevanceUBM import RelevanceUBM
from pyclick.utils.Hdf5ResultsWriter import Hdf5ResultsWriter
from pyclick.utils.Utils import Utils
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser
from pyclick.utils.YandexRelPredChallengeParserhdf5 import YandexRelPredChallengeParserHdf5
from pyclick.utils.YandexRelevanceParser import YandexRelevanceParser

__author__ = 'Ilya Markov'


def train():
    print "==============================="
    print "This is an example of using PyClick for training and testing click models."
    print "==============================="

    if len(sys.argv) < 4:
        print "USAGE: %s <click_model> <dataset> <relevance> <sessions_max>" % sys.argv[0]
        print "\tclick_model - the name of a click model to use."
        print "\tdataset - the path to the dataset from Yandex Relevance Prediction Challenge"
        print "\trelevance - the path to the relevance judgments from Yandex Relevance Prediction Challenge"
        print "\tsessions_max - the maximum number of one-query search sessions to consider"
        print ""
        sys.exit(1)

    click_model = globals()[args.click_model]()

    train_sessions = YandexRelPredChallengeParserHdf5().parse(args.train_dataset, args.sessions_max)
    # This is still hardcoded, but should be changed to a fixed length test dataset
    test_sessions = YandexRelPredChallengeParserHdf5().parse(args.test_dataset, sessions_max=1000000, first_session=1000000)

    print "-------------------------------"
    print "Training on %d search sessions." % (len(train_sessions))
    print "-------------------------------"

    start = time.time()
    click_model.train(train_sessions)
    end = time.time()
    print "\tTrained %s click model in %i secs:\n%r" % (click_model.__class__.__name__, end - start, click_model)

    print "-------------------------------"
    print "Testing on %d search sessions." % (len(test_sessions))
    print "-------------------------------"

    loglikelihood = LogLikelihood()

    start = time.time()

    if args.test_predictions_file is not None:
        resultsWriter = Hdf5ResultsWriter(args.test_predictions_file)
        resultsWriter.write(click_model, test_sessions)

    end = time.time()
    print "\tWritten test results to %s; time: %i secs" % (args.test_predictions_file, end - start)

    start = time.time()

    ll_value = loglikelihood.evaluate(click_model, test_sessions)
    end = time.time()
    print "\tlog-likelihood: %f; time: %i secs" % (ll_value, end - start)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is an example of using PyClick for training and testing click models.')
    parser.add_argument('--click_model', type=str, default=None, help="the name of a click model to use.")
    parser.add_argument('--train_dataset', type=str, default=None,
                        help="the path to the dataset from Yandex Relevance Prediction Challenge")
    parser.add_argument('--test_dataset', type=str, default=None,
                        help="the path to the dataset from Yandex Relevance Prediction Challenge")
    parser.add_argument('--sessions_max', type=int, default=None,
                        help="the maximum number of one-query search sessions to consider")
    parser.add_argument('--test_predictions_file', type=str, default=None,
                        help="A hdf5 file path to store the output for the test dataset.")
    args = parser.parse_args()

    train()
