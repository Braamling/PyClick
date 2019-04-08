#
# Copyright (C) 2019  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from collections import defaultdict

__author__ = 'Bram van den Akker'


class RelevanceLookup:
    """
    A lookup class for query-document pair relevance judgments.
    """

    def __init__(self):
        self.relevance = defaultdict(defaultdict)

    def set_relevance(self, query, document, relevance):
        """
        This method set the relevance for a specific query-document pair.

        :param query: The query id as an integer, to be used as identifier to obtain the relevance label
        :param document: The document id as an integer, to be used as identifier to obtain the relevance label
        :param relevance: The relevance in any format, for the Yandex dataset an integer [0-1] is expected.
        :return: None
        """
        self.relevance[query][document] = relevance

    def __call__(self, query, document):
        """
        This method provides the relevance label of a specific query document pair. In case the relevance has not been
        set, the method will return -1.

        :param query: The query id as an integer, to be used as identifier to obtain the relevance label
        :param document: The document id as an integer, to be used as identifier to obtain the relevance label
        :return: The relevance label in any format, for the Yandex dataset it will be an integer [-1-+1]
        """
        try:
            return self.relevance[query][document]
        except:
            return -1
