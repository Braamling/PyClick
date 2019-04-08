#
# Copyright (C) 2019  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from pyclick.utils.RelevanceLookup import RelevanceLookup

__author__ = 'Bram van den Akker'


class YandexRelevanceParser:
    """
    A parser for the publicly available dataset, released by Yandex (https://www.yandex.com)
    for the Relevance Prediction Challenge (http://imat-relpred.yandex.ru/en).
    """

    @staticmethod
    def parse(relevance_filename):
        lookup_table = RelevanceLookup()

        relevance_file = open(relevance_filename)

        for line in relevance_file:
            query, _, document, rel = line.rstrip().split("\t")

            # Relevance can different for different sessions, we always take the highest relevance judgment.
            rel = max(int(rel), lookup_table(query, document))
            lookup_table.set_relevance(query, document, rel)

        return lookup_table
