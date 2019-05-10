#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from pyclick.search_session.SearchResult import SearchResult
from pyclick.search_session.SearchSession import SearchSession
import h5py
__author__ = 'Ilya Markov, Bart Vredebregt, Nick de Wolf'


class YandexRelPredChallengeParserHdf5:

    @staticmethod
    def parse(sessions_filename, sessions_max=None, first_session=0):
        """

        :param sessions_filename:
        :param sessions_max:
        :param judgements_lookup:
        :param first_session:
        :return:
        """
        with h5py.File(sessions_filename, "r") as sessions_file:

            sessions = []

            for relevance, clicks in sessions_file['data'][first_session:first_session+sessions_max]:
                    session = SearchSession(-1)

                    session.web_results = [SearchResult(-1, c, r) for (c, r) in zip(clicks, relevance)]

                    sessions.append(session)


            return sessions
