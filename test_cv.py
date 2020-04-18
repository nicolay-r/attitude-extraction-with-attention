#!/usr/bin/python
from os import path

import io_utils
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from experiments.rusentrel.utils import iter_by_same_size_parts_cv


def check(train, test):
    for x in test:
        if x in train:
            print "{}'th-doc in train. Failed".format(x)

    for x in train:
        if x in test:
            print "{}'th-doc in test. Failed".format(x)


def check_2(prior_tests, test):
    for x in test:
        if x in prior_tests:
            print "{}'th-doc in prior tests. Failed".format(x)


stemmer = MystemWrapper()
synonyms = RuSentRelSynonymsCollection.read_collection(stemmer=stemmer)

stat = []
for doc_id in RuSentRelIOUtils.iter_collection_indices():

    entities = RuSentRelDocumentEntityCollection.read_collection(
        doc_id=doc_id,
        stemmer=stemmer,
        synonyms=synonyms)

    news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)

    stat.append((doc_id, news.sentences_count()))

stat_filepath = path.join(io_utils.get_data_root(), "rusentrel_docs_stat.txt")
with open(stat_filepath, 'w') as f:
    for doc_index, s_count in stat:
        f.write("{}: {}\n".format(doc_index, s_count))


it = iter_by_same_size_parts_cv(3)
prior_tests = []
for train, test in it:
    print 'TRAIN:', train
    print 'TEST:', test

    check(train, test)
    check_2(prior_tests, test)

    prior_tests += test


