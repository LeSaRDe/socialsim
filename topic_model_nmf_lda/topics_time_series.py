import json
import logging
from datetime import datetime, timedelta
import os
import time
import sqlite3
import multiprocessing
import threading
import math
from gensim.models import KeyedVectors
import numpy as np
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_time_series_data_db_format = g_time_series_data_path_prefix + '{0}/{1}_nec_data.db'
g_time_series_data_txt_by_uid_db_format = g_time_series_data_path_prefix + '{0}/{1}_txt_by_uid.db'
g_unique_raw_cln_by_uid_json_path = g_time_series_data_path_prefix + 'unique_raw_cln_by_uid.json'
g_topic_sum_format = g_time_series_data_path_prefix + '{0}/{1}_topic_sum'
g_topic_weights_by_uid_format = g_time_series_data_path_prefix + '{0}/{1}_topic_weights_by_uid.json'
g_time_format = '%Y%m%d'
g_time_int_str = '20180718_20180724'

# n_samples = 2000
# n_features = 1000
g_n_features = None
# n_components = 10
g_n_components_nmf = 50
# n_components_lda = 10
g_n_top_words = 40
# learning_method_lda = 'batch'
# g_en_lda = True
# g_en_nmf = True


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-g_n_top_words - 1:-1]])


def topics_by_uid_for_one_time_int(time_int, tfidf_model):
    time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
    txt_db_conn = sqlite3.connect(g_time_series_data_txt_by_uid_db_format.format(time_int_str, time_int_str))
    txt_db_cur = txt_db_conn.cursor()
    sql_str = '''SELECT uid, raw_cln FROM wh_txt_by_uid'''
    txt_db_cur.execute(sql_str)
    l_txt_recs = txt_db_cur.fetchall()
    d_txt_by_uid = dict()
    m_doc_term = []
    for idx, rec in enumerate(l_txt_recs):
        uid = rec[0]
        raw_cln = rec[1]
        raw_cln_tfidf_vec = tfidf_model.transform([raw_cln])
        d_txt_by_uid[uid] = {'mat_idx': idx, 'raw_cln': raw_cln}
        m_doc_term.append(list(np.asarray(raw_cln_tfidf_vec.todense()).flatten()))
    m_doc_term = np.asarray(m_doc_term)

    nmf = NMF(n_components=g_n_components_nmf,
              random_state=1,
              alpha=0.1,
              l1_ratio=0.5,
              solver='cd',
              beta_loss='frobenius',
              max_iter=1000).fit(m_doc_term)
    with open(g_topic_sum_format.format(time_int_str, time_int_str), 'w+') as out_fd:
        for topic in nmf.components_:
            d_topics = dict()
            for topic_item in [(tfidf_model.get_feature_names()[i], topic[i]) \
                               for i in topic.argsort()[:-g_n_top_words-1:-1]]:
                d_topics[topic_item[0]] = topic_item[1]
            json.dump(d_topics, out_fd)
            out_fd.write('\n')
        out_fd.close()
    logging.debug('%s topic summary is written.' % time_int_str)

    d_topic_weights_by_uid = dict()
    for uid in d_txt_by_uid:
        topic_weights = nmf.transform(m_doc_term[d_txt_by_uid[uid]['mat_idx']].reshape(1, -1))
        # print_topics(nmf, tfidf_model, n_top_words)
        d_topic_weights_by_uid[uid] = dict()
        for w_idx, weight in enumerate(topic_weights[0]):
            d_topic_weights_by_uid[uid][w_idx] = weight
    with open(g_topic_weights_by_uid_format.format(time_int_str, time_int_str), 'w+') as out_fd:
        json.dump(d_topic_weights_by_uid, out_fd)
    logging.debug('%s topic weights by uid are written.' % time_int_str)


def topics_by_uid_for_time_ints(tfidf_model, l_time_ints):
    for time_int in l_time_ints:
        topics_by_uid_for_one_time_int(time_int, tfidf_model)


# we consider the texts collected for a user within a specific time interval as a document.
def get_docs():
    l_docs = []
    l_time_ints = data_preprocessing_utils.read_time_ints()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        txt_db_conn = sqlite3.connect(g_time_series_data_txt_by_uid_db_format.format(time_int_str, time_int_str))
        txt_db_cur = txt_db_conn.cursor()
        sql_str = '''SELECT raw_cln FROM wh_txt_by_uid'''
        txt_db_cur.execute(sql_str)
        l_txt_recs = txt_db_cur.fetchall()
        for txt_rec in l_txt_recs:
            l_docs.append(txt_rec[0])
        txt_db_conn.close()
        logging.debug('%s: %s docs are read.' % (time_int_str, str(len(l_txt_recs))))
    logging.debug('All docs are read. %s docs in total.' % str(len(l_docs)))
    return l_docs


def get_tfidf_vecterizer_over_all_docs():
    l_docs = get_docs()
    tfidf_vecterizer = TfidfVectorizer(max_df=0.8,
                                       min_df=1,
                                       max_features=g_n_features)
    tfidf_model = tfidf_vecterizer.fit(l_docs)
    return tfidf_model


def topics_by_uid_multithreads(tfidf_model, l_time_ints):
    # batch_size = math.ceil(len(l_time_ints) / multiprocessing.cpu_count())
    batch_size = math.ceil(len(l_time_ints) / 4)
    l_l_time_ints = []
    for i in range(0, len(l_time_ints), batch_size):
        if i + batch_size < len(l_time_ints):
            l_l_time_ints.append(l_time_ints[i:i + batch_size])
        else:
            l_l_time_ints.append(l_time_ints[i:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_time_ints:
        t = threading.Thread(target=topics_by_uid_for_time_ints, args=(tfidf_model, l_each_batch))
        t.setName('Topic_t_' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('Thread %s is finished.' % t.getName())

    logging.debug('All topic summaries and topic weights have been written.')



def main():
    tfidf_model = get_tfidf_vecterizer_over_all_docs()
    l_time_ints = data_preprocessing_utils.read_time_ints()
    # topics_by_uid_for_time_ints(tfidf_model, l_time_ints)
    topics_by_uid_multithreads(tfidf_model, l_time_ints)


if __name__ == '__main__':
    main()