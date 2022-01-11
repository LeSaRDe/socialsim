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
import data_preprocessing_utils
from scipy.spatial import distance
from scipy.special import softmax
import scipy.spatial.distance as scipyd
import matplotlib.pyplot as plt
import copy


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_raw_data_path = g_path_prefix + 'Tng_an_WH_Twitter_v3.json'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_reply_quote_db = g_time_series_data_path_prefix + 'wh_tw_reply_quote.db'
g_prop_db = g_time_series_data_path_prefix + 'wh_tw_prop_v3.db'
g_nec_data_sum_db = g_time_series_data_path_prefix + 'wh_tw_nec_data.db'
g_src_tid_prop_cnt_path = g_time_series_data_path_prefix + 'wh_tw_src_tid_prop_cnt.txt'
g_time_series_data_db_format = g_time_series_data_path_prefix + '{0}/{1}_nec_data.db'
g_src_tw_txt_path = g_time_series_data_path_prefix + 'top_src_tw_txt.txt'
g_src_tw_sim_path = g_time_series_data_path_prefix + 'top_src_tw_sim.txt'
# g_time_series_data_txt_by_uid_db_format = g_time_series_data_path_prefix + '{0}/{1}_txt_by_uid.db'
g_propagation_count = 10
g_time_int_str = '20180718_20180724'
# l_top_src_tids = ['JEz-PZCIBJ-I_sjLpn2A_w',
#                   'bQe0ontKD10INXSyo2QoeA',
#                   'ZcGAn1_tMq3jh-EHeciNbw',
#                   'UrAOYpxlWttEMM7gtrrq2w',
#                   'z1NdvBEBE1oebaewRnJEeA',
#                   'UyD7wKxfQXdxEep1kXjTGw',
#                   'dY1xxfRtS6SJ8jwi6K_sJg',
#                   'RV-6G1NsHDVaMFZOnHdO7A',
#                   'pWhpBteAk6bCBwaCONPzvw',
#                   '9eNe5X_mDa_MXVXJ2W6EyQ']
l_top_src_tids = ['CUn8Idr1st0FtgOUcJUPCw', 'd9Iww3WnxM6Buum4F7sj2g']


def combine_nec_data():
    db_conn = sqlite3.connect(g_nec_data_sum_db)
    db_cur = db_conn.cursor()
    sql_str = '''CREATE TABLE IF NOT EXISTS wh_nec_data (tid TEXT PRIMARY KEY, uid TEXT NOT NULL, ''' \
              '''time TEXT NOT NULL, type TEXT NOT NULL, src TEXT, raw TEXT, raw_cln TEXT, org TEXT, org_cln TEXT, raw_cln_vec TEXT)'''
    db_cur.execute(sql_str)
    db_conn.commit()

    l_time_ints = data_preprocessing_utils.read_time_ints()
    timer_start = time.time()
    count = 0
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        nec_db_conn = sqlite3.connect(g_time_series_data_db_format.format(time_int_str, time_int_str))
        nec_db_cur = nec_db_conn.cursor()
        sql_str = '''SELECT * FROM wh_nec_data'''
        nec_db_cur.execute(sql_str)
        l_nec_recs = nec_db_cur.fetchall()
        for nec_rec in l_nec_recs:
            sql_str = '''INSERT INTO wh_nec_data VALUES (?,?,?,?,?,?,?,?,?,?)'''
            db_cur.execute(sql_str, nec_rec)
            count += 1
            if count % 50000 and count >= 50000:
                logging.debug('%s nec recs have been written in %s seconds.' % (count, str(time.time()-timer_start)))
                db_conn.commit()
        logging.debug('%s nec recs have been written in %s seconds.' % (count, str(time.time() - timer_start)))
        db_conn.commit()
        nec_db_conn.close()
        logging.debug('%s is done.' % time_int_str)
    logging.debug('%s nec recs have been written in %s seconds.' % (count, str(time.time() - timer_start)))
    db_conn.commit()
    db_conn.close()
    logging.debug('All done.')


def get_token_distributions_for_top_src_tweets(l_top_src_tids):
    d_d_token_dists = dict()
    db_conn = sqlite3.connect(g_reply_quote_db)
    db_cur = db_conn.cursor()
    sql_str_1 = '''SELECT tid FROM wh_reply_quote WHERE src_tid=?'''
    sql_str_2 = '''SELECT cln_txt FROM wh_reply_quote WHERE tid=?'''
    for top_src_tid in l_top_src_tids:
        d_token_dist = dict()
        db_cur.execute(sql_str_1, (top_src_tid,))
        l_recs = db_cur.fetchall()
        for rec in l_recs:
            tid = rec[0]
            db_cur.execute(sql_str_2, (tid,))
            cln_txt = db_cur.fetchone()[0]
            l_cln_sents = cln_txt.split('\n')
            for cln_sent in l_cln_sents:
                l_tokens = [token.strip().lower() for token in cln_sent.split(' ')]
                for token in l_tokens:
                    if token == '':
                        continue
                    if token not in d_token_dist:
                        d_token_dist[token] = 1
                    else:
                        d_token_dist[token] += 1
        if top_src_tid in d_d_token_dists:
            raise Exception('%s already exists.' % top_src_tid)
        d_d_token_dists[top_src_tid] = d_token_dist
        logging.debug('%s token distribution is done.' % top_src_tid)
    db_conn.close()
    logging.debug('All token distributions are done.')
    return d_d_token_dists


def pairwise_js_div(d_token_dist_1, d_token_dist_2):
    if len(d_token_dist_1) == 0 or len(d_token_dist_2) == 0:
        logging.debug('At least one input distribution is empty.')
        return 1.0

    l_dist_1 = []
    l_dist_2 = []

    for idx_1, tid_1 in enumerate(d_token_dist_1):
        l_dist_1.append(d_token_dist_1[tid_1])
        if tid_1 in d_token_dist_2:
            l_dist_2.append(d_token_dist_2[tid_1])
            del d_token_dist_2[tid_1]
        else:
            l_dist_2.append(0.0)
    for idx_2, tid_2 in enumerate(d_token_dist_2):
        l_dist_2.append(d_token_dist_2[tid_2])
        l_dist_1.append(0.0)

    l_dist_1 = softmax(l_dist_1)
    l_dist_2 = softmax(l_dist_2)
    js_div = distance.jensenshannon(l_dist_1, l_dist_2)
    # if js_div == 0.0:
    #     print()
    return js_div


def js_div_mat_for_token_dists(d_d_token_dists, l_top_src_tids):
    # l_dims = list(d_d_token_dists.keys())
    # l_dims = copy.deepcopy(l_top_src_tids)
    l_dels = []
    m_js_divs = np.zeros((len(l_top_src_tids), len(l_top_src_tids)))
    for i in range(0, len(l_top_src_tids)):
        for j in range(i, len(l_top_src_tids)):
            if i == j:
                m_js_divs[i][j] = 0.0
            else:
                if len(d_d_token_dists[l_top_src_tids[i]]) == 0 or len(d_d_token_dists[l_top_src_tids[j]]) == 0:
                    logging.debug('%s or %s has no key words: %s, %s.' %
                                  (l_top_src_tids[i], l_top_src_tids[j],
                                   len(d_d_token_dists[l_top_src_tids[i]]), len(d_d_token_dists[l_top_src_tids[j]])))
                    m_js_divs[i][j] = 1.0
                    m_js_divs[j][i] = 1.0
                    if len(d_d_token_dists[l_top_src_tids[i]]) == 0:
                        l_dels.append(i)
                    if len(d_d_token_dists[l_top_src_tids[j]]) == 0:
                        l_dels.append(j)
                    continue
                js_div = pairwise_js_div(d_d_token_dists[l_top_src_tids[i]], d_d_token_dists[l_top_src_tids[j]])
                if str(js_div) == 'nan':
                    js_div = 1.0
                m_js_divs[i][j] = js_div
                m_js_divs[j][i] = js_div

    # l_ret = [l_dims.index(ele) for ele in l_dims if l_dims.index(ele) not in l_dels]
    return m_js_divs, l_dels

    # l_ticks = [i for i in range(0, len(l_dims), 10)]
    # plt.imshow(m_js_divs)
    # plt.colorbar()
    # # plt.plot(m_js_divs)
    # plt.xticks(l_ticks)
    # plt.yticks(l_ticks)
    # plt.show()


def js_div_mat_for_top_src_tws(l_top_src_tids):
    # l_dims = copy.deepcopy(l_top_src_tids)
    l_dels = []
    m_js_divs = np.zeros((len(l_top_src_tids), len(l_top_src_tids)))
    a_js_divs_avg = np.zeros(len(l_top_src_tids))
    d_word_dist = dict()
    d_top_src_tw_dists = dict()
    db_conn = sqlite3.connect(g_nec_data_sum_db)
    db_cur = db_conn.cursor()
    sql_str = '''SELECT raw_cln FROM wh_nec_data WHERE tid=?'''
    for i in range(0, len(l_top_src_tids)):
        db_cur.execute(sql_str, (l_top_src_tids[i],))
        raw_cln_rec = db_cur.fetchone()
        if raw_cln_rec is None:
            d_top_src_tw_dists[l_top_src_tids[i]] = dict()
            l_dels.append(i)
            continue
        l_raw_sents = [sent.strip() for sent in raw_cln_rec[0].split('\n')]
        l_raw_cln_words = []
        for sent in l_raw_sents:
            l_raw_cln_words += [word.strip().lower() for word in sent.split(' ')]
        d_raw_cln_dist = dict()
        for word in l_raw_cln_words:
            if word not in d_raw_cln_dist:
                d_raw_cln_dist[word] = 1
            else:
                d_raw_cln_dist[word] += 1
            if word not in d_word_dist:
                d_word_dist[word] = 1
            else:
                d_word_dist[word] += 1
        d_top_src_tw_dists[l_top_src_tids[i]] = d_raw_cln_dist
    db_conn.close()

    # for del_tid in l_dels:
    #     l_dims.remove(del_tid)

    for i in range(0, len(l_top_src_tids)):
        for j in range(i, len(l_top_src_tids)):
            if i == j:
                m_js_divs[i][j] = 0.0
            else:
                m_js_divs[i][j] = pairwise_js_div(d_top_src_tw_dists[l_top_src_tids[i]],
                                                  d_top_src_tw_dists[l_top_src_tids[j]])
                m_js_divs[j][i] = m_js_divs[i][j]

    for i in range(0, len(l_top_src_tids)):
        a_js_divs_avg[i] = pairwise_js_div(d_word_dist, d_top_src_tw_dists[l_top_src_tids[i]])

    return m_js_divs, a_js_divs_avg, l_dels


def sem_sim_mat_for_top_src_tws(l_top_src_tids):
    # l_dims = copy.deepcopy(l_top_src_tids)
    m_sem_sims = np.zeros((len(l_top_src_tids), len(l_top_src_tids)))
    l_dels = []
    db_conn = sqlite3.connect(g_nec_data_sum_db)
    db_cur = db_conn.cursor()
    sql_str = '''SELECT raw_cln_vec FROM wh_nec_data WHERE tid=?'''
    for i in range(0, len(l_top_src_tids)):
        for j in range(i, len(l_top_src_tids)):
            if i == j:
                m_sem_sims[i][j] = 1.0
            else:
                db_cur.execute(sql_str, (l_top_src_tids[i],))
                raw_cln_vec_str_1 = db_cur.fetchone()
                db_cur.execute(sql_str, (l_top_src_tids[j],))
                raw_cln_vec_str_2 = db_cur.fetchone()
                if raw_cln_vec_str_1 is None or raw_cln_vec_str_2 is None:
                    m_sem_sims[i][j] = 0.0
                    m_sem_sims[j][i] = 0.0
                    logging.debug('%s or %s does not have tweet text.' % (l_top_src_tids[i], l_top_src_tids[j]))
                    if raw_cln_vec_str_1 is None:
                        l_dels.append(i)
                    if raw_cln_vec_str_2 is None:
                        l_dels.append(j)
                    continue
                l_raw_cln_vec_1 = np.asarray([float(ele.strip()) for ele in raw_cln_vec_str_1[0].split(',')])
                l_raw_cln_vec_2 = np.asarray([float(ele.strip()) for ele in raw_cln_vec_str_2[0].split(',')])
                sim = 1 - scipyd.cosine(l_raw_cln_vec_1, l_raw_cln_vec_2)
                if str(sim) == 'nan':
                    sim = 0.0
                m_sem_sims[i][j] = sim
                m_sem_sims[j][i] = sim
    db_conn.close()

    # l_rets = [l_dims.index(ele) for ele in l_dims if l_dims.index(ele) not in l_dels]
    return m_sem_sims, l_dels

    # l_ticks = [i for i in range(0, len(l_dims), 10)]
    # plt.imshow(m_sem_sims)
    # plt.colorbar()
    # plt.xticks(l_ticks)
    # plt.yticks(l_ticks)
    # plt.show()


def get_time_int(time_str):
    trg_time_int = datetime.strptime(time_str, '%Y%m%d%H%M%S')
    # l_time_int_strs = [data_preprocessing_utils.time_int_to_time_int_str(time_int) for time_int
    #                in data_preprocessing_utils.read_time_ints()]
    l_time_ints = data_preprocessing_utils.read_time_ints()
    for time_int in l_time_ints:
        if time_int[0] <= trg_time_int <= time_int[1]:
            return data_preprocessing_utils.time_int_to_time_int_str(time_int)
    logging.debug('Cannot find any time interval for %s' % time_str)
    return None


def init_reply_quote_db():
    db_conn = sqlite3.connect(g_reply_quote_db)
    db_cur = db_conn.cursor()
    sql_str = '''CREATE TABLE IF NOT EXISTS wh_reply_quote (tid TEXT PRIMARY KEY NOT NULL, type TEXT NOT NULL,''' \
              '''src_tid TEXT, cln_txt TEXT)'''
    db_cur.execute(sql_str)
    db_conn.commit()
    db_conn.close()
    logging.debug('wh_reply_quote init is done.')

# to be obsoleted
def extract_replies_and_quotes():
    db_conn = sqlite3.connect(g_reply_quote_db)
    db_cur = db_conn.cursor()
    timer_start = time.time()
    count = 0
    with open(g_raw_data_path, 'r') as in_fd:
        t_str = in_fd.readline()
        while t_str:
            t_json = json.loads(t_str)
            tid = t_json['id_str_h']
            # if tid != 'pQroi9UrSZTwlZlYk4WD2Q':
            #     t_str = in_fd.readline()
            #     continue
            t_type = data_preprocessing_utils.get_tweet_type(t_json)
            if t_type == 'r':
                src_tid = t_json['in_reply_to_status_id_str_h']
            elif t_type == 'q':
                src_tid = t_json['quoted_status']['id_str_h']
            else:
                t_str = in_fd.readline()
                continue
            t_time = data_preprocessing_utils.get_user_time(t_json['created_at'])
            time_int_str = get_time_int(t_time)
            nec_db_conn = sqlite3.connect(g_time_series_data_db_format.format(time_int_str, time_int_str))
            nec_db_cur = nec_db_conn.cursor()
            sql_str = '''SELECT org_cln FROM wh_nec_data WHERE tid=?'''
            nec_db_cur.execute(sql_str, (tid,))
            nec_rec = nec_db_cur.fetchone()
            nec_db_conn.close()
            t_cln_txt = ''
            if nec_rec is not None:
                t_cln_txt = nec_rec[0]
            sql_str = '''INSERT INTO wh_reply_quote VALUES (?, ?, ?, ?)'''
            db_cur.execute(sql_str, (tid, t_type, src_tid, t_cln_txt))
            count += 1
            if count % 10000 and count >= 10000:
                db_conn.commit()
                logging.debug('%s records have been written to wh_reply_quote in %s seconds.'
                              % (count, str(time.time()-timer_start)))
            t_str = in_fd.readline()
        db_conn.commit()
        logging.debug('%s records have been written to wh_reply_quote in %s seconds.'
                      % (count, str(time.time() - timer_start)))
        in_fd.close()
    db_conn.close()


def extract_propagations(en_reply, en_quote, en_retweet, en_original):
    db_conn = sqlite3.connect(g_prop_db)
    db_cur = db_conn.cursor()
    timer_start = time.time()
    count = 0
    with open(g_raw_data_path, 'r') as in_fd:
        t_str = in_fd.readline()
        while t_str:
            t_json = json.loads(t_str)
            tid = t_json['id_str_h']
            t_type = data_preprocessing_utils.get_tweet_type(t_json)
            if t_type == 'r':
                src_tid = t_json['in_reply_to_status_id_str_h']
            elif t_type == 'q':
                src_tid = t_json['quoted_status']['id_str_h']
            else:
                t_str = in_fd.readline()
                continue
            t_time = data_preprocessing_utils.get_user_time(t_json['created_at'])
            time_int_str = get_time_int(t_time)
            nec_db_conn = sqlite3.connect(g_time_series_data_db_format.format(time_int_str, time_int_str))
            nec_db_cur = nec_db_conn.cursor()
            sql_str = '''SELECT org_cln FROM wh_nec_data WHERE tid=?'''
            nec_db_cur.execute(sql_str, (tid,))
            nec_rec = nec_db_cur.fetchone()
            nec_db_conn.close()
            t_cln_txt = ''
            if nec_rec is not None:
                t_cln_txt = nec_rec[0]
            sql_str = '''INSERT INTO wh_reply_quote VALUES (?, ?, ?, ?)'''
            db_cur.execute(sql_str, (tid, t_type, src_tid, t_cln_txt))
            count += 1
            if count % 10000 and count >= 10000:
                db_conn.commit()
                logging.debug('%s records have been written to wh_reply_quote in %s seconds.'
                              % (count, str(time.time()-timer_start)))
            t_str = in_fd.readline()
        db_conn.commit()
        logging.debug('%s records have been written to wh_reply_quote in %s seconds.'
                      % (count, str(time.time() - timer_start)))
        in_fd.close()
    db_conn.close()


def fetch_top_src_tids(propagation_count):
    db_conn = sqlite3.connect(g_reply_quote_db)
    db_cur = db_conn.cursor()
    sql_str = '''SELECT src_tid, count(tid) FROM wh_reply_quote GROUP BY src_tid ORDER BY count(tid) DESC'''
    db_cur.execute(sql_str)
    l_recs = db_cur.fetchall()
    d_src_tids = dict()
    for rec in l_recs:
        src_tid = rec[0]
        prop_cnt = rec[1]
        if prop_cnt >= propagation_count:
            d_src_tids[src_tid] = prop_cnt
    l_src_tids = [ele[0] for ele in sorted(d_src_tids.items(), key=lambda k: k[1], reverse=True)]
    return d_src_tids, l_src_tids


def output_src_tid_prop_cnt(d_src_tids):
    with open(g_src_tid_prop_cnt_path, 'w+') as out_fd:
        sorted_d_src_tids = sorted(d_src_tids.items(), key=lambda k: k[1], reverse=True)
        for item in sorted_d_src_tids:
            out_fd.write(item[0] + ':' + str(item[1]))
            out_fd.write('\n')
        out_fd.close()


def output_src_tw_sim_txt(l_src_tids_valid, m_src_tw_sim_valid):
    if len(l_src_tids_valid) != m_src_tw_sim_valid.shape[0]:
        logging.error('Inconsistent shape.')
        return
    db_conn = sqlite3.connect(g_nec_data_sum_db)
    db_cur = db_conn.cursor()
    sql_str = '''SELECT raw_cln FROM wh_nec_data WHERE tid=?'''
    with open(g_src_tw_txt_path, 'w+') as out_fd:
        for idx, tid in enumerate(l_src_tids_valid):
            db_cur.execute(sql_str, (tid,))
            src_txt = '|'.join(db_cur.fetchone()[0].split('\n'))
            out_str = str(idx) + ':' + tid + ':' + src_txt
            out_fd.write(out_str)
            out_fd.write('\n')
        out_fd.close()

    with open(g_src_tw_sim_path, 'w+') as out_fd:
        for i in range(0, m_src_tw_sim_valid.shape[0]-1):
            for j in range(i, m_src_tw_sim_valid.shape[0]):
                out_str = '(%s,%s):%s' % (i, j, str(m_src_tw_sim_valid[i][j]))
                out_fd.write(out_str)
                out_fd.write('\n')
        out_fd.close()


def main():
    # init_reply_quote_db()
    # extract_replies_and_quotes()

    d_src_tids, l_src_tids = fetch_top_src_tids(g_propagation_count)
    # output_src_tid_prop_cnt(d_src_tids)

    # l_src_tids = l_top_src_tids

    d_d_token_dists = get_token_distributions_for_top_src_tweets(l_src_tids)
    m_reply_quote_js, l_reply_quote_dels = js_div_mat_for_token_dists(d_d_token_dists, l_src_tids)
    m_src_tw_sim, l_src_tw_sim_dels = sem_sim_mat_for_top_src_tws(l_src_tids)
    m_src_tw_js, a_src_tw_avg_js, l_src_tw_js_dels = js_div_mat_for_top_src_tws(l_src_tids)

    l_inter_dims = [l_src_tids.index(ele) for ele in l_src_tids if l_src_tids.index(ele) not in l_reply_quote_dels
                    and l_src_tids.index(ele) not in l_src_tw_js_dels
                    and l_src_tids.index(ele) not in l_src_tw_sim_dels]

    m_reply_quote_js_valid = np.zeros((len(l_inter_dims), len(l_inter_dims)))
    m_src_tw_sim_valid = np.zeros((len(l_inter_dims), len(l_inter_dims)))
    m_src_tw_js_valid = np.zeros((len(l_inter_dims), len(l_inter_dims)))
    a_src_tw_avg_js_valid = np.zeros(len(l_inter_dims))

    for i in range(0, len(l_inter_dims)):
        for j in range(i, len(l_inter_dims)):
            if i == j:
                if m_reply_quote_js[l_inter_dims[i]][l_inter_dims[j]] != 0.0:
                    logging.debug('m_reply_quote_js of (%s, %s) is incorrect.' % (l_inter_dims[i], l_inter_dims[j]))
                if m_src_tw_sim[l_inter_dims[i]][l_inter_dims[j]] != 1.0:
                    logging.debug('m_src_tw_sim of (%s, %s) is incorrect.' % (l_inter_dims[i], l_inter_dims[j]))
                if m_src_tw_js[l_inter_dims[i]][l_inter_dims[j]] != 0.0:
                    logging.debug('m_src_tw_js of (%s, %s) is incorrect.' % (l_inter_dims[i], l_inter_dims[j]))
            m_reply_quote_js_valid[i][j] = m_reply_quote_js[l_inter_dims[i]][l_inter_dims[j]]
            m_reply_quote_js_valid[j][i] = m_reply_quote_js[l_inter_dims[j]][l_inter_dims[i]]
            m_src_tw_sim_valid[i][j] = m_src_tw_sim[l_inter_dims[i]][l_inter_dims[j]]
            m_src_tw_sim_valid[j][i] = m_src_tw_sim[l_inter_dims[j]][l_inter_dims[i]]
            m_src_tw_js_valid[i][j] = m_src_tw_js[l_inter_dims[i]][l_inter_dims[j]]
            m_src_tw_js_valid[j][i] = m_src_tw_js[l_inter_dims[j]][l_inter_dims[i]]
        a_src_tw_avg_js_valid[i] = a_src_tw_avg_js[l_inter_dims[i]]

    # l_ticks = [i for i in range(0, len(l_inter_dims), 30)]
    #
    # plt.imshow(m_reply_quote_js_valid)
    # plt.colorbar()
    # # plt.xticks(l_ticks)
    # # plt.yticks(l_ticks)
    # plt.show()
    #
    # plt.imshow(m_src_tw_sim_valid)
    # plt.colorbar()
    # # plt.xticks(l_ticks)
    # # plt.yticks(l_ticks)
    # plt.show()
    #
    # plt.imshow(m_src_tw_js_valid)
    # plt.colorbar()
    # # plt.xticks(l_ticks)
    # # plt.yticks(l_ticks)
    # plt.show()
    #
    # plt.plot(a_src_tw_avg_js_valid)
    # plt.show()

    l_src_tids_valid = [ele for ele in l_src_tids if l_src_tids.index(ele) in l_inter_dims]
    output_src_tw_sim_txt(l_src_tids_valid, m_src_tw_sim_valid)




    # l_inter_dims = [ele for ele in l_js_div_dims if ele in l_sem_sim_dim]
    # m_js_div_valid = np.zeros((len(l_inter_dims), len(l_inter_dims)))
    # m_sem_sim_valid = np.zeros((len(l_inter_dims), len(l_inter_dims)))
    # for i in range(0, len(l_inter_dims)):
    #     for j in range(i, len(l_inter_dims)):
    #         if i == j:
    #             if m_js_div[l_inter_dims[i]][l_inter_dims[j]] != 0.0:
    #                 logging.debug('js of (%s, %s) is incorrect.' % (l_inter_dims[i], l_inter_dims[j]))
    #             if m_sem_sim[l_inter_dims[i]][l_inter_dims[j]] != 1.0:
    #                 logging.debug('cos of (%s, %s) is incorrect.' % (l_inter_dims[i], l_inter_dims[j]))
    #         m_js_div_valid[i][j] = m_js_div[l_inter_dims[i]][l_inter_dims[j]]
    #         m_js_div_valid[j][i] = m_js_div[l_inter_dims[j]][l_inter_dims[i]]
    #         m_sem_sim_valid[i][j] = m_sem_sim[l_inter_dims[i]][l_inter_dims[j]]
    #         m_sem_sim_valid[j][i] = m_sem_sim[l_inter_dims[j]][l_inter_dims[i]]
    # l_ticks = [i for i in range(0, len(l_inter_dims), 30)]
    # plt.imshow(m_js_div_valid)
    # plt.colorbar()
    # plt.xticks(l_ticks)
    # plt.yticks(l_ticks)
    # plt.show()
    # plt.imshow(m_sem_sim_valid)
    # plt.colorbar()
    # plt.xticks(l_ticks)
    # plt.yticks(l_ticks)


    print()


    # combine_nec_data()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()