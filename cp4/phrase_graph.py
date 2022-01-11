import logging
import time
import math
import threading
import multiprocessing
import sys
from os import walk, path
import os
import random
import traceback

import sqlite3
import json
import networkx as nx
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.special import softmax
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import normalize
import psycopg2

import global_settings
sys.path.insert(1, global_settings.g_lexvec_model_folder)
import model as lexvec


'''
Raw Phrase Extraction
'''
# def make_phrase_extraction_jobs(num_jobs):
#     su_db_conn = sqlite3.connect(global_settings.g_tw_sem_units_db)
#     su_db_cur = su_db_conn.cursor()
#     su_sql_str = '''select tw_id from ven_tw_sem_units'''
#     su_db_cur.execute(su_sql_str)
#     l_tw_ids = [rec[0].strip() for rec in su_db_cur.fetchall()]
#     batch_size = math.ceil(len(l_tw_ids) / num_jobs)
#     l_jobs = []
#     for i in range(0, len(l_tw_ids), batch_size):
#         l_jobs.append(l_tw_ids[i:i + batch_size])
#     for idx, job in enumerate(l_jobs):
#         with open(global_settings.g_tw_phrase_job_file_format.format(idx), 'w+') as out_fd:
#             out_str = '\n'.join(job)
#             out_fd.write(out_str)
#             out_fd.close()
#     print('[make_phrase_extraction_jobs] %s raw phrase jobs.' % len(l_jobs))


def extract_phrases_from_cls_json_str(cls_json_str):
    if cls_json_str is None:
        return None
    s_covered_nodes = []
    s_phrases = set([])
    try:
        cls_json = json.loads(cls_json_str)
        cls_graph = nx.adjacency_graph(cls_json)
        for edge in cls_graph.edges(data=True):
            node_1 = edge[0]
            node_2 = edge[1]
            node_1_txt = cls_graph.nodes(data=True)[node_1]['txt']
            node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
            node_2_txt = cls_graph.nodes(data=True)[node_2]['txt']
            node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
            phrase = (node_1_txt, node_2_txt, node_1_pos + '_' + node_2_pos)
            s_covered_nodes.append(node_1)
            s_covered_nodes.append(node_2)
            s_phrases.add(phrase)
        s_covered_nodes = set(s_covered_nodes)
        if len(s_covered_nodes) < len(cls_graph.nodes):
            for node in cls_graph.nodes(data=True):
                if node[0] not in s_covered_nodes:
                    node_txt = node[1]['txt']
                    node_pos = node[1]['pos']
                    s_phrases.add((node_txt, node_pos))
    except Exception as err:
        print('[extract_phrases_from_cls_json_str] %s' % err)
        traceback.print_exc()
    if len(s_phrases) > 0:
        return list(s_phrases)
    return None


def extract_phrase_from_nps_str(nps_str):
    if nps_str is None:
        return None
    l_nps = [noun_phrase.strip() for noun_phrase in nps_str.split('\n')]
    s_phrase = set(l_nps)
    if len(s_phrase) > 0:
        s_phrase = set([(noun_phrase, 'NOUN') for noun_phrase in s_phrase])
        return list(s_phrase)
    return None


# def extract_phrases_from_sem_units_single_thread(l_tasks, t_id):
#     """
#     A phrase is a list of tokens with a POS tag at the end. Return a set of phrases.
#     :param
#         l_tasks: A list of tw_ids
#     """
#     logging.debug('[extract_phrases_from_sem_units_single_thread] Thread_%s: starts...' % t_id)
#     timer_start = time.time()
#     su_db_conn = sqlite3.connect(global_settings.g_tw_sem_units_db)
#     su_db_cur = su_db_conn.cursor()
#     su_sql_str = '''select cls_json_str, nps_str from ven_tw_sem_units where tw_id = ?'''
#     s_phrases = set([])
#     iter_cnt = 0
#     for tw_id in l_tasks:
#         su_db_cur.execute(su_sql_str, (tw_id,))
#         rec = su_db_cur.fetchone()
#         if rec is None:
#             break
#         cls_json_str = rec[0]
#         nps_str = rec[1]
#         s_phrases_cls = extract_phrases_from_cls_json_str(cls_json_str)
#         s_phrases_nps = extract_phrase_from_nps_str(nps_str)
#         if s_phrases_cls is not None:
#             s_phrases = s_phrases.union(s_phrases_cls)
#         if s_phrases_nps is not None:
#             s_phrases = s_phrases.union(s_phrases_nps)
#         iter_cnt += 1
#         if iter_cnt % 5000 == 0 and iter_cnt >= 5000:
#             logging.debug('[extract_phrases_from_sem_units] Thread_%s: %s iterations, %s phrase, in %s secs.' %
#                           (t_id, iter_cnt, len(s_phrases), time.time() - timer_start))
#     logging.debug('[extract_phrases_from_sem_units] Thread_%s: Extraction done: %s iterations, %s phrase, in %s secs.' %
#                   (t_id, iter_cnt, len(s_phrases), time.time() - timer_start))
#     su_db_conn.close()
#
#     with open(global_settings.g_tw_raw_phrases_file_format.format(t_id), 'w+') as out_fd:
#         for phrase in s_phrases:
#             out_str = ','.join(phrase[:-1])
#             out_str += '|'
#             out_str += phrase[-1]
#             out_fd.write(out_str)
#             out_fd.write('\n')
#         out_fd.close()
#     logging.debug('[extract_phrases_from_sem_units] Thread_%s: All done in %s secs!'
#                   % (t_id, time.time() - timer_start))
#     # return s_phrases
#
#
# def extract_phrases_from_sem_units_multithreading(l_tasks, op_func, num_threads, job_id):
#     # su_db_conn = sqlite3.connect(global_settings.g_tw_sem_units_db)
#     # su_db_cur = su_db_conn.cursor()
#
#     timer_1 = time.time()
#     logging.debug('[extract_phrases_from_sem_units_multithreading] %s tasks in total.' % len(l_tasks))
#     batch_size = math.ceil(len(l_tasks) / num_threads)
#     l_l_subtasks = []
#     for i in range(0, len(l_tasks), batch_size):
#         if i + batch_size < len(l_tasks):
#             l_l_subtasks.append(l_tasks[i:i + batch_size])
#         else:
#             l_l_subtasks.append(l_tasks[i:])
#     logging.debug('[extract_phrases_from_sem_units_multithreading] %s threads.' % len(l_l_subtasks))
#
#     l_threads = []
#     t_id = 0
#     for l_each_batch in l_l_subtasks:
#         t = threading.Thread(target=op_func, args=(l_each_batch, str(job_id) + '_' + str(t_id)))
#         t.setName('t_mul_task_' + str(t_id))
#         t.start()
#         l_threads.append(t)
#         t_id += 1
#
#     while len(l_threads) > 0:
#         for t in l_threads:
#             if t.is_alive():
#                 t.join(1)
#             else:
#                 l_threads.remove(t)
#                 logging.debug('[extract_phrases_from_sem_units_multithreading] Thread %s is finished.' % t.getName())
#
#     # su_db_conn.close()
#     logging.debug('[extract_phrases_from_sem_units_multithreading] All done in %s sec for %s tasks.'
#                   % (time.time() - timer_1, len(l_tasks)))
#
#
# def extract_phrases_from_sem_units():
#     if len(sys.argv) < 3:
#         raise Exception('[extract_phrases_from_sem_units] Invalid parameters!')
#     task_file_path = sys.argv[1]
#     job_id = sys.argv[2]
#     l_tasks = []
#     with open(task_file_path, 'r') as in_fd:
#         for ln in in_fd:
#             l_tasks.append(ln.strip())
#         in_fd.close()
#     extract_phrases_from_sem_units_multithreading(l_tasks,
#                                                   extract_phrases_from_sem_units_single_thread,
#                                                   multiprocessing.cpu_count(), job_id)


def extract_all_phrases_and_phrases_for_each_tw_single_thread(l_tasks, t_id):
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw_single_thread] Thread %s: Starts with %s tasks...'
                  % (t_id, len(l_tasks)))
    timer_start = time.time()

    l_ready_recs = []
    cnt = 0
    for rec in l_tasks:
        l_phrases = []
        tw_id = rec[0]
        cls_json_str = rec[1]
        l_nps = rec[2]

        if l_nps is not None:
            l_nps = [nph.strip().lower() for nph in l_nps]
            l_phrases += l_nps
        l_cls_phrases = extract_phrases_from_cls_json_str(cls_json_str)
        if l_cls_phrases is not None:
            l_cls_phrases = [' '.join(clsp_tup[:-1]).lower() for clsp_tup in l_cls_phrases]
            l_phrases += l_cls_phrases
        l_phrases = list(set(l_phrases))
        ready_rec = (tw_id, l_phrases)
        l_ready_recs.append(ready_rec)
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[extract_all_phrases_and_phrases_for_each_tw_single_thread] Thread %s: %s tws done in %s secs.'
                          % (t_id, cnt, time.time() - timer_start))
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw_single_thread] Thread %s: All %s tws done in %s secs.'
                  % (t_id, cnt, time.time() - timer_start))

    out_df = pd.DataFrame(l_ready_recs, columns=['tw_id', 'raw_phs'])
    out_df.to_pickle(global_settings.g_tw_phrase_extraction_tw_to_phrases_int_file_format.format(str(t_id)))
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw_single_thread] Thread %s: All done in %s secs.'
                  % (t_id, time.time() - timer_start))


def extract_all_phrases_and_phrases_for_each_tw_multithread(tb_name, num_threads, job_id):
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = '''select tw_id, cls_json_str, nps from {0}'''.format(tb_name)
    tw_db_cur.execute(tw_db_sql_str)
    l_tasks = tw_db_cur.fetchall()
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw] Load in %s sem units recs in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))

    batch_size = math.ceil(len(l_tasks) / int(num_threads))
    l_l_subtasks = []
    for i in range(0, len(l_tasks), batch_size):
        if i + batch_size < len(l_tasks):
            l_l_subtasks.append(l_tasks[i:i + batch_size])
        else:
            l_l_subtasks.append(l_tasks[i:])
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw] %s threads.' % len(l_l_subtasks))

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_subtasks:
        t = threading.Thread(target=extract_all_phrases_and_phrases_for_each_tw_single_thread,
                             args=(l_each_batch, str(job_id) + '_' + str(t_id)))
        t.setName('t_mul_task_' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('[extract_all_phrases_and_phrases_for_each_tw] Thread %s is finished.' % t.getName())

    logging.debug('[extract_all_phrases_and_phrases_for_each_tw] All done in %s sec for %s tasks.'
                  % (time.time() - timer_start, len(l_tasks)))


def extract_all_phrases_and_phrases_for_each_tw_int_to_out():
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw_int_to_out] Starts...')
    timer_start = time.time()

    l_dfs = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_phrase_extraction_int_folder):
        for filename in filenames:
            if filename[:18] != 'tw_to_phrases_int_' or filename[-7:] != '.pickle':
                continue
            tw_to_phs_df = pd.read_pickle(dirpath + '/' + filename)
            l_dfs.append(tw_to_phs_df)
    out_tw_to_phs_df = pd.concat(l_dfs)
    out_tw_to_phs_df.to_pickle(global_settings.g_tw_phrase_extraction_tw_to_phrases_file)
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw_int_to_out] Output g_tw_phrase_extraction_tw_to_phrases_file with %s tws in %s secs.'
                  % (len(out_tw_to_phs_df), time.time() - timer_start))

    l_phrases = []
    for tw_to_phs_rec in out_tw_to_phs_df.values:
        l_tw_phs = tw_to_phs_rec[1]
        l_phrases += l_tw_phs
    l_phrases = list(set(l_phrases))
    d_ph_to_id = {ph: idx for idx, ph in enumerate(l_phrases)}
    d_id_to_ph = {d_ph_to_id[ph]: ph for ph in d_ph_to_id}
    with open(global_settings.g_tw_raw_phrases_phrase_to_id, 'w+') as out_fd:
        json.dump(d_ph_to_id, out_fd)
        out_fd.close()
    with open(global_settings.g_tw_raw_phrases_id_to_phrase, 'w+') as out_fd:
        json.dump(d_id_to_ph, out_fd)
        out_fd.close()
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw_int_to_out] Output g_tw_raw_phrases_phrase_to_id and g_tw_raw_phrases_id_to_phrase in %s secs'
                  % str(time.time() - timer_start))

    for tw_to_phs_rec in out_tw_to_phs_df.values:
        tw_to_phs_rec[1] = [d_ph_to_id[ph] for ph in tw_to_phs_rec[1]]
    out_tw_to_phs_df.to_pickle(global_settings.g_tw_phrase_extraction_tw_to_phids_file)
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw_int_to_out] Output g_tw_phrase_extraction_tw_to_phids_file with %s tws in %s secs.'
                  % (len(out_tw_to_phs_df), time.time() - timer_start))
    logging.debug('[extract_all_phrases_and_phrases_for_each_tw_int_to_out] All done.')


def extract_src_trg_tw_data(l_usr_ids, l_tw_types, dt_start, dt_end, com_id):
    logging.debug('[extract_srg_trg_tw_pairs] Starts with %s usrs, l_tw_types=%s, dt_start=%s, dt_end=%s...'
                  % (len(l_usr_ids) if l_usr_ids is not None else None, str(l_tw_types), dt_start, dt_end))
    timer_start = time.time()

    if l_usr_ids is not None:
        s_usr_ids = set(l_usr_ids)
    else:
        s_usr_ids = None

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = '''select tw_id, usr_id, tw_type, tw_src_id, tw_datetime from cp4.mf3jh_ven_tw_en_all'''
    tw_db_cur.execute(tw_db_sql_str)
    l_recs = tw_db_cur.fetchall()
    logging.debug('[extract_src_trg_tw_data] Fetch %s tw data in %s secs.' % (len(l_recs), time.time() - timer_start))

    l_src_trg_recs = []
    cnt = 0
    for rec in l_recs:
        tw_id = rec[0]
        usr_id = rec[1]
        tw_type = rec[2]
        tw_src_id = rec[3]
        tw_datetime = rec[4]
        if s_usr_ids is not None and usr_id not in s_usr_ids:
            continue
        if tw_type not in l_tw_types:
            continue
        if tw_datetime < dt_start or tw_datetime > dt_end:
            continue
        l_src_trg_recs.append((tw_id, tw_src_id, usr_id, tw_type, tw_datetime))
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[extract_src_trg_tw_data] Extract %s src trg recs in %s secs.'
                          % (cnt, time.time() - timer_start))
    logging.debug('[extract_src_trg_tw_data] Extract %s src trg recs in %s secs.'
                  % (cnt, time.time() - timer_start))
    df_src_trg = pd.DataFrame(l_src_trg_recs, columns=['tw_id', 'tw_src_id', 'usr_id', 'tw_type', 'tw_datetime'])
    out_str = str(com_id) + '_' + '#'.join(l_tw_types) + '_' + dt_start + '#' + dt_end
    df_src_trg.to_pickle(global_settings.g_tw_src_trg_data_file_format.format(out_str))
    logging.debug('[extract_src_trg_tw_data] All done in %s secs.' % str(time.time() - timer_start))


# def load_sem_units():
#     logging.debug('[load_sem_units] Starts...')
#     timer_start = time.time()
#     su_db_conn = sqlite3.connect(global_settings.g_tw_sem_units_db)
#     su_db_cur = su_db_conn.cursor()
#     su_sql_str = '''select tw_id, cls_json_str, nps_str from ven_tw_sem_units'''
#     su_db_cur.execute(su_sql_str)
#     l_recs = su_db_cur.fetchall()
#     d_sem_units = dict()
#     for rec in l_recs:
#         tw_id = rec[0]
#         cls_json_str = rec[1]
#         nps_str = rec[2]
#         d_sem_units[tw_id] = [cls_json_str, nps_str]
#     logging.debug('[load_sem_units] d_sem_units is done in %s secs.' % str(time.time() - timer_start))
#     # TEST ONLY STARTS
#     # size_d_sem_units = sys.getsizeof(d_sem_units)
#     # for tw_id in d_sem_units:
#     #     size_d_sem_units += sys.getsizeof(tw_id)
#     #     size_d_sem_units += sys.getsizeof(d_sem_units[tw_id])
#     #     size_d_sem_units += sys.getsizeof(d_sem_units[tw_id][0])
#     #     size_d_sem_units += sys.getsizeof(d_sem_units[tw_id][1])
#     # logging.debug('[load_sem_units] Size of d_sem_units = %s.' % size_d_sem_units)
#     # TEST ONLY ENDS
#     return d_sem_units


# def extract_src_trg_tw_pairs_for_replies_quotes(offset, batch_size, d_sem_units):
#     '''
#     For originals, we temporarily skip them as the amount of originals is significantly larger than the sum of replies
#     and quotes.
#     For replies and quotes, src and trg are regular.
#     For retweets, we temporarily skip them as most tweet objects are retweets.
#     Pairs are stored in a dict: {src: [(trg#1, trg#1_datetime), ...]}
#     '''
#     timer_start = time.time()
#     tw_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
#     tw_db_cur = tw_db_conn.cursor()
#     tw_sql_str = '''select tw_id, tw_src_id, tw_type, tw_datetime from ven_tw_en limit {0}, {1}'''.format(offset, batch_size)
#     tw_db_cur.execute(tw_sql_str)
#     l_recs = tw_db_cur.fetchall()
#     logging.debug('[extract_src_trg_tw_pairs] Fetch done in %s secs' % str(time.time() - timer_start))
#
#     d_src_trg_pairs = dict()
#     cnt = 0
#     for rec in l_recs:
#         tw_id = rec[0]
#         tw_src_id = rec[1]
#         tw_type = rec[2]
#         tw_datetime = rec[3]
#         if tw_type == 'n':
#             # if tw_id not in d_sem_units:
#             #     continue
#             # if tw_id in d_src_trg_pairs:
#             #     d_src_trg_pairs[tw_id].add((tw_id, tw_datetime))
#             # else:
#             #     d_src_trg_pairs[tw_id] = {(tw_id, tw_datetime)}
#             continue
#         elif tw_type == 'r' or tw_type == 'q':
#             if tw_src_id not in d_sem_units or tw_id not in d_sem_units:
#                 continue
#             if tw_src_id in d_src_trg_pairs:
#                 d_src_trg_pairs[tw_src_id].add((tw_id, tw_datetime))
#             else:
#                 d_src_trg_pairs[tw_src_id] = {(tw_id, tw_datetime)}
#         elif tw_type == 't':
#             # if tw_src_id in d_sem_units:
#             #     if tw_src_id in d_src_trg_pairs:
#             #         d_src_trg_pairs[tw_src_id].add((tw_src_id, tw_datetime))
#             #     else:
#             #         d_src_trg_pairs[tw_src_id] = {(tw_src_id, tw_datetime)}
#             continue
#         else:
#             logging.error('[extract_src_trg_tw_pairs] Unknown tw_type = %s' % tw_type)
#         cnt += 1
#         if cnt % 10000 == 0 and cnt >= 10000:
#             logging.debug('[extract_src_trg_tw_pairs] %s tws processed in %s secs.' % (cnt, time.time() - timer_start))
#     logging.debug('[extract_src_trg_tw_pairs] All %s tws processed in %s secs.' % (cnt, time.time() - timer_start))
#
#     for tw_id in d_src_trg_pairs:
#         d_src_trg_pairs[tw_id] = list(d_src_trg_pairs[tw_id])
#     with open(global_settings.g_tw_src_trg_tw_id_pairs, 'w+') as out_fd:
#         json.dump(d_src_trg_pairs, out_fd)
#         out_fd.close()
#     logging.debug('[extract_src_trg_tw_pairs] Output %s srcs in %s secs.'
#                   % (len(d_src_trg_pairs), str(time.time() - timer_start)))


# def build_trg_src_tw_pairs():
#     logging.debug('[build_trg_src_tw_pairs] Starts...')
#     with open(global_settings.g_tw_src_trg_tw_id_pairs, 'r') as in_fd:
#         d_src_trg_pairs = json.load(in_fd)
#         in_fd.close()
#     logging.debug('[build_trg_src_tw_pairs] %s srcs in total.' % len(d_src_trg_pairs))
#     d_trg_src_pairs = dict()
#     for src_tw_id in d_src_trg_pairs:
#         for trg_tw_id, tw_datetime in d_src_trg_pairs[src_tw_id]:
#             if trg_tw_id not in d_trg_src_pairs:
#                 d_trg_src_pairs[trg_tw_id] = (src_tw_id, tw_datetime)
#             else:
#                 raise Exception('[build_trg_src_tw_pairs]trg: %s has multiple srcs: %s.' % (trg_tw_id, d_trg_src_pairs[trg_tw_id][0]))
#     with open(global_settings.g_tw_trg_src_tw_id_pairs, 'w+') as out_fd:
#         json.dump(d_trg_src_pairs, out_fd)
#         out_fd.close()
#     logging.debug('[build_trg_src_tw_pairs] All done. %s trgs in total.' % len(d_trg_src_pairs))


def replies_quotes_per_user():
    logging.debug('[replies_quotes_per_user] Starts...')
    timer_start = time.time()
    tw_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
    tw_db_cur = tw_db_conn.cursor()
    tw_sql_str = '''select tw_id, usr_id, tw_type from ven_tw_en'''
    tw_db_cur.execute(tw_sql_str)
    l_recs = tw_db_cur.fetchall()
    d_rq_per_usr = dict()
    cnt = 0
    for rec in l_recs:
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[replies_quotes_per_user] %s tws processed in %s secs.' % (cnt, time.time() - timer_start))
        tw_id = rec[0]
        usr_id = rec[1]
        tw_type = rec[2]
        if tw_type != 'r' and tw_type != 'q':
            continue
        if usr_id not in d_rq_per_usr:
            d_rq_per_usr[usr_id] = [tw_id]
        else:
            d_rq_per_usr[usr_id].append(tw_id)
    logging.debug('[replies_quotes_per_user] All tws processed in %s secs. %s users.'
                  % (time.time() - timer_start, len(d_rq_per_usr)))

    with open(global_settings.g_tw_replies_quotes_per_user, 'w+') as out_fd:
        json.dump(d_rq_per_usr, out_fd)
        out_fd.close()
    logging.debug('[replies_quotes_per_user] All done in %s secs.' % str(time.time() - timer_start))


def extract_tw_ids_from_srg_trg_pairs_for_replies_quotes():
    logging.debug('[extract_tw_ids_from_srg_trg_pairs_for_replies_quotes] Starts...')
    timer_start = time.time()
    with open(global_settings.g_tw_src_trg_tw_id_pairs, 'r') as in_fd:
        d_src_trg_pairs = json.load(in_fd)
        in_fd.close()
    l_tw_ids = list(d_src_trg_pairs.keys())
    cnt = 0
    for tw_id in d_src_trg_pairs:
        l_trg_tw_ids = [item[0] for item in d_src_trg_pairs[tw_id]]
        l_tw_ids += l_trg_tw_ids
        cnt += 1
        if cnt % 5000 == 0 and cnt >= 5000:
            logging.debug('[extract_tw_ids_from_srg_trg_pairs_for_replies_quotes] % srcs processed in %s sec.'
                          % (cnt, time.time() - timer_start))
    s_tw_ids = set(l_tw_ids)
    logging.debug('[extract_tw_ids_from_srg_trg_pairs] All done. %s tw_ids.' % len(s_tw_ids))
    return s_tw_ids


def extract_phrases_for_each_tw(offset, batch_size, s_tw_ids, d_phrase_idx):
    logging.debug('[extract_phrases_for_each_tw] Starts...')
    timer_start = time.time()
    d_tw_phrase_idx = dict()
    su_db_conn = sqlite3.connect(global_settings.g_tw_sem_units_db)
    su_db_cur = su_db_conn.cursor()
    su_sql_str = """select tw_id, cls_json_str, nps_str from ven_tw_sem_units limit {0}, {1}""".format(offset, batch_size)
    su_db_cur.execute(su_sql_str)
    l_recs = su_db_cur.fetchmany(5000)
    miss_id_cnt = 0
    cnt = 0
    while l_recs:
        for rec in l_recs:
            tw_id = rec[0]
            cls_json_str = rec[1]
            nps_str = rec[2]
            if tw_id not in s_tw_ids:
                continue
            l_all_phrases = []
            s_cls_phrases = extract_phrases_from_cls_json_str(cls_json_str)
            if s_cls_phrases is not None:
                l_all_phrases += list(s_cls_phrases)
            s_nps_phrases = extract_phrase_from_nps_str(nps_str)
            if s_nps_phrases is not None:
                l_all_phrases += list(s_nps_phrases)
            s_all_phrases = set(l_all_phrases)
            for phrase in s_all_phrases:
                phrase = ' '.join(phrase[:-1])
                phrase = phrase.strip().lower()
                try:
                    phrase_idx = d_phrase_idx[phrase]
                    if tw_id not in d_tw_phrase_idx:
                        d_tw_phrase_idx[tw_id] = [phrase_idx]
                    else:
                        d_tw_phrase_idx[tw_id].append(phrase_idx)
                except:
                    logging.debug('[extract_phrases_for_each_tw] %s is not in d_phrase_idx.' % phrase)
                    miss_id_cnt += 1
        cnt += 1
        logging.debug('[extract_phrases_for_each_tw] %s tws processed in %s secs.'
                      % (cnt*5000, time.time() - timer_start))
        l_recs = su_db_cur.fetchmany(5000)
    for tw_id in d_tw_phrase_idx:
        d_tw_phrase_idx[tw_id] = list(set(d_tw_phrase_idx[tw_id]))
    logging.debug('[extract_phrases_for_each_tw] All tws processed in %s secs. Output phrase for %s tws.'
                  % (time.time() - timer_start, len(d_tw_phrase_idx)))
    with open(global_settings.g_tw_phrases_by_idx, 'w+') as out_fd:
        json.dump(d_tw_phrase_idx, out_fd)
        out_fd.close()
    logging.debug('[extract_phrases_for_each_tw] All done.')


def read_phrases_into_one_file():
    """
    This function is only for multiprocessing results, as in that case there will many output files.
    """
    l_raw_phrases = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_phrase_graph_folder):
        for filename in filenames:
            if filename[:15] != 'tw_raw_phrases_':
                continue
            with open(dirpath + '/' + filename, 'r') as in_fd:
                for ln in in_fd:
                    l_raw_phrases.append(ln.strip())
                in_fd.close()
    with open(global_settings.g_tw_raw_phrases_output, 'w+') as out_fd:
        out_str = '\n'.join(l_raw_phrases)
        out_fd.write(out_str)
        out_fd.close()
    logging.debug('[read_phrases_into_one_file] All done.')


def load_pos_mapping():
    d_pos_code = dict()
    with open(global_settings.g_tw_raw_phrases_pos_mapping, 'r') as in_fd:
        for ln in in_fd:
            fields = [item.strip() for item in ln.split(':')]
            pos_str = fields[0]
            pos_code = int(fields[1])
            d_pos_code[pos_str] = pos_code
            pos_fields = [item.strip() for item in pos_str.split('_')]
            if len(pos_fields) > 1:
                rev_pos_str = pos_fields[1] + '_' + pos_fields[0]
                d_pos_code[rev_pos_str] = pos_code
        in_fd.close()
    logging.debug('[load_pos_mapping] All done.')
    return d_pos_code


def extract_unique_phrases():
    logging.debug('[extract_unique_phrases] Starts...')
    d_phrase_with_pos = dict()
    with open(global_settings.g_tw_raw_phrases_output, 'r') as in_fd:
        for ln in in_fd:
            fields = ln.split('|')
            phrase = ' '.join([token.strip().lower() for token in fields[0].split(',')])
            pos = fields[1].strip()
            if phrase not in d_phrase_with_pos:
                d_phrase_with_pos[phrase] = [pos]
            else:
                d_phrase_with_pos[phrase].append(pos)
        in_fd.close()

    d_pos_code = load_pos_mapping()
    for phrase in d_phrase_with_pos:
        s_pos_strs = set(d_phrase_with_pos[phrase])
        l_pos_code = []
        for pos_str in s_pos_strs:
            try:
                pos_code = d_pos_code[pos_str]
                l_pos_code.append(pos_code)
            except Exception as err:
                logging.error('[extract_unique_phrases] %s' % err)
        d_phrase_with_pos[phrase] = list(set(l_pos_code))

    logging.debug('[extract_unique_phrases] %s unique phrases.' % len(d_phrase_with_pos))
    with open(global_settings.g_tw_raw_phrases_unique, 'w+') as out_fd:
        json.dump(d_phrase_with_pos, out_fd)
        out_fd.close()
    logging.debug('[extract_unique_phrases] All done.')


# def id_phrases():
#     logging.debug('[id_phrases] Starts...')
#     with open(global_settings.g_tw_raw_phrases_unique, 'r') as in_fd:
#         d_phrases = json.load(in_fd)
#         in_fd.close()
#     d_phrase_id = dict()
#     d_id_phrase = dict()
#     id = 0
#     for phrase in d_phrases:
#         d_phrase_id[phrase] = id
#         d_id_phrase[id] = phrase
#         id += 1
#     with open(global_settings.g_tw_raw_phrases_phrase_to_id, 'w+') as out_fd:
#         json.dump(d_phrase_id, out_fd)
#         out_fd.close()
#     logging.debug('[id_phrases] Output g_tw_raw_phrases_phrase_to_id.')
#     with open(global_settings.g_tw_raw_phrases_id_to_phrase, 'w+') as out_fd:
#         json.dump(d_id_phrase, out_fd)
#         out_fd.close()
#     logging.debug('[id_phrases] Output g_tw_raw_phrases_id_to_phrase.')


def phrases_type_encode():
    d_phrases_pos = dict()

    with open(global_settings.g_tw_raw_phrases_output, 'r') as in_fd:
        for ln in in_fd:
            fields = [item.strip() for item in ln.split('|')]
            phrase = fields[0]
            pos_pair = fields[1]
            l_pos = [pos.strip() for pos in pos_pair.split('_')]
            if len(l_pos) == 1:
                rev_pos_pair = pos_pair
            else:
                rev_pos_pair = l_pos[1] + '_' + l_pos[0]
            if pos_pair not in d_phrases_pos:
                if rev_pos_pair not in d_phrases_pos:
                    d_phrases_pos[pos_pair] = [phrase]
                else:
                    d_phrases_pos[rev_pos_pair].append(phrase)
            else:
                d_phrases_pos[pos_pair].append(phrase)
        in_fd.close()
    logging.debug('[phrases_type_encode] d_phrases_pos and s_all_pos are done.')

    with open(global_settings.g_tw_raw_phrases_pos_stat, 'w+') as out_fd:
        sorted_pos_pairs_by_len = sorted(d_phrases_pos.keys(), key=lambda k: len(d_phrases_pos[k]), reverse=True)
        for pos_pair in sorted_pos_pairs_by_len:
            out_fd.write(pos_pair + ':' + str(len(d_phrases_pos[pos_pair])))
            out_fd.write('\n')
        out_fd.close()
    logging.debug('[phrases_type_encode] pos stat is done.')

    with open(global_settings.g_tw_raw_phrases_pos_mapping, 'w+') as out_fd:
        sorted_pos_pairs = sorted(d_phrases_pos.keys())
        for idx, pos_pair in enumerate(sorted_pos_pairs):
            out_fd.write(pos_pair + ':' + str(idx))
            out_fd.write('\n')
        out_fd.close()
    logging.debug('[phrases_type_encode] pos mapping is done.')

    with open(global_settings.g_tw_raw_phrases_output_by_pos, 'w+') as out_fd:
        json.dump(d_phrases_pos, out_fd)
        out_fd.close()
    logging.debug('[phrases_type_encode] raw phrases by pos are done.')


def extract_vocab_from_phrases():
    s_vocab = set([])
    with open(global_settings.g_tw_raw_phrases_output, 'r') as in_fd:
        for ln in in_fd:
            fields = [item.strip() for item in ln.split('|')]
            phrase = fields[0]
            l_tokens = [token.strip().lower() for token in phrase.split(',')]
            for token in l_tokens:
                l_sub_tokens = [sub_token.strip() for sub_token in token.split(' ')]
                for sub_token in l_sub_tokens:
                    s_vocab.add(sub_token)
        in_fd.close()
    with open(global_settings.g_tw_raw_phrases_vocab_file, 'w+') as out_fd:
        out_str = '\n'.join(sorted(s_vocab))
        out_fd.write(out_str)
        out_fd.close()
    logging.debug('[extract_vocab_from_phrases] All done: %s tokens in total.' % len(s_vocab))


'''Phrase Clustering'''
# def make_phrase_embeds_tasks():
#     with open(global_settings.g_tw_raw_phrases_unique, 'r') as in_fd:
#         d_phrases = json.load(in_fd)
#         in_fd.close()
#     l_phrases = list(d_phrases.keys())
#     logging.debug('[make_phrase_embeds_tasks] All done. %s phrases.' % len(l_phrases))
#     return l_phrases

def load_lexvec_model():
    # global g_lexvec_model, g_embedding_len
    lexvec_model = lexvec.Model(global_settings.g_lexvec_vect_file_path)
    embedding_len = len(lexvec_model.word_rep('the'))
    logging.debug('[load_lexvec_model] The length of embeddings is %s' % embedding_len)
    return lexvec_model, embedding_len


def phrase_embedding(lexvec_model, embedding_len, phrase_str):
    '''
    Take a phrase (i.e. a sequence of tokens connected by whitespaces) and compute an embedding for it.
    This function acts as a wrapper of word embeddings. Refactor this function to adapt various word embedding models.
    :param
        phrase_str: An input phrase
    :return:
        An embedding for the phrase
    '''
    if global_settings.g_word_embedding_model == 'lexvec':
        if lexvec_model is None:
            raise Exception('lexvec_model is not loaded!')
        phrase_vec = np.zeros(embedding_len)
        l_words = [word.strip().lower() for word in phrase_str.split(' ')]
        for word in l_words:
            word_vec = lexvec_model.word_rep(word)
            phrase_vec += word_vec
        if not np.isfinite(phrase_vec).all():
            logging.error('Invalid embedding for %s!' % phrase_str)
            phrase_vec = np.zeros(embedding_len)
        return phrase_vec

# @profile
def phrase_embeds_single_proc(df_id_ph, lexvec_model, embedding_len, t_id):
    timer_start = time.time()
    logging.debug('[phrase_embeds_single_proc] Proc %s: %s tasks.' % (t_id, len(df_id_ph)))

    cnt = 0
    l_ph_embed_recs = []
    for ph_rec in df_id_ph.values:
        ph_id = int(ph_rec[0])
        ph_str = ph_rec[1]
        ph_embed = phrase_embedding(lexvec_model, embedding_len, ph_str)
        ph_embed = np.asarray(ph_embed, dtype=np.float32)
        ph_embed = preprocessing.normalize(ph_embed.reshape(1, -1))
        ph_embed = ph_embed[0]
        l_ph_embed_recs.append((ph_id, ph_embed))
        cnt += 1
        if cnt % 5000 == 0 and cnt >= 5000:
            logging.debug('[phrase_embeds_single_proc] Proc %s: embed %s phrases in %s secs.'
                          % (t_id, cnt, time.time() - timer_start))
    logging.debug('[phrase_embeds_single_proc] Proc %s: embed all %s phrases in % secs.'
                  % (t_id, cnt, time.time() - timer_start))

    out_df = pd.DataFrame(l_ph_embed_recs, columns=['phid', 'embed'])
    out_df.to_pickle(global_settings.g_tw_raw_phrases_embeds_int_format.format(str(t_id)))
    logging.debug('[phrase_embeds_single_proc] Proc %s: All done in % secs.'
                  % (t_id, time.time() - timer_start))


# @profile
def phrase_embeds_multiproc(num_procs, job_id):
    logging.debug('[phrase_embeds_multiproc] Starts...')
    timer_start = time.time()

    with open(global_settings.g_tw_raw_phrases_id_to_phrase, 'r') as in_fd:
        d_id_to_ph = json.load(in_fd)
        in_fd.close()
    l_id_ph = [(phid, d_id_to_ph[phid]) for phid in d_id_to_ph]
    df_id_ph = pd.DataFrame(l_id_ph, columns=['phid', 'ph'])
    df_id_ph.sort_values(by=['phid'])
    num_tasks = len(df_id_ph)
    print('[phrase_embeds_multiproc] Samples of df_id_ph:')
    print(df_id_ph[:10])
    logging.debug('[phrase_embeds_multiproc] Load g_tw_raw_phrases_id_to_phrase with %s phrases in %s secs.'
                  % (num_tasks, str(time.time() - timer_start)))

    batch_size = math.ceil(num_tasks / int(num_procs))
    l_batches = []
    for i in range(0, num_tasks, batch_size):
        if i + batch_size < num_tasks:
            l_batches.append(df_id_ph[i:i + batch_size])
        else:
            l_batches.append(df_id_ph[i:])
    logging.debug('[phrase_embeds_multiproc] %s procs.' % len(l_batches))

    l_procs = []
    t_id = 0
    for each_batch in l_batches:
        lexvec_model, embedding_len = load_lexvec_model()
        t = multiprocessing.Process(target=phrase_embeds_single_proc,
                                    args=(each_batch, lexvec_model, embedding_len, str(job_id) + '_' + str(t_id)))
        t.name = 't_mul_task_' + str(job_id) + '_' + str(t_id)
        t.start()
        l_procs.append(t)
        t_id += 1

    while len(l_procs) > 0:
        for t in l_procs:
            if t.is_alive():
                t.join(1)
            else:
                l_procs.remove(t)
                logging.debug('[phrase_embeds_multiproc] Proc %s is finished.' % t.name)

    logging.debug('[phrase_embeds_multiproc] All done in %s sec for %s tasks.'
                  % (time.time() - timer_start, num_tasks))


def phrase_embeds_int_to_out():
    logging.debug('[phrase_embeds_int_to_out] Starts...')
    timer_start = time.time()

    l_ph_embed_df = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_raw_phrases_embeds_int_folder):
        for filename in filenames:
            if filename[:14] != 'phrase_embeds_' or filename[-7:] != '.pickle':
                continue
            df_ph_embed = pd.read_pickle(dirpath + filename)
            l_ph_embed_df.append(df_ph_embed)
    out_df_ph_embed = pd.concat(l_ph_embed_df)
    out_df_ph_embed = out_df_ph_embed.sort_values(by=['phid'])
    out_df_ph_embed.to_pickle(global_settings.g_tw_raw_phrases_embeds)
    logging.debug('[phrase_embeds_int_to_out] All done %s phrases in %s secs.'
                  % (len(out_df_ph_embed), time.time() - timer_start))


'''
WARNING!
    The raw phrase embeding files can be too large to be filled into memory, and this function can be killed for 
    out-of-memory!
    So try not to use this function!
'''
# def merge_raw_phrase_embeds():
#     timer_start = time.time()
#     l_recs = []
#     for (dirpath, dirname, filenames) in walk(global_settings.g_tw_raw_phrases_embeds_int_folder):
#         for filename in filenames:
#             if filename[:18] != 'raw_phrase_embeds_':
#                 continue
#             with open(dirpath + '/' + filename, 'r') as in_fd:
#                 for ln in in_fd:
#                     fields = ln.split('|')
#                     phrase = fields[0].strip()
#                     embed = [float(ele.strip()) for ele in fields[1].split(',')]
#                     rec = (phrase, embed)
#                     l_recs.append(rec)
#                 in_fd.close()
#     out_df = pd.DataFrame(l_recs, columns=['phrase', 'embed'])
#     logging.debug('[merge_raw_phrase_embeds] out_df is done in %s secs, size=%s.'
#                   % (time.time() - timer_start, sys.getsizeof(out_df)))
#     out_df.memory_usage(index=True)
#     logging.debug('[merge_raw_phrase_embeds] out_df output is done in % secs.' % str(time.time() - timer_start))
#     out_df.to_pickle(global_settings.g_tw_raw_phrases_embeds)


# def merge_raw_phrase_embeds():
#     timer_start = time.time()
#     d_phrase_indexes = dict()
#     l_embeds = []
#     cnt = 0
#     for (dirpath, dirname, filenames) in walk(global_settings.g_tw_raw_phrases_embeds_int_folder):
#         for filename in filenames:
#             if filename[:20] != 'raw_phrase_embeds_0_' or filename[-5:] != '.json':
#                 continue
#             with open(dirpath + '/' + filename, 'r') as in_fd:
#                 d_raw_phrase_embeds = json.load(in_fd)
#                 in_fd.close()
#                 logging.debug('[merge_raw_phrase_embeds] Read in %s in %s secs.' % (filename, time.time() - timer_start))
#             for phrase_id in d_raw_phrase_embeds:
#                 rec = (phrase_id, d_raw_phrase_embeds[phrase_id])
#                 l_embeds.append(rec)
#                 cnt += 1
#                 if cnt % 10000 == 0 and cnt >= 10000:
#                     logging.debug('[merge_raw_phrase_embeds] %s embed recs in %s secs.'
#                                   % (cnt, time.time() - timer_start))
#     logging.debug('[merge_raw_phrase_embeds] %s embed recs in %s secs.'
#                   % (cnt, time.time() - timer_start))
#     l_embeds = sorted(l_embeds, key=lambda k: int(k[0]), reverse=True)
#     out_df = pd.DataFrame(l_embeds, columns=['phrase_id', 'embed'])
#     out_df.to_pickle(global_settings.g_tw_raw_phrases_embeds)
#     logging.debug('[convert_raw_phrase_embed_strs_to_vecs] Output g_tw_raw_phrases_embeds in %s secs.'
#                   % str(time.time() - timer_start))
#     logging.debug('[convert_raw_phrase_embed_strs_to_vecs] All done.')


# def build_reverse_phrase_index():
#     '''
#     Also, this function will lowercase all phrases.
#     '''
#     timer_start = time.time()
#     with open(global_settings.g_tw_raw_phrases_embeds_npy_phrase_idx, 'r') as in_fd:
#         d_phrase_idx = json.load(in_fd)
#         in_fd.close()
#         d_phrase_idx = {key.lower(): d_phrase_idx[key] for key in d_phrase_idx}
#         d_idx_phrase = {d_phrase_idx[key]: key for key in d_phrase_idx}
#     logging.debug('[build_reverse_phrase_index] d_phrase_idx and d_idx_phrase are done in %s secs'
#                   % str(time.time() - timer_start))
#     with open(global_settings.g_tw_raw_phrases_embeds_npy_phrase_idx, 'w+') as out_fd:
#         json.dump(d_phrase_idx, out_fd)
#         out_fd.close()
#     logging.debug('[build_reverse_phrase_index] d_phrase_idx is output in %s secs.' % str(time.time() - timer_start))
#     with open(global_settings.g_tw_raw_phrases_embeds_npy_rev_phrase_idx, 'w+') as out_fd:
#         json.dump(d_idx_phrase, out_fd)
#         out_fd.close()
#     logging.debug('[build_reverse_phrase_index] d_idx_phrase is output in %s secs' % str(time.time() - timer_start))


def build_phrase_clustering_dataset():
    logging.debug('[build_phrase_clustering_dataset] Starts...')
    timer_start = time.time()

    df_ph_embed = pd.read_pickle(global_settings.g_tw_raw_phrases_embeds)
    np_ph_embed = np.stack(df_ph_embed['embed'].to_list())
    logging.debug('[build_phrase_clustering_dataset] Load g_tw_raw_phrases_embeds in %s secs. Shape = %s'
                  % (str(time.time() - timer_start), str(np_ph_embed.shape)))

    np.save(global_settings.g_tw_raw_phrases_embeds_for_clustering, np_ph_embed)
    logging.debug('[build_phrase_clustering_dataset] Output g_tw_raw_phrases_embeds_for_clustering in %s secs.'
                  % str(time.time() - timer_start))
    logging.debug('[build_phrase_clustering_dataset] All done.')


def phrase_clustering(n_clusters):
    logging.debug('[phrase_clustering] Starts with %s clusters...' % str(n_clusters))
    timer_start = time.time()

    raw_phrase_embeds = np.load(global_settings.g_tw_raw_phrases_embeds_for_clustering)
    logging.debug('[phrase_clustering] Load in g_tw_raw_phrases_embeds_for_clustering in %s secs, shape = %s'
                  % (time.time() - timer_start, raw_phrase_embeds.shape))

    kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1).fit(raw_phrase_embeds)
    logging.debug('[phrase_clustering] clustering done in %s secs.' % (str(time.time() - timer_start)))
    labels = kmeans.labels_
    with open(global_settings.g_tw_raw_phrases_clustering_labels_format.format(str(n_clusters)), 'w+') as out_fd:
        out_str = '\n'.join([str(ele) for ele in labels])
        out_fd.write(out_str)
        out_fd.close()
    logging.debug('[phrase_clustering] All done in %s secs.' % (str(time.time() - timer_start)))
    # silhouette = metrics.silhouette_score(raw_phrase_embeds, labels, metric='euclidean')
    # calinski_harabasz = metrics.calinski_harabasz_score(raw_phrase_embeds, labels)
    # davies_bouldin = metrics.davies_bouldin_score(raw_phrase_embeds, labels)
    # logging.debug('[phrase_clustering] silhouette = %s, calinski_harabasz = %s, davies_bouldin = %s'
    #               % (silhouette, calinski_harabasz, davies_bouldin))


def compute_phrase_cluster_info(n_clusters):
    '''
    phrase ids start from 0
    '''
    logging.debug('[compute_phrase_cluster_centers] Starts...')
    timer_start = time.time()

    l_labels = []
    with open(global_settings.g_tw_raw_phrases_clustering_labels_format.format(str(n_clusters)), 'r') as in_fd:
        for ln in in_fd:
            label = ln.strip()
            l_labels.append(label)
        in_fd.close()
    logging.debug('[compute_phrase_cluster_centers] Load %s cluster labels in %s secs.'
                  % (len(l_labels), str(time.time() - timer_start)))

    df_ph_embed = pd.read_pickle(global_settings.g_tw_raw_phrases_embeds)
    logging.debug('[compute_phrase_cluster_centers] Load g_tw_raw_phrases_embeds %s in %s secs. '
                  % (len(df_ph_embed), time.time() - timer_start))

    df_ph_embed['cluster'] = l_labels
    l_clusters = sorted([int(ele) for ele in list(set(l_labels))])
    if len(l_clusters) != n_clusters:
        raise Exception('[compute_phrase_cluster_centers] Incorrect num of clusters! %s' % str(len(l_clusters)))
    logging.debug('[compute_phrase_cluster_centers] Attach cluster labels')

    out_str = ''
    l_cluster_center_recs = []
    for c_id in l_clusters:
        np_c_ph_embeds = np.stack(df_ph_embed.loc[df_ph_embed['cluster'] == str(c_id)]['embed'].to_list())
        if len(np_c_ph_embeds.shape) != 2:
            raise Exception('[compute_phrase_cluster_centers] Incorrect embeds for cluster %s' % str(c_id))
        c_center_vec = np.sum(np_c_ph_embeds, axis=0) / np_c_ph_embeds.shape[0]
        c_center_vec = preprocessing.normalize(c_center_vec.reshape(1, -1))[0]
        if not np.isfinite(c_center_vec).all():
            raise Exception('[compute_phrase_cluster_centers] Invalid vec for cluster %s' % str(c_id))
        l_c_members = df_ph_embed.loc[df_ph_embed['cluster'] == str(c_id)]['phid'].to_list()
        if len(l_c_members) != np_c_ph_embeds.shape[0]:
            raise Exception('[compute_phrase_cluster_centers] Invalid memberships for cluster %s' % str(c_id))
        np_member_to_center_sims = np.matmul(np_c_ph_embeds, c_center_vec)
        l_cluster_center_recs.append((c_id, c_center_vec, l_c_members, np_member_to_center_sims))
        out_str += 'Cluster %s: %s' % (c_id, len(l_c_members))
        out_str += '\n'
    logging.debug('[compute_phrase_cluster_centers] %s center vecs, members and sims done in % secs.'
                  % (len(l_cluster_center_recs), time.time() - timer_start))
    logging.debug('[compute_phrase_cluster_centers] Cluster stats: \n %s' % out_str)

    df_c_center = pd.DataFrame(l_cluster_center_recs, columns=['cid', 'cvec', 'cmember', 'm2csim'])
    df_c_center.to_pickle(global_settings.g_tw_raw_phrases_clustering_info_format.format(str(n_clusters)))
    logging.debug('[compute_phrase_cluster_centers] Output g_tw_raw_phrases_clustering_centers_and_groups in %s secs.'
                  % str(time.time() - timer_start))
    logging.debug('[compute_phrase_cluster_centers] All done.')


def convert_tw_phrases_to_clusters():
    logging.debug('[convert_tw_phrases_to_clusters] Starts...')
    with open(global_settings.g_tw_phrases_by_idx, 'r') as in_fd:
        d_tw_phrases_by_idx = json.load(in_fd)
        in_fd.close()
    logging.debug('[convert_tw_phrases_to_clusters] Load in g_tw_phrases_by_idx')

    l_phrase_cluster_labels = []
    with open(global_settings.g_tw_raw_phrases_clustering_labels_format.format(str(global_settings.g_num_phrase_clusters)), 'r') as in_fd:
        for ln in in_fd:
            l_phrase_cluster_labels.append(int(ln.strip()))
        in_fd.close()
    logging.debug('[convert_tw_phrases_to_clusters] Load in g_tw_raw_phrases_clustering_labels')

    d_tw_phrases_by_clusters = dict()
    for tw_id in d_tw_phrases_by_idx:
        d_tw_phrases_by_clusters[tw_id] = dict()
        l_phrases = d_tw_phrases_by_idx[tw_id]
        for phrase_idx in l_phrases:
            cluster_label = l_phrase_cluster_labels[phrase_idx]
            if cluster_label not in d_tw_phrases_by_clusters[tw_id]:
                d_tw_phrases_by_clusters[tw_id][cluster_label] = 1
            else:
                d_tw_phrases_by_clusters[tw_id][cluster_label] += 1
    with open(global_settings.g_tw_phrases_by_clusters_format.format(str(global_settings.g_num_phrase_clusters)), 'w+') as out_fd:
        json.dump(d_tw_phrases_by_clusters, out_fd)
        out_fd.close()
    logging.debug('[convert_tw_phrases_to_clusters] Output d_tw_phrases_by_clusters done.')


def phrase_clustering_stats(n_clusters):
    logging.debug('[phrase_clustering_stats] Starts...')
    timer_start = time.time()

    df_c_info = pd.read_pickle(global_settings.g_tw_raw_phrases_clustering_info_format.format(str(n_clusters)))
    logging.debug('[phrase_clustering_stats] Load g_tw_raw_phrases_clustering_info in %s secs.'
                  % str(time.time() - timer_start))

    l_clusters = df_c_info['cid'].to_list()
    if len(l_clusters) != n_clusters:
        raise Exception('[phrase_clustering_stats] Incorrect num of clusters!')
    np_cvecs = np.stack(df_c_info['cvec'].to_list())
    center_sim_mat = np.matmul(np_cvecs, np.transpose(np_cvecs))
    l_center_dists = []
    for i in range(n_clusters - 1):
        for j in range(i + 1, n_clusters):
            l_center_dists.append(center_sim_mat[i][j])
    l_center_dists = sorted(l_center_dists)
    dist_pdf = stats.norm.pdf(l_center_dists, np.mean(l_center_dists), np.std(l_center_dists))
    plt.subplot(2, 1, 1)
    plt.plot(l_center_dists, dist_pdf)
    plt.hist(l_center_dists, density=True)

    plt.subplot(2, 1, 2)
    plt.imshow(center_sim_mat, cmap='hot')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(global_settings.g_tw_raw_phrases_clustering_center_sim_heatmap_format.format(str(n_clusters)), format="PNG")
    plt.clf()

    for cid in l_clusters:
        l_m2csim = sorted(df_c_info.loc[df_c_info['cid'] == cid]['m2csim'].to_list())
        dist_pdf = stats.norm.pdf(l_m2csim, np.mean(l_m2csim), np.std(l_m2csim))
        plt.plot(l_m2csim, dist_pdf)
        # plt.hist(l_dists, density=True)
    plt.savefig(global_settings.g_tw_raw_phrases_clustering_member_to_center_sim_fig_format.format(str(n_clusters)), format="PNG")
    plt.clf()


# @profile
def classify_phrase_embeds_to_cluster_space_single_proc(n_clusters, task_id, phid_offset, batch_size, p_id):
    '''
    We don't normalize the pc_embeds so that it'd be more convenient to have probability vecs when building
    response graphs.
    '''
    logging.debug('[classify_phrase_embeds_to_cluster_space_single_proc] Proc %s: Starts with phid_offset = %s batch_size = %s...'
                  % (str(p_id), phid_offset, batch_size))
    timer_start = time.time()

    df_center_info = pd.read_pickle(global_settings.g_tw_raw_phrases_clustering_info_format.format(str(n_clusters)))
    np_cvecs = np.stack(df_center_info['cvec'].to_list())
    logging.debug('[classify_phrase_embeds_to_cluster_space_single_proc] Proc %s: Load in g_tw_raw_phrases_clustering_info in %s secs.'
                  % (p_id, str(time.time() - timer_start)))

    df_phrase_embeds = pd.read_pickle(global_settings.g_tw_phrase_cluster_embeds_tasks_file_format.format(task_id))
    np_phrase_embeds = np.stack(df_phrase_embeds['embed'].to_list())
    logging.debug('[classify_phrase_embeds_to_cluster_space_single_proc] Proc %s: Load g_tw_phrase_cluster_embeds_tasks_file %s in %s secs.'
                  % (str(p_id), len(np_phrase_embeds), time.time() - timer_start))

    np_pc_embeds = np.matmul(np_phrase_embeds, np.transpose(np_cvecs))
    threshold_func = np.vectorize(lambda x: x if x > global_settings.g_sim_threshold_for_phrase_cluster_embeds else 0.0)
    threshold_func(np_pc_embeds)
    # np_pc_embeds = preprocessing.normalize(np_pc_embeds)
    logging.debug('[classify_phrase_embeds_to_cluster_space_single_proc] Proc %s: Compute %s phrase cluster embeds in %s secs.'
                  % (p_id, len(np_pc_embeds), time.time() - timer_start))

    l_pc_embed_recs = zip(df_phrase_embeds['phid'].values.tolist(), np_pc_embeds.tolist())
    df_pc_embeds = pd.DataFrame(l_pc_embed_recs, columns=['phid', 'pcembed'])
    df_pc_embeds.to_pickle(global_settings.g_tw_phrase_cluster_embeds_int_file_format.format(str(n_clusters) + '_' + str(p_id)))
    logging.debug('[classify_phrase_embeds_to_cluster_space_single_proc] Proc %s: All done in %s secs.'
                  % (str(p_id), time.time() - timer_start))


def classify_phrase_embeds_to_cluster_space_multiproc(n_clusters, num_procs, job_id):
    timer_start = time.time()

    df_phrase_embeds = pd.read_pickle(global_settings.g_tw_raw_phrases_embeds)
    num_phrase_embeds = len(df_phrase_embeds)
    logging.debug('[classify_phrase_embeds_to_cluster_space_multiproc] Load g_tw_raw_phrases_embeds with %s phrase embeds in %s secs.'
                  % (num_phrase_embeds, str(time.time() - timer_start)))

    batch_size = math.ceil(num_phrase_embeds / int(num_procs))
    l_tasks = []
    offset = 0
    while offset + batch_size < num_phrase_embeds:
        df_tasks = df_phrase_embeds[offset : offset + batch_size]
        task_id = str(n_clusters) + '_' + str(offset)
        df_tasks.to_pickle(global_settings.g_tw_phrase_cluster_embeds_tasks_file_format.format(task_id))
        l_tasks.append((task_id, offset, batch_size))
        offset += batch_size
    if offset < num_phrase_embeds:
        df_tasks = df_phrase_embeds[offset : offset + batch_size]
        task_id = str(n_clusters) + '_' + str(offset)
        df_tasks.to_pickle(global_settings.g_tw_phrase_cluster_embeds_tasks_file_format.format(task_id))
        l_tasks.append((task_id, offset, len(df_tasks)))
    logging.debug('[classify_phrase_embeds_to_cluster_space_multiproc] Output %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))

    l_procs = []
    p_id = 0
    for task_id, offset, batch_size in l_tasks:
        t = multiprocessing.Process(target=classify_phrase_embeds_to_cluster_space_single_proc,
                                    args=(n_clusters, task_id, offset, batch_size, str(job_id) + '_' + str(p_id)))
        t.name = 't_mul_task_' + str(job_id) + '_' + str(p_id)
        t.start()
        l_procs.append(t)
        p_id += 1

    while len(l_procs) > 0:
        for t in l_procs:
            if t.is_alive():
                t.join(1)
            else:
                l_procs.remove(t)
                logging.debug('[classify_phrase_embeds_to_cluster_space_multiproc] Proc %s is finished.' % t.name)

    logging.debug('[classify_phrase_embeds_to_cluster_space_multiproc] All done in %s sec for %s tasks.'
                  % (time.time() - timer_start, len(l_tasks)))


def phrase_cluster_embeds_int_to_out(n_cluster):
    logging.debug('[phrase_cluster_embeds_int_to_out] Starts...')
    timer_start = time.time()
    l_pc_dfs = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_phrase_cluster_embeds_int_folder):
        for filename in filenames:
            if filename[:26] != 'phrase_cluster_embeds_int_' or filename[-7:] != '.pickle':
                continue
            df_pcembeds = pd.read_pickle(dirpath + filename)
            l_pc_dfs.append(df_pcembeds)
    df_pcembeds_all = pd.concat(l_pc_dfs)
    df_pcembeds_all = df_pcembeds_all.sort_values(by=['phid'])
    df_pcembeds_all['pcembed'] = df_pcembeds_all['pcembed'].apply(np.asarray, args=(np.float32,))
    df_pcembeds_all.to_pickle(global_settings.g_tw_phrase_cluster_embeds_all_file_format.format(str(n_cluster)))
    logging.debug('[phrase_cluster_embeds_int_to_out] Output g_tw_phrase_cluster_embeds_all_file with %s phrase embeds in %s secs.'
                  % (len(df_pcembeds_all), time.time() - timer_start))
    logging.debug('[phrase_cluster_embeds_int_to_out] All done.')


def phrase_cluster_label_onehot_embed(n_cluster):
    logging.debug('[phrase_cluster_label_onehot_embed] Starts...')
    timer_start = time.time()

    l_pc_onhot = []
    with open(global_settings.g_tw_raw_phrases_clustering_labels_format.format(str(n_cluster)), 'r') as in_fd:
        phid = 0
        for ln in in_fd:
            onehot_id = int(ln.strip())
            pc_onehot_vec = np.zeros(n_cluster, dtype=np.float32)
            pc_onehot_vec[onehot_id] = 1.0
            l_pc_onhot.append((phid, pc_onehot_vec))
            phid += 1
    df_pc_onehot = pd.DataFrame(l_pc_onhot, columns=['phid', 'onehot'])
    df_pc_onehot.to_pickle(global_settings.g_tw_phrase_cluster_onehot_embeds_all_file_format.format(str(n_cluster)))
    logging.debug('[phrase_cluster_label_onehot_embed] All done with %s phrase onehot embeds in %s secs.'
                  % (len(df_pc_onehot), time.time() - timer_start))


def tw_to_onehot_vec_single_proc(n_clusters, task_id, p_id):
    logging.debug('[tw_to_onehot_vec_single_proc] Proc %s: Starts with %s...' % (p_id, task_id))
    timer_start = time.time()

    df_task = pd.read_pickle(global_settings.g_tw_pc_onehot_task_file_format.format(task_id))
    logging.debug('[tw_to_onehot_vec_single_proc] Proc %s: Load %s tasks in %s secs.'
                  % (p_id, len(df_task), time.time() - timer_start))

    df_ph_onehot = pd.read_pickle(global_settings.g_tw_phrase_cluster_onehot_embeds_all_file_format.format(str(n_clusters)))
    df_ph_onehot = df_ph_onehot.set_index('phid')
    logging.debug('[tw_to_onehot_vec_single_proc] Proc %s: Load g_tw_phrase_cluster_onehot_embeds_all_file with %s recs in %s secs.'
                  % (p_id, len(df_ph_onehot), time.time() - timer_start))

    cnt = 0
    l_ready_recs = []
    for _, rec in df_task.iterrows():
        tw_id = rec['tw_id']
        l_phids = rec['raw_phs']
        tw_pc_onehot = np.zeros(n_clusters, dtype=np.float32)
        for phid in l_phids:
            ph_onehot = df_ph_onehot.loc[phid]['onehot']
            tw_pc_onehot += ph_onehot
        tw_pc_onehot = tw_pc_onehot / np.sum(tw_pc_onehot)
        if not np.isfinite(tw_pc_onehot).all():
            raise Exception('[tw_to_onehot_vec_single_proc] Proc %s: tw %s has invalid pc_onehot.' % (p_id, tw_id))
        l_ready_recs.append((tw_id, tw_pc_onehot))
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[tw_to_onehot_vec_single_proc] Proc %s: %s ready recs in %s secs.'
                          % (p_id, cnt, time.time() - timer_start))
    logging.debug('[tw_to_onehot_vec_single_proc] Proc %s:  All %s ready recs in %s secs.'
                  % (p_id, cnt, time.time() - timer_start))

    df_out = pd.DataFrame(l_ready_recs, columns=['tw_id', 'pc_onehot'])
    df_out.to_pickle(global_settings.g_tw_pc_onehot_int_file_format.format(task_id))
    logging.debug('[tw_to_onehot_vec_single_proc] Proc %s: All done in %s secs.' % (p_id, time.time() - timer_start))


def tw_to_onehot_vec_multiproc(num_procs, n_clusters, job_id):
    logging.debug('[tw_to_onehot_vec_multiproc] Starts...')
    timer_start = time.time()

    df_tw_to_phids = pd.read_pickle(global_settings.g_tw_phrase_extraction_tw_to_phids_file)
    num_tasks = len(df_tw_to_phids)
    logging.debug('[tw_to_onehot_vec_multiproc] Load g_tw_phrase_extraction_tw_to_phids_file with %s recs in %s secs.'
                  % (num_tasks, time.time() - timer_start))

    batch_size = math.ceil(num_tasks / int(num_procs))
    l_tasks = []
    offset = 0
    while offset + batch_size < num_tasks:
        df_tasks = df_tw_to_phids[offset : offset + batch_size]
        task_id = str(n_clusters) + '_' + str(offset)
        df_tasks.to_pickle(global_settings.g_tw_pc_onehot_task_file_format.format(task_id))
        l_tasks.append((task_id, offset, batch_size))
        offset += batch_size
    if offset < num_tasks:
        df_tasks = df_tw_to_phids[offset : offset + batch_size]
        task_id = str(n_clusters) + '_' + str(offset)
        df_tasks.to_pickle(global_settings.g_tw_pc_onehot_task_file_format.format(task_id))
        l_tasks.append((task_id, offset, len(df_tasks)))
    logging.debug('[tw_to_onehot_vec_multiproc] Output %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))

    l_procs = []
    p_id = 0
    for task_id, offset, batch_size in l_tasks:
        t = multiprocessing.Process(target=tw_to_onehot_vec_single_proc,
                                    args=(n_clusters, task_id, str(job_id) + '_' + str(p_id)))
        t.name = 't_mul_task_' + str(job_id) + '_' + str(p_id)
        t.start()
        l_procs.append(t)
        p_id += 1

    while len(l_procs) > 0:
        for t in l_procs:
            if t.is_alive():
                t.join(1)
            else:
                l_procs.remove(t)
                logging.debug('[tw_to_onehot_vec_multiproc] Proc %s is finished.' % t.name)

    logging.debug('[tw_to_onehot_vec_multiproc] All done in %s sec for %s tasks.'
                  % (time.time() - timer_start, len(l_tasks)))


def tw_to_onehot_vec_int_to_out(n_clusters):
    logging.debug('[tw_to_onehot_vec_int_to_out] Starts...')
    l_tw_onehot_dfs = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_pc_onehot_int_folder):
        for filename in filenames:
            if filename[:17] != 'tw_pc_onehot_int_' or filename[-7:] != '.pickle':
                continue
            df_tw_onehot = pd.read_pickle(dirpath + filename)
            l_tw_onehot_dfs.append(df_tw_onehot)
    df_out = pd.concat(l_tw_onehot_dfs)
    df_out.to_pickle(global_settings.g_tw_pc_onehot_file_format.format(str(n_clusters)))
    logging.debug('[tw_to_onehot_vec_int_to_out] All done.')


# def load_phrase_id_to_we_id(n_cluster):
#     d_phrase_id_to_embed_id = dict()
#     with open(global_settings.g_tw_raw_phrases_weid2phid, 'r') as in_fd:
#         for ln_id, ln in enumerate(in_fd):
#             d_phrase_id_to_embed_id[ln.strip()] = ln_id
#         in_fd.close()
#     logging.debug('[load_phrase_id_to_we_id] Load in phrase_id_to_we_id')
#     return d_phrase_id_to_embed_id
#
#
# def load_we_id_to_phrase_id(cluster):
#     d_we_id_to_phrase_id = dict()
#     with open(global_settings.g_tw_raw_phrases_weid2phid, 'r') as in_fd:
#         for ln_id, ln in enumerate(in_fd):
#             d_we_id_to_phrase_id[ln_id] = ln.strip()
#         in_fd.close()
#     logging.debug('[load_we_id_to_phrase_id] Load in we_id_to_phrase_id')
#     return d_we_id_to_phrase_id

'''
Response Graph
'''
def make_response_graph_tasks(num_tasks):
    logging.debug('[make_response_graph_tasks] Starts...')
    with open(global_settings.g_tw_src_trg_tw_id_pairs, 'r') as in_fd:
        d_src_trg_tws = json.load(in_fd)
        in_fd.close()
    logging.debug('[make_response_graph_tasks] Load in g_tw_src_trg_tw_id_pairs: %s' % len(d_src_trg_tws))

    l_tasks = list(d_src_trg_tws.keys())
    batch_size = math.ceil(len(l_tasks) / num_tasks)
    l_l_subtasks = []
    for i in range(0, len(l_tasks), batch_size):
        if i + batch_size < len(l_tasks):
            l_l_subtasks.append(l_tasks[i:i + batch_size])
        else:
            l_l_subtasks.append(l_tasks[i:])
    logging.debug('[make_response_graph_tasks] Gen %s tasks' % str(len(l_l_subtasks)))

    for batch_id, batch in enumerate(l_l_subtasks):
        with open(global_settings.g_tw_phrases_response_graph_task_format.format(str(batch_id)), 'w+') as out_fd:
            out_str = '\n'.join(batch)
            out_fd.write(out_str)
            out_fd.close()
    logging.debug('[make_response_graph_tasks] Output all tasks.')


# @profile
def build_response_graph_single_proc(task_id, n_clusters, p_id):
    logging.debug('[build_response_graph_single_proc] Proc %s: Starts with task id %s.' % (p_id, task_id))
    timer_start = time.time()

    df_src_trg = pd.read_pickle(global_settings.g_tw_response_graph_task_file_format.format(task_id))
    logging.debug('[build_response_graph_single_proc] Proc %s: Load task %s with %s src-trg pairs in %s secs.'
                  % (p_id, task_id, len(df_src_trg), time.time() - timer_start))

    # df_pc_embed = pd.read_pickle(global_settings.g_tw_phrase_cluster_embeds_all_file_format.format(str(n_clusters)))
    df_pc_embed = pd.read_pickle(global_settings.g_tw_phrase_cluster_onehot_embeds_all_file_format.format(str(n_clusters)))
    d_pc_embeds = df_pc_embed.set_index('phid').to_dict(orient='index')
    logging.debug('[build_response_graph_single_proc] Proc %s: Load g_tw_phrase_cluster_onehot_embeds_all_file with %s embeds in %s secs.'
                  % (p_id, len(d_pc_embeds), time.time() - timer_start))

    df_tw_phid = pd.read_pickle(global_settings.g_tw_phrase_extraction_tw_to_phids_file)
    d_tw_phid = df_tw_phid.set_index('tw_id').to_dict(orient='index')
    logging.debug('[build_response_graph_single_proc] Proc %s: Load g_tw_phrase_extraction_tw_to_phids_file with %s tw-phid maps in %s secs.'
                  % (p_id, len(d_tw_phid), time.time() - timer_start))

    cnt = 0
    l_usr_beliefs_train_recs = []
    np_srg_pc_sum = []
    np_trg_pc_sum = []
    for src_trg_rec in df_src_trg.values:
        trg_tw_id = src_trg_rec[0]
        src_tw_id = src_trg_rec[1]
        if trg_tw_id not in d_tw_phid or src_tw_id not in d_tw_phid:
            continue
        src_pc_sum = np.zeros(n_clusters, dtype=np.float32)
        for src_phid in d_tw_phid[src_tw_id]['raw_phs']:
            if src_phid not in d_pc_embeds:
                continue
            # src_pcembed = d_pc_embeds[src_phid]['pcembed']
            src_pcembed = d_pc_embeds[src_phid]['onehot']
            src_pc_sum += src_pcembed
        trg_pc_sum = np.zeros(n_clusters, dtype=np.float32)
        for trg_phid in d_tw_phid[trg_tw_id]['raw_phs']:
            if trg_phid not in d_pc_embeds:
                continue
            # trg_pcembed = d_pc_embeds[trg_phid]['pcembed']
            trg_pcembed = d_pc_embeds[trg_phid]['onehot']
            trg_pc_sum += trg_pcembed
        # src_pc_sum = softmax(src_pc_sum)
        # trg_pc_sum = softmax(trg_pc_sum)
        src_pc_sum = src_pc_sum / np.sum(src_pc_sum)
        trg_pc_sum = trg_pc_sum / np.sum(trg_pc_sum)
        np_srg_pc_sum.append(src_pc_sum)
        np_trg_pc_sum.append(trg_pc_sum)
        l_usr_beliefs_train_recs.append((src_pc_sum, trg_pc_sum))
        cnt += 1
    df_usr_beliefs_train_set = pd.DataFrame(l_usr_beliefs_train_recs, columns=['src_ce', 'trg_ce'])
    df_usr_beliefs_train_set.to_pickle(global_settings.g_tw_usr_beliefs_train_set_int_file_format.format(task_id))
    logging.debug('[build_response_graph_single_proc] Proc %s: Collect %s src->trg pcembed pairs in %s secs.'
                  % (p_id, cnt, time.time() - timer_start))
    np_adj_mat = np.matmul(np.transpose(np_srg_pc_sum), np_trg_pc_sum)
    logging.debug('[build_response_graph_single_proc] Proc %s: Build np_adj_mat in %s secs. Shape = %s'
                  % (p_id, time.time() - timer_start, str(np_adj_mat.shape)))

    np.save(global_settings.g_tw_response_graph_int_file_format.format(task_id), np_adj_mat)
    logging.debug('[build_response_graph] Proc %s: All done. resp_graph is output in %s secs.'
                  % (p_id, time.time() - timer_start))


def build_response_graph_multiproc(task_name, n_clusters, num_procs, job_id):
    logging.debug('[build_response_graph_multiproc] Starts...')
    timer_start = time.time()

    df_src_trg = pd.read_pickle(global_settings.g_tw_src_trg_data_file_format.format(task_name))
    num_tasks = len(df_src_trg)
    logging.debug('[build_response_graph_multiproc] Load g_tw_src_trg_data_file with %s src-trg pairs in %s secs.'
                  % (len(df_src_trg), time.time() - timer_start))

    batch_size = math.ceil(num_tasks / int(num_procs))
    l_task_ids = []
    for i in range(0, num_tasks, batch_size):
        task_id = task_name + '_' + str(n_clusters) + '_' + str(job_id) + '_' + str(i)
        df_src_trg[i:i + batch_size].to_pickle(global_settings.g_tw_response_graph_task_file_format.format(task_id))
        l_task_ids.append(task_id)
    logging.debug('[build_response_graph_multiproc] %s procs.' % len(l_task_ids))

    l_procs = []
    t_id = 0
    for task_id in l_task_ids:
        t = multiprocessing.Process(target=build_response_graph_single_proc,
                                    args=(task_id, n_clusters, str(job_id) + '#' + str(t_id)))
        t.name = 't_mul_task_' + str(job_id) + '#' + str(t_id)
        t.start()
        l_procs.append(t)
        t_id += 1

    while len(l_procs) > 0:
        for t in l_procs:
            if t.is_alive():
                t.join(1)
            else:
                l_procs.remove(t)
                logging.debug('[build_response_graph_multiproc] Proc %s is finished.' % t.name)

    logging.debug('[build_response_graph_multiproc] All done in %s sec for %s tasks.'
                  % (time.time() - timer_start, len(l_task_ids)))


def resp_graph_int_to_trans_mat(task_name):
    logging.debug('[resp_graph_int_to_trans_mat] Starts...')
    timer_start = time.time()

    resp_graph = None
    l_adj_mats = []
    cnt = 0
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_response_graph_int_folder):
        for filename in filenames:
            if filename[:15] != 'resp_graph_int_' or filename[-4:] != '.npy':
                continue
            np_adj_mat = np.load(dirpath + filename)
            l_adj_mats.append(np_adj_mat)
    adj_mat = np.sum(l_adj_mats, axis=0)
    adj_mat = np.apply_along_axis(lambda v: v / np.sum(v), 1, adj_mat)
    np.save(global_settings.g_tw_response_graph_transition_matrix_format.format(task_name), adj_mat)
    logging.debug('[resp_graph_int_to_trans_mat] Output g_tw_response_graph_transition_matrix in %s secs. Shape=%s'
                  % (time.time() - timer_start, str(adj_mat.shape)))
    logging.debug('[resp_graph_int_to_trans_mat] All done.')


def draw_trans_mat_heatmap(task_name):
    logging.debug('[draw_trans_mat_heatmap] Starts...')
    np_trans_mat = np.load(global_settings.g_tw_response_graph_transition_matrix_format.format(task_name))
    plt.imshow(np_trans_mat, cmap='hot')
    plt.colorbar()
    plt.savefig(global_settings.g_tw_response_graph_transition_matrix_fig_format.format(task_name), format="PNG")
    plt.clf()
    logging.debug('[draw_trans_mat_heatmap] All done.')


def build_comprehensive_src_trg_data(ds_name):
    logging.debug('[build_comprehensive_src_trg_data] Starts...')
    timer_start = time.time()

    df_src_trg = pd.read_pickle(global_settings.g_tw_src_trg_data_file_format.format(task_name))
    logging.debug('[build_comprehensive_src_trg_data] Load %s src-trg pairs in %s secs.'
                  % (len(df_src_trg), time.time() - timer_start))

    # df_tw_phid = pd.read_pickle(global_settings.g_tw_phrase_extraction_tw_to_phids_file)
    # logging.debug('[build_comprehensive_src_trg_data] Load g_tw_phrase_extraction_tw_to_phids_file with %s tw-phid maps in %s secs.'
    #               % (len(df_tw_phid), time.time() - timer_start))

    df_tw_phs = pd.read_pickle(global_settings.g_tw_phrase_extraction_tw_to_phrases_file)
    s_df_tw_phs_tw_ids = set(df_tw_phs['tw_id'].to_list())
    logging.debug('[build_comprehensive_src_trg_data] Load g_tw_phrase_extraction_tw_to_phids_file with %s tw phrases maps in %s secs.'
                  % (len(df_tw_phs), time.time() - timer_start))

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_sql_str = """select raw_txt from cp4.mf3jh_ven_tw_en_all where tw_id = %s"""

    l_ready_recs = []
    for rec in df_src_trg.values:
        trg_tw_id = rec[0]
        src_tw_id = rec[1]
        if trg_tw_id not in s_df_tw_phs_tw_ids:
            print('df_tw_phs trg_tw_id = %s' % trg_tw_id)
            continue
        trg_tw_phs = df_tw_phs.loc[df_tw_phs['tw_id'] == trg_tw_id]['raw_phs'].to_list()[0]
        if src_tw_id not in s_df_tw_phs_tw_ids:
            print('df_tw_phs src_tw_id = %s' % src_tw_id)
            continue
        src_tw_phs = df_tw_phs.loc[df_tw_phs['tw_id'] == src_tw_id]['raw_phs'].to_list()[0]
        tw_db_cur.execute(tw_sql_str, (trg_tw_id,))
        trg_tw_raw_txt = tw_db_cur.fetchone()
        if trg_tw_raw_txt is None:
            print('trg_tw_raw_txt = %s' % trg_tw_raw_txt)
            continue
        trg_tw_raw_txt = trg_tw_raw_txt[0]
        tw_db_cur.execute(tw_sql_str, (src_tw_id,))
        src_tw_raw_txt = tw_db_cur.fetchone()
        if src_tw_raw_txt is None:
            print('src_tw_raw_txt = %s' % src_tw_raw_txt)
            continue
        src_tw_raw_txt = src_tw_raw_txt[0]
        rec = (src_tw_id, trg_tw_id, src_tw_phs, trg_tw_phs, src_tw_raw_txt, trg_tw_raw_txt)
        l_ready_recs.append(rec)
    df_out = pd.DataFrame(l_ready_recs, columns=['src_tw_id', 'trg_tw_id', 'src_tw_phs', 'trg_tw_phs',
                                                 'src_tw_raw_txt', 'trg_tw_raw_txt'])
    df_out.to_pickle(global_settings.g_tw_src_trg_comprehensive_data_file_format.format(ds_name))
    logging.debug('[build_comprehensive_src_trg_data] All done with %s recs in %s secs.'
                  % (len(df_out), time.time() - timer_start))
    tw_db_cur.close()
    tw_db_conn.close()



# def get_transition_mat_from_resp_graph(n_clusters):
#     with open(global_settings.g_tw_phrases_response_graph_format.format(str(n_clusters)), 'r') as in_fd:
#         resp_graph_data = json.load(in_fd)
#         in_fd.close()
#     resp_graph = nx.adjacency_graph(resp_graph_data)
#
#     adj_mat = nx.adjacency_matrix(resp_graph).toarray().astype(np.float32)
#     for i in range(adj_mat.shape[0]):
#         adj_mat[i] = softmax(adj_mat[i])
#
#     np.save(global_settings.g_tw_phrases_response_graph_transition_mat_format.format(str(n_clusters)), adj_mat)
#     logging.debug('[get_transition_mat_from_resp_graph] Output g_tw_phrases_response_graph_transition_mat done.')
#     print(adj_mat)


'''
Semantics to Narratives Modeling
'''
def make_sem_to_nar_train_sets():

    return



'''
User Beliefs Modeling
'''
# def make_user_beliefs_train_sets(n_clusters, l_src_tw_ids, dt_start, dt_end, train_set_id):
#     logging.debug('[make_user_beliefs_train_sets] Starts with %s src_tw_ids...' % str(len(l_src_tw_ids)))
#     timer_start = time.time()
#     with open(global_settings.g_tw_src_trg_tw_id_pairs, 'r') as in_fd:
#         d_src_trg_tws = json.load(in_fd)
#         in_fd.close()
#     logging.debug('[make_user_beliefs_train_sets] Load in g_tw_src_trg_tw_id_pairs: %s' % len(d_src_trg_tws))
#
#     with open(global_settings.g_tw_phrases_by_idx, 'r') as in_fd:
#         d_tw_to_phrases = json.load(in_fd)
#         in_fd.close()
#     logging.debug('[make_user_beliefs_train_sets] Load in g_tw_phrases_by_idx')
#
#     np_phrase_cluster_space_embeds = np.load(global_settings.g_tw_raw_phrases_cluster_space_embeds_format.
#                                              format(str(n_clusters)))
#     logging.debug('[make_user_beliefs_train_sets] Load in g_tw_raw_phrases_cluster_space_embeds in %s secs. Shape = %s'
#                   % (str(time.time() - timer_start), np_phrase_cluster_space_embeds.shape))
#
#     d_phrase_id_to_embed_id = load_phrase_id_to_we_id(n_clusters)
#
#     l_src_trg_ce_pairs = []
#     for src_tw_id in l_src_tw_ids:
#         if src_tw_id not in d_tw_to_phrases:
#             continue
#
#         l_src_phrases = d_tw_to_phrases[src_tw_id]
#         src_ce = np.zeros(n_clusters, dtype=np.float32)
#         for src_phrase_id in l_src_phrases:
#             src_phrase_ceid = d_phrase_id_to_embed_id[str(src_phrase_id)]
#             src_phrase_ce = np_phrase_cluster_space_embeds[src_phrase_ceid]
#             src_ce += src_phrase_ce
#
#         l_trg_items = d_src_trg_tws[src_tw_id]
#         for trg_tw_id, tw_datetime in l_trg_items:
#             if dt_start is not None and tw_datetime < dt_start:
#                 continue
#             if dt_end is not None and tw_datetime > dt_end:
#                 continue
#             if trg_tw_id not in d_tw_to_phrases:
#                 continue
#             trg_ce = np.zeros(n_clusters, dtype=np.float32)
#             l_trg_phrases = d_tw_to_phrases[trg_tw_id]
#             for trg_phrase_id in l_trg_phrases:
#                 trg_phrase_ceid = d_phrase_id_to_embed_id[str(trg_phrase_id)]
#                 trg_phrase_ce = np_phrase_cluster_space_embeds[trg_phrase_ceid]
#                 trg_ce += trg_phrase_ce
#             l_src_trg_ce_pairs.append((src_ce, trg_ce))
#     out_df = pd.DataFrame(l_src_trg_ce_pairs, columns=['src_ce', 'trg_ce'])
#     out_df.to_pickle(global_settings.g_tw_usr_beliefs_train_set_format.format(train_set_id + '_' + str(n_clusters)))
#     logging.debug('[make_user_beliefs_train_sets] Train set done with %s src->trg ce pairs in %s secs.'
#                   % (len(l_src_trg_ce_pairs), time.time() - timer_start))

def make_user_beliefs_train_sets(ts_name):
    logging.debug('[make_user_beliefs_train_sets] Starts...')
    l_train_sets = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_usr_beliefs_train_set_int_folder):
        for filename in filenames:
            if filename[:26] != 'usr_beliefs_train_set_int_' or filename[-7:] != '.pickle':
                continue
            df_train_set = pd.read_pickle(dirpath + filename)
            l_train_sets.append(df_train_set)
    df_out = pd.concat(l_train_sets)
    df_out.to_pickle(global_settings.g_tw_usr_beliefs_train_set_file_format.format(ts_name))
    logging.debug('[make_user_beliefs_train_sets] All done with %s training recs.' % str(len(df_out)))


def user_beliefs_model(ts_name, n_clusters, pc_dep_d1_d, rd_d1_d, is_train=True, src_ce=None, model_name=None):
    import tensorflow as tf

    class PhraseClusterDep(tf.keras.layers.Layer):
        def __init__(self, d1_d):
            super(PhraseClusterDep, self).__init__()
            self.dense_1 = tf.keras.layers.Dense(d1_d, activation='selu')

        def call(self, input_vec):
            d_1_out = self.dense_1(input_vec)
            return d_1_out

    class RespDep(tf.keras.layers.Layer):
        def __init__(self, d1_d):
            super(RespDep, self).__init__()
            self.dense_1 = tf.keras.layers.Dense(d1_d, activation='selu')
            self.drop_1 = tf.keras.layers.Dropout(.2, input_shape=(150,))

        def call(self, input_vec, training=False):
            d_1_out = self.dense_1(input_vec)
            resp_dep_out = d_1_out
            # if training:
            #     resp_dep_out = self.drop_1(d_1_out)
            return resp_dep_out

    class UserBelief(tf.keras.Model):
        def __init__(self, pcd_d1_d, rd_d1_d, trans_mat):
            super(UserBelief, self).__init__()
            self.pc_dep_layer = PhraseClusterDep(d1_d=pcd_d1_d)
            self.resp_dep_layer = RespDep(d1_d=rd_d1_d)
            self.trans_mat = trans_mat

        def call(self, input_vec, training=False):
            pc_dep_out = self.pc_dep_layer(input_vec)
            # pc_dep_out = input_vec
            # p_resp_out = tf.transpose(tf.matmul(self.trans_mat, tf.transpose(pc_dep_out)))
            p_resp_out = pc_dep_out
            resp_dep_out = self.resp_dep_layer(p_resp_out, training=training)
            # resp_dep_out = p_resp_out
            resp_ret = tf.nn.softmax(resp_dep_out)
            return resp_ret

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    # TODO
    # [Strategy modification]
    def load_train_test_sets(df_input_set, batch_size, strategy):
        # df_train_set = pd.read_pickle(global_settings.g_tw_usr_beliefs_train_set_file_format.format(str(ts_id)))
        # np_src_ce = softmax(df_train_set.pop('src_ce').values.tolist(), axis=1)
        # np_trg_ce = softmax(df_train_set.pop('trg_ce').values.tolist(), axis=1)
        np_src_ce = df_input_set['src_ce'].values.tolist()
        np_trg_ce = df_input_set['trg_ce'].values.tolist()
        input_set = tf.data.Dataset.from_tensor_slices((np_src_ce, np_trg_ce))
        input_set = input_set.shuffle(len(df_train_set))
        train_set_size = int(0.8 * len(list(input_set)))
        train_set = input_set.take(train_set_size)
        test_set = input_set.skip(train_set_size)
        train_set = train_set.batch(batch_size)
        train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
        test_set = test_set.batch(batch_size)
        test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)
        # TODO
        # [Strategy modification]
        train_set = strategy.experimental_distribute_dataset(train_set)
        test_set = strategy.experimental_distribute_dataset(test_set)
        return train_set, test_set

    # @tf.function
    def train_step(user_belief_ins, src_ce, trg_ce, optimizer, loss_obj):
        with tf.GradientTape() as gt:
            pred_trg_ce = user_belief_ins(src_ce, training=True)
            loss = loss_obj(trg_ce, pred_trg_ce)
            grads = gt.gradient(loss, user_belief_ins.trainable_variables)
            optimizer.apply_gradients(zip(grads, user_belief_ins.trainable_variables))
        return loss

    def test_step(user_belief_ins, src_ce, trg_ce, loss_obj):
        pred_trg_ce = resp_infer(user_belief_ins, src_ce)
        loss = loss_obj(trg_ce, pred_trg_ce)
        return loss

    # @tf.function
    def train_distributed(user_belief_ins, opt_ins, epoch_cnt, loss_obj, train_loss_met, test_loss_met,
                          batch_size, df_train_set):
        # TODO
        # [Strategy modification]
        # [WARNING!] strategy has a very significant overhead when running on Rivanna.
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            for epoch in range(epoch_cnt):
                '''Training'''
                train_loss_met.reset_states()
                train_set, test_set = load_train_test_sets(df_train_set, batch_size, strategy)
                cnt = 0
                for src_ce, trg_ce in train_set:
                    # TODO
                    # [Strategy modification]
                    loss = strategy.run(train_step, args=(user_belief_ins, src_ce, trg_ce, opt_ins, loss_obj))
                    # loss = train_step(input_vecs, embed_ae, opt_ins, loss_obj, loss_obj_2)
                    train_loss_met(loss)
                    cnt += 1
                print('[train_distributed] Epoch:%s Rec:%s Training done. Loss:%s Time:%s'
                      % (epoch, cnt, train_loss_met.result().numpy(), time.time() - timer_start))

                '''Testing'''
                test_loss_met.reset_states()
                cnt = 0
                for src_ce, trg_ce in test_set:
                    test_loss = strategy.run(test_step, args=(user_belief_ins, src_ce, trg_ce, loss_obj))
                    test_loss_met(test_loss)
                    cnt += 1
                print('[train_distributed] Epoch:%s Rec:%s Testing done. Loss:%s Time:%s'
                      % (epoch, cnt, test_loss_met.result().numpy(), time.time() - timer_start))
            print('[train_distributed] All done. Train loss:%s Test loss:%s Time:%s'
                  % (train_loss_met.result().numpy(), test_loss_met.result().numpy(), time.time() - timer_start))
            return user_belief_ins

    def resp_infer(user_belief_ins, src_ce):
        pred_trg_ce = user_belief_ins(src_ce, training=False)
        return pred_trg_ce

    '''Function starts here'''
    trans_mat = np.load(global_settings.g_tw_phrases_response_graph_transition_mat_format.format(str(n_clusters)))
    for row_id in range(len(trans_mat)):
        trans_mat[row_id] = softmax(trans_mat[row_id])
        # trans_mat[row_id] = trans_mat[row_id] / np.sum(trans_mat[row_id])
    if is_train:
        df_train_set = pd.read_pickle(global_settings.g_tw_usr_beliefs_train_set_file_format.format(str(ts_name)))
        user_belief_ins = UserBelief(pc_dep_d1_d, rd_d1_d, trans_mat)
        learn_rate = CustomSchedule(pc_dep_d1_d)
        opt_ins = tf.keras.optimizers.Adam(learn_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        batch_size = 1
        epoch_cnt = 1500
        # ts_id = 'all_' + str(n_clusters)
        # ckpt_fmt = '_{0}'

        timer_start = time.time()
        loss_obj = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        train_loss_met = tf.keras.metrics.Mean()
        test_loss_met = tf.keras.metrics.Mean()

        user_belief_ins = train_distributed(user_belief_ins, opt_ins, epoch_cnt, loss_obj, train_loss_met, test_loss_met, batch_size, df_train_set)
        save_model_path = global_settings.g_tw_usr_beliefs_saved_models_folder + model_name + '/'
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        tf.saved_model.save(user_belief_ins, save_model_path)

    else:
        save_model_path = global_settings.g_tw_usr_beliefs_saved_models_folder + model_name + '/'
        user_belief_ins = tf.saved_model.load(save_model_path)
        pred_trg_ce = resp_infer(user_belief_ins, src_ce)
        return pred_trg_ce


def random_baseline(ts_name, n_clusters, n_trials):
    import tensorflow as tf
    logging.debug('[random_baseline] Starts with %s...' % ts_name)
    timer_start = time.time()
    batch_size = 1

    df_input_set = pd.read_pickle(global_settings.g_tw_usr_beliefs_train_set_file_format.format(str(ts_name)))
    np_src_ce = df_input_set['src_ce'].values.tolist()
    np_trg_ce = df_input_set['trg_ce'].values.tolist()
    input_set = tf.data.Dataset.from_tensor_slices((np_src_ce, np_trg_ce))
    input_set = input_set.batch(batch_size)
    input_set = input_set.prefetch(tf.data.experimental.AUTOTUNE)

    loss_obj = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    loss_met = tf.keras.metrics.Mean()

    l_trial_loss_vals = []
    for i in range(n_trials):
        loss_met.reset_states()
        for src_ce, trg_ce in input_set:
            pred_trg_ce = tf.convert_to_tensor(np.random.rand(1, n_clusters), dtype=tf.float32)
            pred_trg_ce = tf.nn.softmax(pred_trg_ce)
            loss = loss_obj(trg_ce, pred_trg_ce)
            loss_met(loss)
        loss_val = loss_met.result().numpy()
        l_trial_loss_vals.append(loss_val)
        logging.debug('[random_baseline] Trial %s: loss = %s' % (i, loss_val))
    logging.debug('[random_baseline] Trial losses = %s' % str(l_trial_loss_vals))
    logging.debug('[random_baseline] All trials done. mean = %s, std = %s'
                  % (np.mean(l_trial_loss_vals), np.std(l_trial_loss_vals)))


def trans_mat_baseline(ts_name, n_clusters, n_trials):
    import tensorflow as tf
    logging.debug('[trans_mat_baseline] Starts with %s...' % ts_name)
    timer_start = time.time()
    batch_size = 1

    df_input_set = pd.read_pickle(global_settings.g_tw_usr_beliefs_train_set_file_format.format(str(ts_name)))
    np_src_ce = df_input_set['src_ce'].values.tolist()
    np_trg_ce = df_input_set['trg_ce'].values.tolist()
    input_set = tf.data.Dataset.from_tensor_slices((np_src_ce, np_trg_ce))
    input_set = input_set.batch(batch_size)
    input_set = input_set.prefetch(tf.data.experimental.AUTOTUNE)

    loss_obj = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    loss_met = tf.keras.metrics.Mean()

    trans_mat = np.load(global_settings.g_tw_phrases_response_graph_transition_mat_format.format(str(n_clusters)))
    l_trial_loss_vals = []
    for i in range(n_trials):
        loss_met.reset_states()
        for src_ce, trg_ce in input_set:
            # src_ce = tf.nn.softmax(src_ce)
            pred_trg_ce = tf.transpose(tf.matmul(trans_mat, tf.transpose(src_ce)))
            pred_trg_ce = tf.nn.softmax(pred_trg_ce)
            loss = loss_obj(trg_ce, pred_trg_ce)
            loss_met(loss)
        loss_val = loss_met.result().numpy()
        l_trial_loss_vals.append(loss_val)
        logging.debug('[trans_mat_baseline] Trial %s: loss = %s' % (i, loss_val))
    logging.debug('[trans_mat_baseline] Trial losses = %s' % str(l_trial_loss_vals))
    logging.debug('[trans_mat_baseline] All trials done. mean = %s, std = %s'
                  % (np.mean(l_trial_loss_vals), np.std(l_trial_loss_vals)))




'''
Autoencoder Embedding Compression
'''
g_lexvec_model = None
g_embedding_len = 300


def make_phrase_training_sets():
    if not path.exists(global_settings.g_tw_raw_phrases_ae_training_sets_folder):
        os.mkdir(global_settings.g_tw_raw_phrases_ae_training_sets_folder)
    timer_start = time.time()
    l_batch = []
    batch_cnt = 0
    with open(global_settings.g_tw_raw_phrases_output, 'r') as in_fd:
        for ln in in_fd:
            fields = [item.strip() for item in ln.split('|')]
            phrase = ' '.join([token.strip() for token in fields[0].split(',')])
            embed = phrase_embedding(phrase)
            embed_str = ','.join([str(ele) for ele in embed])
            l_batch.append(phrase + '|' + embed_str)
            if len(l_batch) % 20000 == 0 and len(l_batch) >= 20000:
                with open(global_settings.g_tw_raw_phrases_ae_training_sets_file_format.format(batch_cnt), 'w+') \
                        as out_fd:
                    out_str = '\n'.join(l_batch)
                    out_fd.write(out_str)
                    out_fd.close()
                    logging.debug('[make_training_sets] Training set #%s is done in %s secs.'
                                  % (batch_cnt, time.time() - timer_start))
                batch_cnt += 1
                l_batch = []
        if len(l_batch) > 0:
            with open(global_settings.g_tw_raw_phrases_ae_training_sets_file_format.format(batch_cnt), 'w+') as out_fd:
                out_str = '\n'.join(l_batch)
                out_fd.write(out_str)
                out_fd.close()
                logging.debug('[make_training_sets] Training set #%s is done in %s secs.'
                              % (batch_cnt, time.time() - timer_start))
    logging.debug('[make_training_sets] All done.')


def make_token_training_sets():
    if not path.exists(global_settings.g_tw_raw_phrases_token_ae_training_sets_folder):
        os.mkdir(global_settings.g_tw_raw_phrases_token_ae_training_sets_folder)
    timer_start = time.time()
    l_batch = []
    batch_cnt = 0
    with open(global_settings.g_tw_raw_phrases_vocab_file, 'r') as in_fd:
        for ln in in_fd:
            token = ln.strip()
            embed = phrase_embedding(token)
            embed_str = ','.join([str(ele) for ele in embed])
            l_batch.append(token + '|' + embed_str)
            if len(l_batch) % 20000 == 0 and len(l_batch) >= 20000:
                with open(global_settings.g_tw_raw_phrases_token_ae_training_sets_file_format.format(batch_cnt), 'w+') \
                        as out_fd:
                    out_str = '\n'.join(l_batch)
                    out_fd.write(out_str)
                    out_fd.close()
                    logging.debug('[make_token_training_sets] Training set #%s is done in %s secs.'
                                  % (batch_cnt, time.time() - timer_start))
                batch_cnt += 1
                l_batch = []
        if len(l_batch) > 0:
            with open(global_settings.g_tw_raw_phrases_token_ae_training_sets_file_format.format(batch_cnt), 'w+') as out_fd:
                out_str = '\n'.join(l_batch)
                out_fd.write(out_str)
                out_fd.close()
                logging.debug('[make_token_training_sets] Training set #%s is done in %s secs.'
                              % (batch_cnt, time.time() - timer_start))
    logging.debug('[make_token_training_sets] All done.')


def ae_embed_compress(enc_dense_1_d, enc_dense_2_d, enc_dense_3_d, dec_dense_1_d, dec_dense_2_d):
    import tensorflow as tf

    class Encoder(tf.keras.layers.Layer):
        def __init__(self, ed1_d, ed2_d, ed3_d):
            super(Encoder, self).__init__()
            self.enc_dense_1 = tf.keras.layers.Dense(ed1_d, activation='selu')
            # self.enc_dense_2 = tf.keras.layers.Dense(ed2_d, activation='selu')
            # self.enc_dense_3 = tf.keras.layers.Dense(ed3_d, activation='selu')

        def call(self, input_vec):
            # n_input_vec = tf.keras.utils.normalize(input_vec)
            ed_1_out = self.enc_dense_1(input_vec)
            # ed_2_out = self.enc_dense_2(ed_1_out)
            # ed_3_out = self.enc_dense_3(ed_2_out)
            return ed_1_out

    class Decoder(tf.keras.layers.Layer):
        def __init__(self, dd1_d, dd2_d):
            super(Decoder, self).__init__()
            self.dec_dense_1 = tf.keras.layers.Dense(dd1_d, activation='sigmoid')
            # self.dec_dense_2 = tf.keras.layers.Dense(dd2_d, activation='sigmoid')

        def call(self, int_vec):
            dd_1_out = self.dec_dense_1(int_vec)
            # dd_2_out = self.dec_dense_2(int_vec)
            # dd_2_out = tf.keras.utils.normalize(dd_2_out)
            return dd_1_out, int_vec

    class Embed_AE(tf.keras.Model):
        def __init__(self, ed1_d, ed2_d, ed3_d, dd1_d, dd2_d):
            super(Embed_AE, self).__init__()
            self.enc_layer = Encoder(ed1_d=ed1_d, ed2_d=ed2_d, ed3_d=ed3_d)
            self.dec_layer = Decoder(dd1_d=dd1_d, dd2_d=dd2_d)

        def call(self, input_vecs):
            enc_tensor = self.enc_layer(input_vecs)
            dec_tensor, int_tensor = self.dec_layer(enc_tensor)
            return dec_tensor, int_tensor

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    # TODO
    # [Strategy modification]
    def load_one_token_train_set(ts_id, batch_size, strategy):
        l_embeds = []
        with open(global_settings.g_tw_raw_phrases_token_ae_training_sets_file_format.format(str(ts_id)), 'r') as in_fd:
            for ln in in_fd:
                fields = [item.strip() for item in ln.split('|')]
                embed = np.asarray([float(ele) for ele in fields[1].split(',')]).astype('float32')
                l_embeds.append(embed)
            in_fd.close()
        train_set = tf.data.Dataset.from_tensor_slices(l_embeds)
        train_set = train_set.shuffle(len(l_embeds))
        train_set = train_set.batch(batch_size)
        train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
        # TODO
        # [Strategy modification]
        train_set = strategy.experimental_distribute_dataset(train_set)
        return train_set

    @tf.function
    def train_step(input_vecs, embed_ae, optimizer, loss_obj, loss_obj_2):
        with tf.GradientTape() as gt:
            dec_tensor, int_tensor = embed_ae(input_vecs)
            loss = loss_obj(input_vecs, dec_tensor)
            loss = tf.reduce_mean(loss)
            loss_2 = loss_obj_2(input_vecs, dec_tensor)
            loss_2 = tf.reduce_mean(loss_2)
            grads = gt.gradient(loss, embed_ae.trainable_variables)
            optimizer.apply_gradients(zip(grads, embed_ae.trainable_variables))
        return loss, loss_2

    # @tf.function
    def train_distributed(embed_ae, epoch_cnt, loss_obj, loss_met, batch_size, train_set_cnt, loss_obj_2, loss_met_2):
        # TODO
        # [Strategy modification]
        # [WARNING!] strategy has a very significant overhead when running on Rivanna.
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            for epoch in range(epoch_cnt):
                loss_met.reset_states()
                loss_met_2.reset_states()
                for train_set_id in range(train_set_cnt):
                    # for train_set_id in ['full']:
                    # train_set = load_one_token_train_set(train_set_id, batch_size, strategy)
                    train_set = load_one_token_train_set(train_set_id, batch_size, None)
                    for batch_idx, input_vecs in enumerate(train_set):
                        # TODO
                        # [Strategy modification]
                        loss, loss_2 = strategy.run(train_step, args=(input_vecs, embed_ae, opt_ins, loss_obj, loss_obj_2))
                        # loss, loss_2 = train_step(input_vecs, embed_ae, opt_ins, loss_obj, loss_obj_2)
                        loss_met(loss)
                        loss_met_2(loss_2)
                        if batch_idx % 100 == 0 and batch_idx >= 100:
                            print('[ae_embed_compress] Epoch:%s Set:%s Batch:%s Loss:%s Loss_2:%s Time:%s'
                                  % (epoch, train_set_id, batch_idx, loss_met.result().numpy(), loss_met_2.result().numpy(), time.time() - timer_start))
                print('[ae_embed_compress] Epoch:%s done. Loss:%s Loss_2:%s Time:%s' % (epoch, loss_met.result().numpy(), loss_met_2.result().numpy(), time.time() - timer_start))
            print('[ae_embed_compress] All done. Loss:%s Loss_2:%s Time:%s' % (loss_met.result().numpy(), loss_met_2.result().numpy(), time.time() - timer_start))

    '''Function starts here'''
    embed_ae_ins = Embed_AE(enc_dense_1_d, enc_dense_2_d, enc_dense_3_d, dec_dense_1_d, dec_dense_2_d)
    learn_rate = CustomSchedule(enc_dense_1_d)
    opt_ins = tf.keras.optimizers.Adam(learn_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    # opt_ins = tf.keras.optimizers.Adamax(learn_rate)
    batch_size = 1
    train_set_cnt = 14
    epoch_cnt = 10

    timer_start = time.time()
    loss_obj = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
    # loss_obj = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss_met = tf.keras.metrics.Mean()
    loss_obj_2 = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
    loss_met_2 = tf.keras.metrics.Mean()

    train_distributed(embed_ae_ins, epoch_cnt, loss_obj, loss_met, batch_size, train_set_cnt, loss_obj_2, loss_met_2)

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     for epoch in range(epoch_cnt):
    #         loss_met.reset_states()
    #         for train_set_id in range(train_set_cnt):
    #         # for train_set_id in ['full']:
    #             train_set = load_one_token_train_set(train_set_id, batch_size, strategy)
    #             for batch_idx, input_vecs in enumerate(train_set):
    #                 loss = strategy.run(train_step, args=(input_vecs, embed_ae_ins, opt_ins, loss_obj))
    #                 # loss = train_step(input_vecs, embed_ae_ins, opt_ins, loss_obj)
    #                 loss_met(loss)
    #                 if batch_idx % 100 == 0 and batch_idx >= 100:
    #                     print('[ae_embed_compress] Epoch:%s Set:%s Batch:%s Loss:%s Time:%s'
    #                           % (epoch, train_set_id, batch_idx, loss_met.result().numpy(), time.time() - timer_start))
    #         print('[ae_embed_compress] Epoch:%s done. Loss:%s Time:%s' % (epoch, loss_met.result().numpy(), time.time() - timer_start))
    #     print('[ae_embed_compress] All done. Loss:%s Time:%s' % (loss_met.result().numpy(), time.time() - timer_start))


'''
TEST ONLY STARTS
'''
def verify_phrase_embeds_and_indexes():
    logging.debug('[verify_phrase_embeds_and_indexes] Starts...')
    phrase_embeds = pd.read_pickle(global_settings.g_tw_raw_phrases_embeds)
    logging.debug('[verify_phrase_embeds_and_indexes] Load in g_tw_raw_phrases_embeds.')
    phrase_embeds_np = phrase_embeds.to_numpy()
    logging.debug('[verify_phrase_embeds_and_indexes] convert phrase_embeds to phrase_embeds_np')
    with open(global_settings.g_tw_raw_phrases_id_to_phrase, 'r') as in_fd:
        d_id_to_phrase = json.load(in_fd)
        in_fd.close()
    logging.debug('[verify_phrase_embeds_and_indexes] Load in g_tw_raw_phrases_id_to_phrase. type = %s' % type(d_id_to_phrase))
    cnt = 0
    for key in d_id_to_phrase:
        print(key, d_id_to_phrase[key])
        cnt += 1
        if cnt > 5:
            break
    load_lexvec_model()
    logging.debug('[verify_phrase_embeds_and_indexes] Load in lexvec')
    cnt = 0
    for rec in phrase_embeds_np:
        phrase_id = rec[0]
        fetched_embed = np.asarray(rec[1], dtype=np.float32)
        print('phrase_id = %s' % phrase_id)
        phrase = d_id_to_phrase[phrase_id]
        computed_embed = phrase_embedding(phrase)
        computed_embed = preprocessing.normalize(computed_embed.reshape(1, -1))
        cos = cosine(computed_embed, fetched_embed)
        logging.debug('[verify_phrase_embeds_and_indexes] phrase = %s, cos = %s' % (phrase, cos))
        print('fetched_embed = ')
        print(fetched_embed)
        print('computed_embed = ')
        print(computed_embed)
        cnt += 1
        if cnt > 10:
            return


def verify_phrase_cluster_embeds(n_cluster):
    logging.debug('[verify_phrase_cluster_embeds] Starts...')
    timer_start = time.time()
    np_phrase_cluster_embeds = np.load(global_settings.g_tw_raw_phrases_cluster_space_embeds_format.format(str(n_cluster)))
    logging.debug('[verify_phrase_cluster_embeds] Load in g_tw_raw_phrases_cluster_space_embeds in %s secs.'
                  % str(time.time() - timer_start))
    for i in range(10):
        embed_idx = random.randint(0, 4216735)
        embed = np_phrase_cluster_embeds[embed_idx]
        sum = np.sum(embed)
        logging.debug('[verify_phrase_cluster_embeds] id: %s, sum = %s' % (embed_idx, sum))


def verify_src_to_trg_tw_id_pairs_phrases_by_idx():
    with open(global_settings.g_tw_src_trg_tw_id_pairs, 'r') as in_fd:
        d_src_trg_tw_id_pairs = json.load(in_fd)
        in_fd.close()
    with open(global_settings.g_tw_phrases_by_idx, 'r') as in_fd:
        d_phrase_idx = json.load(in_fd)
        in_fd.close()
    l_valid = []
    l_invalid = []
    for src_id in d_src_trg_tw_id_pairs:
        if src_id not in d_phrase_idx:
            l_invalid.append(src_id)
            print('src = %s' % src_id)
        else:
            l_trg_ids = d_src_trg_tw_id_pairs[src_id]
            for trg_id, trg_datetime in l_trg_ids:
                if trg_id not in d_phrase_idx:
                    l_invalid.append(src_id)
                    print('trg = %s' % trg_id)
                else:
                    l_valid.append([src_id, trg_id])
    print(len(l_valid), len(l_invalid))


def verify_phrase_embeds_to_cluster_space(n_clusters):
    logging.debug('[verify_phrase_embeds_to_cluster_space] Starts...')
    timer_start = time.time()

    np_raw_phrase_embeds = np.load(global_settings.g_tw_raw_phrases_we)
    logging.debug('[verify_phrase_embeds_to_cluster_space] Load in g_tw_raw_phrases_we')

    with open(global_settings.g_tw_raw_phrases_clustering_centers_format.format(str(n_clusters)), 'r') as in_fd:
        d_cluster_centers = json.load(in_fd)
        in_fd.close()
    logging.debug('[verify_phrase_embeds_to_cluster_space] Load in cluster centers')

    d_weid2label = dict()
    with open(global_settings.g_tw_raw_phrases_clustering_labels_format.format(str(n_clusters)), 'r') as in_fd:
        for ln_idx, ln in enumerate(in_fd):
            d_weid2label[ln_idx] = ln.strip()
        in_fd.close()

    l_phrase_cluster_embeds = []
    for i in range(100):
        rand_id = random.randint(0, len(np_raw_phrase_embeds) - 1)
        phrase_we = np.asarray(np_raw_phrase_embeds[rand_id], dtype=np.float32)
        phrase_cluster_embed = np.zeros(n_clusters)
        cluster_label = d_weid2label[rand_id]

        for k in range(n_clusters):
            cluster_center_vec = np.asarray(d_cluster_centers[str(k)], dtype=np.float32)
            sim = 1.0 - cosine(phrase_we, cluster_center_vec)
            if not np.isfinite(sim):
                raise Exception('[verify_phrase_embeds_to_cluster_space] Invalid sim @ phrase_id = %s, cluster_id = %s'
                                % (rand_id, k))
            if k == int(cluster_label) or sim >= 0.4:
                phrase_cluster_embed[k] = sim

        print('cluster space sim embed %s:' % rand_id)
        print(phrase_cluster_embed)

        avg_embed = phrase_cluster_embed / np.sum(phrase_cluster_embed)
        print('cluster space avg embed %s:' % rand_id)
        print(avg_embed)

        phrase_cluster_embed = np.asarray(softmax(avg_embed * 10), dtype=np.float32)
        print('cluster space softmax embed %s:' % rand_id)
        print(phrase_cluster_embed)
        max_ele_id = phrase_cluster_embed.tolist().index(max(phrase_cluster_embed))

        print('cluster_label = %s, max_ele_idx = %s' % (cluster_label, max_ele_id))
        print('\n')
        l_phrase_cluster_embeds.append(phrase_cluster_embed)

    print()


def test_usr_beliefs_train_set_per_usr():
    with open(global_settings.g_tw_replies_quotes_per_user, 'r') as in_fd:
        d_rq_per_usr = json.load(in_fd)
        in_fd.close()
    with open(global_settings.g_tw_src_trg_tw_id_pairs, 'r') as in_fd:
        d_src_trg = json.load(in_fd)
        in_fd.close()

    d_trg_src = dict()
    for src_tw_id in d_src_trg:
        l_trg_item = d_src_trg[src_tw_id]
        for trg_tw_id, trg_datetime in l_trg_item:
            d_trg_src[trg_tw_id] = src_tw_id

    l_top_usr = sorted(d_rq_per_usr, key=lambda k: len(d_rq_per_usr[k]), reverse=True)[:10]
    for usr_id in l_top_usr:
        l_src_id = []
        for trg_id in d_rq_per_usr[usr_id]:
            if trg_id in d_trg_src:
                l_src_id.append(d_trg_src[trg_id])
        make_user_beliefs_train_sets(global_settings.g_num_phrase_clusters,
                                     l_src_id,
                                     None,
                                     '20190131235959',
                                     usr_id)
        logging.debug('[test_usr_beliefs_train_set_per_usr] usr %s: %s.' % (usr_id, len(l_src_id)))
    logging.debug('[test_usr_beliefs_train_set_per_usr] All done.')


def test_kl_with_rand_vecs():
    n_dim = 150
    l_dists = []
    for i in range(n_dim):
        mu = random.random()
        sigma = random.random()
        l_dists.append((mu, sigma))

    n_sampels = 100
    l_rand_vecs = []
    for i in range(n_sampels):
        rand_vec = np.zeros(n_dim, dtype=np.float32)
        for k in range(n_dim):
            mu, sigma = l_dists[k]
            rand_ele = np.random.normal(mu, sigma, 1)
            rand_vec[k] = rand_ele
        rand_vec = softmax(rand_vec)
        l_rand_vecs.append(rand_vec.astype(dtype=np.float32))

    import tensorflow as tf
    loss_obj = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    l_loss = []
    for i in range(len(l_rand_vecs) - 1):
        vec_1 = l_rand_vecs[i]
        for j in range(i + 1, len(l_rand_vecs)):
            vec_2 = l_rand_vecs[j]
            loss = loss_obj(vec_1, vec_2)
            l_loss.append(loss)

    print()


def test_phrase_onehot(n_cluster, num_trials):
    logging.debug('[test_phrase_onehot] Starts...')

    df_tw_phid = pd.read_pickle(global_settings.g_tw_phrase_extraction_tw_to_phids_file)
    logging.debug('[test_phrase_onehot] Load g_tw_phrase_extraction_tw_to_phids_file with %s tw phid maps.'
                  % len(df_tw_phid))

    df_onehot = pd.read_pickle(global_settings.g_tw_phrase_cluster_onehot_embeds_all_file_format.format(str(n_cluster)))
    logging.debug('[test_phrase_onehot] Load g_tw_phrase_cluster_onehot_embeds_all_file with %s phrase onehot embeds'
                  % str(len(df_onehot)))

    for k in range(num_trials):
        l_tw_phid_sample = random.choices(df_tw_phid['raw_phs'].values, k=1000)
        l_tw_onehot = []
        for l_tw_phid in l_tw_phid_sample:
            tw_onehot = np.zeros(n_cluster, dtype=np.float32)
            for tw_phid in l_tw_phid:
                ph_onehot = df_onehot.loc[df_onehot['phid'] == int(tw_phid)]['onehot'].to_list()
                ph_onehot = np.asarray(ph_onehot[0], dtype=np.float32)
                tw_onehot += ph_onehot
            l_tw_onehot.append(tw_onehot)
        np_tw_onehot = np.stack(l_tw_onehot)
        np_tw_onehot = preprocessing.normalize(np_tw_onehot)
        logging.debug('[test_phrase_onehot] Trial %s: Get %s tw onehot embeds.' % (k, str(len(np_tw_onehot))))

        tw_sims = np.matmul(np_tw_onehot, np.transpose(np_tw_onehot))
        l_tw_sims = []
        for i in range(len(tw_sims) - 1):
            for j in range(i + 1, len(tw_sims)):
                l_tw_sims.append(tw_sims[i][j])
        logging.debug('[test_phrase_onehot] Trial %s: avg sim = %s, std = %s'
                      % (k, np.mean(l_tw_sims), np.std(l_tw_sims)))

    logging.debug('[test_phrase_onehot] All done.')



'''
TEST ONLY ENDS
'''



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1].strip()
    '''Make raw phrase extraction jobs for different nodes'''
    # make_phrase_extraction_jobs(global_settings.g_tw_raw_phrase_job_num)
    '''Extract raw phrases'''
    # extract_phrases_from_sem_units()
    # read_phrases_into_one_file()
    # phrases_type_encode()
    # extract_vocab_from_phrases()

    if cmd == 'phrase_extraction':
        tb_name = sys.argv[2]
        num_threads = sys.argv[3]
        job_id = sys.argv[4]
        extract_all_phrases_and_phrases_for_each_tw_multithread(tb_name, num_threads, job_id)
        extract_all_phrases_and_phrases_for_each_tw_int_to_out()

        # make_phrase_extraction_jobs(global_settings.g_tw_raw_phrase_job_num)
        # d_sem_units = load_sem_units()
        # extract_src_trg_tw_pairs_for_replies_quotes(0, 12537978, d_sem_units)
        # with open(global_settings.g_tw_raw_phrases_phrase_to_id, 'r') as in_fd:
        #     d_phrase_idx = json.load(in_fd)
        #     in_fd.close()
        # s_tw_ids = extract_tw_ids_from_srg_trg_pairs_for_replies_quotes()
        # extract_phrases_for_each_tw(0, 2067468, s_tw_ids, d_phrase_idx)
        # replies_quotes_per_user()
        # extract_unique_phrases()
        # id_phrases()
        # verify_src_to_trg_tw_id_pairs_phrases_by_idx()

    elif cmd == 'phrase_clustering':
        # num_procs = sys.argv[2]
        # job_id = sys.argv[3]
        # phrase_embeds_multiproc(num_procs, job_id)
        # phrase_embeds_int_to_out()
        # build_phrase_clustering_dataset()
        # phrase_clustering(global_settings.g_num_phrase_clusters)
        # compute_phrase_cluster_info(global_settings.g_num_phrase_clusters)
        # phrase_clustering_stats(global_settings.g_num_phrase_clusters)
        # classify_phrase_embeds_to_cluster_space_multiproc(global_settings.g_num_phrase_clusters, num_procs, job_id)
        # phrase_cluster_embeds_int_to_out(global_settings.g_num_phrase_clusters)
        phrase_cluster_label_onehot_embed(global_settings.g_num_phrase_clusters)

        # l_phrases = make_phrase_embeds_tasks()
        # job_id = 0
        # if len(sys.argv) > 1:
        #     job_id = sys.argv[1]
        # get_phrase_embeds_multithread(l_phrases, get_phrase_embeds_single_thread, 20, job_id)
        # merge_raw_phrase_embeds()
        # convert_raw_phrase_embed_strs_to_vecs()
        # build_reverse_phrase_index()
        # convert_tw_phrases_to_clusters()
        # verify_phrase_embeds_and_indexes()
        # build_phrase_clustering_dataset()
        # phrase_clustering(global_settings.g_num_phrase_clusters)
        # compute_phrase_cluster_centers(global_settings.g_num_phrase_clusters)
        # phrase_clustering_stats(global_settings.g_num_phrase_clusters)
        # classify_phrase_embeds_to_cluster_space_multithread(global_settings.g_num_phrase_clusters, 2)
        # merge_phrase_embeds_to_cluster_space(global_settings.g_num_phrase_clusters)
        # verify_phrase_cluster_embeds(global_settings.g_num_phrase_clusters)
        # verify_phrase_embeds_to_cluster_space(150)

    elif cmd == 'tw_pc_onehot':
        # num_procs = sys.argv[2]
        # job_id = sys.argv[3]
        # tw_to_onehot_vec_multiproc(num_procs,
        #                            global_settings.g_num_phrase_clusters,
        #                            job_id)
        tw_to_onehot_vec_int_to_out(global_settings.g_num_phrase_clusters)

    elif cmd == 'response_graph':
        dt_start = '20181224000000'
        dt_end = '20190131235959'
        com_id = '1404'
        # com_id = 'SFvnX8bgjNO8ft0XQ8Tq5g'
        # df_udt_com = pd.read_pickle(global_settings.g_udt_com_data_file_format.format(com_id))
        # l_usr_ids = df_udt_com['usr_id'].values.tolist()
        # l_usr_ids = ['SFvnX8bgjNO8ft0XQ8Tq5g']
        # l_tw_types = ['r', 'q']
        # extract_src_trg_tw_data(l_usr_ids, l_tw_types, dt_start, dt_end, com_id)
        # num_procs = sys.argv[2]
        # job_id = sys.argv[3]
        task_name = '1404_r#q_20181224000000#20190131235959'
        # build_response_graph_multiproc(task_name, global_settings.g_num_phrase_clusters, num_procs, job_id)
        # resp_graph_int_to_trans_mat(task_name)
        # draw_trans_mat_heatmap(task_name)
        build_comprehensive_src_trg_data(task_name)

        # make_response_graph_tasks(1)
        # task_path = sys.argv[2]
        # job_id = sys.argv[3]
        # dt_start = '20181224000000'
        # dt_start = None
        # dt_end = '20190131235959'
        # build_response_graph_multithread(task_path,
        #                                  dt_start,
        #                                  dt_end,
        #                                  global_settings.g_num_phrase_clusters,
        #                                  1,
        #                                  job_id)
        # merge_resp_graph(global_settings.g_num_phrase_clusters)
        # get_transition_mat_from_resp_graph(global_settings.g_num_phrase_clusters)

    elif cmd == 'user_beliefs':
        train_set_id = '1404_r#q_20181224000000#20190131235959_150'
        # make_user_beliefs_train_sets(train_set_id)
        user_beliefs_model(train_set_id,
                             global_settings.g_num_phrase_clusters,
                             global_settings.g_num_phrase_clusters,
                             global_settings.g_num_phrase_clusters,
                             True,
                             None,
                             train_set_id)
        # random_baseline(train_set_id, global_settings.g_num_phrase_clusters, 10)
        # trans_mat_baseline(train_set_id, global_settings.g_num_phrase_clusters, 10)


        # with open(global_settings.g_tw_src_trg_tw_id_pairs, 'r') as in_fd:
        #     d_src_trg_tws = json.load(in_fd)
        #     in_fd.close()
        # l_src_tw_ids = list(d_src_trg_tws.keys())
        # dt_start = None
        # dt_end = '20190131235959'
        # train_set_id = 'all'
        # make_user_beliefs_train_sets(global_settings.g_num_phrase_clusters, l_src_tw_ids, dt_start, dt_end, train_set_id)
        # build_user_beliefs_model(global_settings.g_num_phrase_clusters,
        #                          global_settings.g_num_phrase_clusters,
        #                          global_settings.g_num_phrase_clusters)
        # test_usr_beliefs_train_set_per_usr()

    elif cmd == 'resp_infer':
        train_set_id = '1404_r#q_20181224000000#20190131235959_150'
        test_src_ce = [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0.1, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0. , 0. ,
                       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                       0.1, 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                       0. , 0. , 0. , 0. , 0. , 0. , 0. ]
        test_src_ce = np.asarray(test_src_ce, dtype=np.float32)
        test_src_ce = test_src_ce.reshape(1, -1)
        pred_trg_ce = user_beliefs_model(train_set_id,
                                           global_settings.g_num_phrase_clusters,
                                           global_settings.g_num_phrase_clusters,
                                           global_settings.g_num_phrase_clusters,
                                           False,
                                           test_src_ce,
                                           train_set_id)
        print(pred_trg_ce)

    elif cmd == 'ae_compress':
        # load_lexvec_model()
        # make_token_training_sets()
        # make_training_sets()

        enc_dense_1_d = 150
        enc_dense_2_d = 150
        enc_dense_3_d = 100
        dec_dense_1_d = g_embedding_len
        dec_dense_2_d = g_embedding_len
        ae_embed_compress(enc_dense_1_d, enc_dense_2_d, enc_dense_3_d, dec_dense_1_d, dec_dense_2_d)

    elif cmd == 'test':
        # test_kl_with_rand_vecs()
        # test_cls_json_str = '''{"directed": false, "multigraph": false, "graph": [], "nodes": [{"txt": "know", "pos": "VERB", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|know"}, {"txt": "observe", "pos": "VERB", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|observe"}, {"txt": "VenezuelaGritaLiberty", "pos": "PROPN", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|VenezuelaGritaLiberty"}, {"txt": "spectator", "pos": "NOUN", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|spectator"}, {"txt": "rest", "pos": "NOUN", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|rest"}, {"txt": "rebel guard", "pos": "NOUN", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|rebel guard"}, {"txt": "identify", "pos": "VERB", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|identify"}, {"txt": "not", "pos": "PART", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|not"}, {"txt": "video", "pos": "NOUN", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|video"}, {"txt": "relevant", "pos": "ADJ", "id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|relevant"}], "adjacency": [[{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|VenezuelaGritaLiberty"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|observe"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|video"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|relevant"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|not"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|know"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|spectator"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|rest"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|rebel guard"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|identify"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|know"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|observe"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|observe"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|observe"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|observe"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|not"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|know"}, {"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|identify"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|know"}], [{"id": "FqP-4EdM0PO7ixf8x0FpmA|27901|0|know"}]]}'''
        # extract_phrases_from_cls_json_str(test_cls_json_str)
        test_phrase_onehot(global_settings.g_num_phrase_clusters, 10)