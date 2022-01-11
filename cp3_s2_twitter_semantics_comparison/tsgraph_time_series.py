import networkx as nx
import logging
# import sent2vec
import scipy.spatial.distance as scipyd
# import json
import multiprocessing
import threading
import math
# from sklearn import metrics
import sqlite3
import time
import os.path
from os import path
# import numpy as np
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_time_series_data_txt_by_uid_db_format = g_time_series_data_path_prefix + '{0}/{1}_txt_by_uid.db'
g_tsgraph_format = g_time_series_data_path_prefix + '{0}/{1}_tsgraph.gml'


def build_and_write_one_tsgraph(tsgraph, time_int):
    timer_start = time.time()
    time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
    print('Build tsgraph for %s' % time_int_str)
    try:
        print('Try to open %s' % g_time_series_data_txt_by_uid_db_format.format(time_int_str, time_int_str))
        txt_db_conn = sqlite3.connect(g_time_series_data_txt_by_uid_db_format.format(time_int_str, time_int_str))
    except Exception as e:
        print('Open %s DB met some trouble.' % time_int_str)
        print(e)
        os._exit(0)
    txt_db_cur = txt_db_conn.cursor()
    sql_str = '''SELECT uid, raw_cln_lexvec from wh_txt_by_uid'''
    txt_db_cur.execute(sql_str)
    l_recs = txt_db_cur.fetchall()
    d_uid_raw_cln_vecs = dict()
    for rec in l_recs:
        uid = rec[0]
        raw_cln_vec = [float(ele.strip()) for ele in rec[1].split(',')]
        tsgraph.add_node(uid)
        d_uid_raw_cln_vecs[uid] = raw_cln_vec
    txt_db_conn.close()

    l_nodes = list(tsgraph.nodes)
    for i in range(0, len(l_nodes)-1):
        for j in range(i+1, len(l_nodes)):
            if tsgraph.has_edge(l_nodes[i], l_nodes[j]):
                logging.error('(%s, %s) has existed in %s tsgraph.' % (l_nodes[i], l_nodes[j], time_int_str))
            else:
                try:
                    if len(d_uid_raw_cln_vecs[l_nodes[i]]) == 0 or len(d_uid_raw_cln_vecs[l_nodes[j]]) == 0:
                        sim = 0.0
                    else:
                        sim = 1.0 - scipyd.cosine(d_uid_raw_cln_vecs[l_nodes[i]], d_uid_raw_cln_vecs[l_nodes[j]])
                except:
                    sim = 0.0
                if str(sim).lower() == 'nan':
                    sim = 0.0
                tsgraph.add_edge(l_nodes[i], l_nodes[j], weight=(sim + 1.0) / 2.0)

    nx.write_gml(tsgraph, g_tsgraph_format.format(time_int_str, time_int_str))
    logging.debug('%s tsgraph is done in %s secs. Graph info: %s' % \
                  (time_int_str, str(time.time()-timer_start), nx.info(tsgraph)))
    # return tsgraph


def build_and_write_tsgraphs(l_time_ints):
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        if path.exists(g_tsgraph_format.format(time_int_str, time_int_str)):
            continue
        tsgraph = nx.Graph()
        build_and_write_one_tsgraph(tsgraph, time_int)


def build_tsgraphs_multithreads():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    l_time_ints_av = []
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        if not path.exists(g_tsgraph_format.format(time_int_str, time_int_str)):
            l_time_ints_av.append(time_int)
    logging.debug('%s jobs in total.' % len(l_time_ints_av))
    batch_size = math.ceil(len(l_time_ints_av) / multiprocessing.cpu_count())
    l_l_time_ints = []
    for i in range(0, len(l_time_ints_av), batch_size):
        if i + batch_size < len(l_time_ints_av):
            l_l_time_ints.append(l_time_ints_av[i:i + batch_size])
        else:
            l_l_time_ints.append(l_time_ints_av[i:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_time_ints:
        t = threading.Thread(target=build_and_write_tsgraphs, args=(l_each_batch,))
        t.setName('tsgraph_t_' + str(t_id))
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

    logging.debug('All tsgraphs are done.')


# def add_nodes_to_one_tsgraph(tsgraph, time_int):
#     time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
#     txt_db_conn = sqlite3.connect(g_time_series_data_txt_by_uid_db_format.format(time_int_str, time_int_str))
#     txt_db_cur = txt_db_conn.cursor()
#     sql_str = '''SELECT uid from wh_txt_by_uid'''
#     txt_db_cur.execute(sql_str)
#     l_uids = [rec[0] for rec in txt_db_cur.fetchall()]
#     tsgraph.add_nodes_from(l_uids)
#     txt_db_conn.close()
#     return tsgraph


def main():
    build_tsgraphs_multithreads()
    # l_time_ints = data_preprocessing_utils.read_time_ints()
    # build_and_write_tsgraphs(l_time_ints)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()