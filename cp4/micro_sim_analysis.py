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
from datetime import datetime, timedelta
import re

import csv
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
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.mplot3d.art3d as art3d
import scipy.stats as stats
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import normalize
import psycopg2
from scipy.spatial.distance import jensenshannon

import global_settings


def build_out_to_in_msg_prop_graphs_by_com(com_id):
    logging.debug('[build_out_to_in_msg_prop_graphs_by_com] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = """select 
                    src_tw_id, 
                    src_usr_id, 
                    src_usr_sna_com_id, 
                    src_tw_type, 
                    src_tw_datetime, 
                    src_l_nars, 
                    src_l_stances, 
                    trg_tw_id, 
                    trg_usr_id, 
                    trg_tw_type, 
                    trg_tw_datetime,
                    trg_l_nars,
                    trg_l_stances
                from cp4.mf3jh_udt_tw_src_trg_sna_data 
                where trg_usr_sna_com_id = '{0}' and src_usr_sna_com_id != '{1}' and src_l_nars is not null""" \
        .format(str(com_id), str(com_id))
    tw_db_cur.execute(sql_str)
    l_out_to_in_recs = tw_db_cur.fetchall()
    logging.debug('[build_out_to_in_msg_prop_graphs_by_com] Fetch %s out_to_in_recs for %s in %s secs.'
                  % (len(l_out_to_in_recs), com_id, time.time() - timer_start))

    sql_str = """select 
                    src_tw_id, 
                    src_usr_id, 
                    src_tw_type, 
                    src_tw_datetime, 
                    src_l_nars, 
                    src_l_stances, 
                    trg_tw_id, 
                    trg_usr_id, 
                    trg_tw_type, 
                    trg_tw_datetime,
                    trg_l_nars, 
                    trg_l_stances
                from cp4.mf3jh_udt_tw_src_trg_sna_data 
                where trg_usr_sna_com_id = '{0}' and src_usr_sna_com_id = '{1}'""" \
        .format(str(com_id), str(com_id))
    tw_db_cur.execute(sql_str)
    l_in_to_in_recs = tw_db_cur.fetchall()
    logging.debug('[build_out_to_in_msg_prop_graphs_by_com] Fetch %s in_to_in_recs for %s in %s secs.'
                  % (len(l_in_to_in_recs), com_id, time.time() - timer_start))

    tw_db_cur.close()
    tw_db_conn.close()

    s_src_tw_ids = []
    graph_msg_prop = nx.DiGraph()
    for out_to_in_rec in l_out_to_in_recs:
        src_tw_id = out_to_in_rec[0]
        src_usr_id = out_to_in_rec[1]
        src_usr_sna_com_id = out_to_in_rec[2]
        src_tw_type = out_to_in_rec[3]
        src_tw_datetime = out_to_in_rec[4]
        src_l_nars = out_to_in_rec[5]
        src_l_stances = out_to_in_rec[6]
        trg_tw_id = out_to_in_rec[7]
        trg_usr_id = out_to_in_rec[8]
        trg_tw_type = out_to_in_rec[9]
        trg_tw_datetime = out_to_in_rec[10]
        trg_l_nars = out_to_in_rec[11]
        trg_l_stances = out_to_in_rec[12]
        graph_msg_prop.add_node(src_tw_id,
                                usr_id=src_usr_id,
                                usr_sna_com_id=src_usr_sna_com_id,
                                tw_type=src_tw_type,
                                tw_datetime=src_tw_datetime,
                                l_nars=src_l_nars,
                                l_stances=src_l_stances)
        graph_msg_prop.add_node(trg_tw_id,
                                usr_id=trg_usr_id,
                                usr_sna_com_id=com_id,
                                tw_type=trg_tw_type,
                                tw_datetime=trg_tw_datetime,
                                l_nars=trg_l_nars,
                                l_stances=trg_l_stances)
        graph_msg_prop.add_edge(src_tw_id, trg_tw_id)
        s_src_tw_ids.append(src_tw_id)
    s_src_tw_ids = set(s_src_tw_ids)

    for in_to_in_rec in l_in_to_in_recs:
        src_tw_id = in_to_in_rec[0]
        src_usr_id = in_to_in_rec[1]
        src_tw_type = in_to_in_rec[2]
        src_tw_datetime = in_to_in_rec[3]
        src_l_nars = in_to_in_rec[4]
        src_l_stances = in_to_in_rec[5]
        trg_tw_id = in_to_in_rec[6]
        trg_usr_id = in_to_in_rec[7]
        trg_tw_type = in_to_in_rec[8]
        trg_tw_datetime = in_to_in_rec[9]
        trg_l_nars = in_to_in_rec[10]
        trg_l_stances = in_to_in_rec[11]
        graph_msg_prop.add_node(src_tw_id,
                                usr_id=src_usr_id,
                                usr_sna_com_id=com_id,
                                tw_type=src_tw_type,
                                tw_datetime=src_tw_datetime,
                                l_nars=src_l_nars,
                                l_stances=src_l_stances)
        graph_msg_prop.add_node(trg_tw_id,
                                usr_id=trg_usr_id,
                                usr_sna_com_id=com_id,
                                tw_type=trg_tw_type,
                                tw_datetime=trg_tw_datetime,
                                l_nars=trg_l_nars,
                                l_stances=trg_l_stances)
        graph_msg_prop.add_edge(src_tw_id, trg_tw_id)

    logging.debug('[build_out_to_in_msg_prop_graphs_by_com] Build the whole msg prop graph for %s in %s secs: %s'
                  % (com_id, time.time() - timer_start, nx.info(graph_msg_prop)))

    d_msg_prop_graphs = dict()
    graph_msg_prop_und = nx.to_undirected(graph_msg_prop)
    for comp in nx.connected_components(graph_msg_prop_und):
        l_src_tw_id = []
        for node in comp:
            if graph_msg_prop.nodes(data=True)[node]['usr_sna_com_id'] != com_id:
                l_src_tw_id.append(node)
        if len(l_src_tw_id) > 1:
            raise Exception('Invalid msg prop graph with %s srcs.' % str(l_src_tw_id))
        if len(l_src_tw_id) == 1:
            d_msg_prop_graphs[l_src_tw_id[0]] = graph_msg_prop.subgraph(comp)
    logging.debug('[build_out_to_in_msg_prop_graphs_by_com] Get %s msg prop graphs in %s secs.'
                  % (len(d_msg_prop_graphs), time.time() - timer_start))

    if len(d_msg_prop_graphs) != len(s_src_tw_ids):
        logging.error('[build_out_to_in_msg_prop_graphs_by_com] Num of msg prop graphs: %s, Num of srcs: %s, Missing srcs: %s'
                      % (len(d_msg_prop_graphs), len(s_src_tw_ids), list(s_src_tw_ids - set(list(d_msg_prop_graphs.keys())))))

    for src_tw_id in d_msg_prop_graphs:
        msg_prop_graph_json = nx.adjacency_data(d_msg_prop_graphs[src_tw_id])
        with open(global_settings.g_microsim_msg_prop_graphs_file_format.format(str(com_id), src_tw_id), 'w+') as out_fd:
            json.dump(msg_prop_graph_json, out_fd)
            out_fd.close()
        # nx.write_gpickle(d_msg_prop_graphs[src_tw_id],
        #                  global_settings.g_microsim_msg_prop_graphs_file_format.format(str(com_id), src_tw_id))

    logging.debug('[build_out_to_in_msg_prop_graphs_by_com] All done in %s secs.' % str(time.time() - timer_start))


def nar_codes_to_nar_vec(l_nar_codes, nar_vec_len):
    nar_vec = np.zeros(nar_vec_len)
    for nar_code in l_nar_codes:
        nar_vec[int(nar_code)] = 1
    return nar_vec


def avg_leaf_nar_and_path_dist_by_msg_prop_graph(com_id, src_tw_id, nar_len):
    logging.debug('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] Starts with com %s and src %s' % (com_id, src_tw_id))

    if not path.exists(global_settings.g_microsim_msg_prop_graphs_file_format.format(str(com_id), src_tw_id)):
        logging.error('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] No msg prop graph for com %s src %s'
                      % (com_id, src_tw_id))
        return None

    with open(global_settings.g_microsim_msg_prop_graphs_file_format.format(str(com_id), src_tw_id), 'r') as in_fd:
        msg_prop_graph_json = json.load(in_fd)
        msg_prop_graph = nx.adjacency_graph(msg_prop_graph_json)
        in_fd.close()
    logging.debug('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] Load msg_prop_graph for com %s src %s: %s'
                  % (com_id, src_tw_id, nx.info(msg_prop_graph)))

    l_leaves = []
    l_roots = []
    for node in msg_prop_graph.nodes(data=True):
        if msg_prop_graph.out_degree(node[0]) == 0:
            l_leaves.append(node[0])
        if msg_prop_graph.in_degree(node[0]) == 0:
            l_roots.append(node[0])
    logging.debug('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] %s leaves for com %s src %s'
                  % (len(l_leaves), com_id, src_tw_id))
    if len(l_roots) > 1:
        raise Exception('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] Multiple roots for com %s src %s: %s' %
                        (com_id, src_tw_id, l_roots))
    elif len(l_roots) == 0:
        logging.error('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] No root for com %s src %s' % (com_id, src_tw_id))

    root_nar_vec = None
    if len(l_roots) > 0:
        l_nars = msg_prop_graph.nodes(data=True)[l_roots[0]]['l_nars']
    else:
        l_nars = None
    if l_nars is not None:
        root_nar_vec = np.asarray([1 if i in l_nars else 0 for i in range(nar_len)], dtype=np.float32)

    l_root_leaf_js = []
    nar_vec_cnt = 0
    avg_nar_vec = np.zeros(nar_len, dtype=np.float32)
    for node in l_leaves:
        l_nars = msg_prop_graph.nodes(data=True)[node]['l_nars']
        if l_nars is not None:
            leaf_nar_vec = np.asarray([1 if i in l_nars else 0 for i in range(nar_len)], dtype=np.float32)
            avg_nar_vec += leaf_nar_vec
            nar_vec_cnt += 1
            if root_nar_vec is not None and np.count_nonzero(leaf_nar_vec) > 0:
                root_leaf_js = jensenshannon(root_nar_vec, leaf_nar_vec)
                if np.isfinite(root_leaf_js):
                    l_root_leaf_js.append(root_leaf_js)
    if nar_vec_cnt > 0:
        avg_nar_vec = avg_nar_vec / nar_vec_cnt
    if not np.isfinite(avg_nar_vec).all():
        raise Exception('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] Invalid avg_nar_vec for com %s src %s'
                        % (com_id, src_tw_id))
    if np.count_nonzero(avg_nar_vec) <= 0:
        avg_nar_vec = None
    # logging.debug('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] avg_nar_vec for com %s src %s is %s'
    #               % (com_id, src_tw_id, avg_nar_vec))
    # logging.debug('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] %s root_leaf_js for com %s rec %s'
    #               % (len(l_root_leaf_js), com_id, src_tw_id))

    s_leaves = set(l_leaves)
    path_lens = nx.shortest_path_length(msg_prop_graph, src_tw_id)
    path_lens = [path_lens[trg_tw_id] for trg_tw_id in path_lens if trg_tw_id in s_leaves]
    # logging.debug('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] %s path lens for com %s src %s: %s'
    #               % (len(path_lens), com_id, src_tw_id, path_lens))

    logging.debug('[avg_leaf_nar_and_path_dist_by_msg_prop_graph] All done for com %s src %s.' % (com_id, src_tw_id))
    return (avg_nar_vec, path_lens, l_root_leaf_js)


def gt_avg_leaf_nar_and_path_dist_by_com(com_id, nar_len):
    logging.debug('[avg_leaf_nar_and_path_dist_by_com] Starts with com %s' % str(com_id))

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = """select 
                    src_tw_id
                from cp4.mf3jh_udt_tw_src_trg_sna_data 
                where trg_usr_sna_com_id = '{0}' and src_usr_sna_com_id != '{1}' and src_l_nars is not null""" \
        .format(str(com_id), str(com_id))
    tw_db_cur.execute(sql_str)
    l_recs = tw_db_cur.fetchall()
    l_src_tw_ids = [rec[0] for rec in l_recs]
    logging.debug('[avg_leaf_nar_and_path_dist_by_com] Fetch %s src_tw_ids for com %s' % (len(l_src_tw_ids), com_id))

    tw_db_cur.close()
    tw_db_conn.close()

    l_ready_recs = []
    for src_tw_id in l_src_tw_ids:
        avg_nar_vec, path_lens, l_root_leaf_js = avg_leaf_nar_and_path_dist_by_msg_prop_graph(com_id, src_tw_id, nar_len)
        path_len_mean = None
        path_len_std = None
        if len(path_lens) > 0:
            path_len_mean = np.mean(path_lens)
            path_len_std = np.std(path_lens)
        else:
            path_lens = None
        root_leaf_js_mean = None
        root_leaf_js_std = None
        if len(l_root_leaf_js) > 0:
            root_leaf_js_mean = np.mean(l_root_leaf_js)
            root_leaf_js_std = np.std(l_root_leaf_js)
        else:
            l_root_leaf_js = None
        l_ready_recs.append((src_tw_id, avg_nar_vec, path_lens, path_len_mean, path_len_std, l_root_leaf_js,
                             root_leaf_js_mean, root_leaf_js_std))
    df_out = pd.DataFrame(l_ready_recs, columns=['src_tw_id', 'avg_nar_vec', 'path_lens', 'path_len_mean',
                                                 'path_len_std', 'root_leaf_nar_js', 'root_leaf_nar_js_mean',
                                                 'root_leaf_nar_js_std'])
    df_out = df_out.drop_duplicates(['src_tw_id'])
    df_out.to_pickle(global_settings.g_microsim_msg_prop_graph_data_by_com_file_format.format(str(com_id)))
    logging.debug('[avg_leaf_nar_and_path_dist_by_com] All done for com %s' % str(com_id))


def build_one_sim_graph(com_id, src_tw_id, seed_id, sim_ret_path):
    logging.debug('[build_one_sim_graph] Starts with %s' % sim_ret_path)

    sim_graph = nx.DiGraph()
    l_pot_edges = []
    with open(sim_ret_path, 'r') as in_fd:
        csv_reader = csv.reader(in_fd, delimiter=',')
        ln_cnt = 0
        for row in csv_reader:
            if ln_cnt == 0:
                ln_cnt += 1
            else:
                usr_id = row[0]
                tweet_id = row[1]
                parent_id = row[2]
                root_id = row[3]
                narratives = row[4]
                if narratives[1:-1] == '':
                    narratives = []
                else:
                    narratives = [int(ele.strip()) for ele in narratives[1:-1].split(',')]
                step = int(row[5])
                root_category = int(row[6])
                if root_category != 0:
                    continue
                sim_graph.add_node(tweet_id,
                                   usr_id=usr_id,
                                   l_nars=narratives,
                                   step=step,
                                   root_id=root_id)
                l_pot_edges.append((parent_id, tweet_id))
        in_fd.close()

    for pot_edge in l_pot_edges:
        node_1 = pot_edge[0]
        node_2 = pot_edge[1]
        if node_1 not in sim_graph.nodes() or node_2 not in sim_graph.nodes():
            continue
        sim_graph.add_edge(node_1, node_2)
    logging.debug('[build_one_sim_graph] sim_graph done: %s.' % str(nx.info(sim_graph)))

    with open(global_settings.g_microsim_sim_graphs_file_format.format(str(com_id), src_tw_id, seed_id), 'w+') as out_fd:
        sim_graph_json = nx.adjacency_data(sim_graph)
        json.dump(sim_graph_json, out_fd)
        out_fd.close()
    logging.debug('[build_one_sim_graph] All done')


def build_sim_graphs_by_com(com_id):
    logging.debug('[build_sim_graphs_by_com] Starts with com %s' % str(com_id))
    timer_start = time.time()

    df_seeds = pd.read_csv(global_settings.g_microsim_seeds_by_com_folder_format.format(str(com_id))
                           + 'micro_sim_outside_gt_seed_{0}.csv'.format(str(com_id)))

    for (dirpath, dirname, filenames) in walk(global_settings.g_microsim_results_by_com_folder_format.format(str(com_id))):
        for filename in filenames:
            if filename[:15] != 'microsim-output' or filename[-4:] != '.csv':
                continue
            seed_id = filename[16:-4]
            if not path.exists(global_settings.g_microsim_seeds_by_com_folder_format.format(str(com_id)) + 'seed_{0}.csv'.format(seed_id)):
                raise Exception('[build_sim_graphs_by_com] No seed file for %s' % seed_id)
            src_tw_id = df_seeds.iloc[int(seed_id)]['src_tw_id']
            build_one_sim_graph(com_id, src_tw_id, seed_id, dirpath + filename)
    logging.debug('[build_sim_graphs_by_com] All done in %s secs.' % str(time.time() - timer_start))


def avg_leaf_nar_and_path_dist_by_sim_graph(sim_graph_path, com_id, src_tw_id, nar_len):
    logging.debug('[avg_leaf_nar_and_path_dist_by_sim_graph] Starts with com %s src %s...' % (com_id, src_tw_id))

    with open(sim_graph_path, 'r') as in_fd:
        sim_graph_json = json.load(in_fd)
        sim_graph = nx.adjacency_graph(sim_graph_json)
        in_fd.close()
    logging.debug('[avg_leaf_nar_and_path_dist_by_sim_graph] Load sim graph: %s' % str(nx.info(sim_graph)))

    l_leaves = []
    l_roots = []
    for node in sim_graph.nodes(data=True):
        if sim_graph.out_degree(node[0]) == 0:
            l_leaves.append(node[0])
        if sim_graph.in_degree(node[0]) == 0:
            l_roots.append(node[0])
    logging.debug('[avg_leaf_nar_and_path_dist_by_sim_graph] %s leaves for com %s src %s'
                  % (len(l_leaves), com_id, src_tw_id))
    if len(l_roots) > 1:
        raise Exception('[avg_leaf_nar_and_path_dist_by_sim_graph] Multiple roots for com %s src %s: %s'
                        % (com_id, src_tw_id, l_roots))
    elif len(l_roots) == 0:
        logging.error('[avg_leaf_nar_and_path_dist_by_sim_graph] No root for com %s src %s' % (com_id, src_tw_id))

    root_nar_vec = None
    if len(l_roots) > 0:
        l_nars = sim_graph.nodes(data=True)[l_roots[0]]['l_nars']
    else:
        l_nars = None
    if l_nars is not None:
        root_nar_vec = np.asarray([1 if i in l_nars else 0 for i in range(nar_len)], dtype=np.float32)

    l_root_leaf_js = []
    nar_vec_cnt = 0
    avg_nar_vec = np.zeros(nar_len, dtype=np.float32)
    for node in l_leaves:
        l_nars = sim_graph.nodes(data=True)[node]['l_nars']
        if l_nars is not None:
            leaf_nar_vec = np.asarray([1 if i in l_nars else 0 for i in range(nar_len)], dtype=np.float32)
            avg_nar_vec += leaf_nar_vec
            nar_vec_cnt += 1
            if root_nar_vec is not None and np.count_nonzero(leaf_nar_vec) > 0:
                root_leaf_js = jensenshannon(root_nar_vec, leaf_nar_vec)
                if np.isfinite(root_leaf_js):
                    l_root_leaf_js.append(root_leaf_js)
    if nar_vec_cnt > 0:
        avg_nar_vec = avg_nar_vec / nar_vec_cnt
    if not np.isfinite(avg_nar_vec).all():
        raise Exception('[avg_leaf_nar_and_path_dist_by_sim_graph] Invalid avg_nar_vec for com %s src %s'
                        % (com_id, src_tw_id))
    if np.count_nonzero(avg_nar_vec) <= 0:
        avg_nar_vec = None
    # logging.debug('[avg_leaf_nar_and_path_dist_by_sim_graph] avg_nar_vec for com %s src %s is %s'
    #               % (com_id, src_tw_id, avg_nar_vec))
    # logging.debug('[avg_leaf_nar_and_path_dist_by_sim_graph] %s root_leaf_js for com %s rec %s'
    #               % (len(l_root_leaf_js), com_id, src_tw_id))

    s_leaves = set(l_leaves)
    if len(l_roots) > 0:
        path_lens = nx.shortest_path_length(sim_graph, l_roots[0])
        path_lens = [path_lens[trg_tw_id] for trg_tw_id in path_lens if trg_tw_id in s_leaves]
    else:
        path_lens = []
    # logging.debug('[avg_leaf_nar_and_path_dist_by_sim_graph] %s path lens for com %s src %s: %s'
    #               % (len(path_lens), com_id, src_tw_id, path_lens))

    logging.debug('[avg_leaf_nar_and_path_dist_by_sim_graph] All done for com %s src %s.' % (com_id, src_tw_id))
    return (avg_nar_vec, path_lens, l_root_leaf_js)


def sim_avg_leaf_nar_and_path_dist_by_com(com_id, nar_len):
    logging.debug('[sim_avg_leaf_nar_and_path_dist_by_com] Starts with com %s' % str(com_id))

    l_ready_recs = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_microsim_sim_graphs_folder):
        for filename in filenames:
            if filename[:9] != 'sim_graph' or filename[-5:] != '.json':
                continue
            fields = filename[len('sim_graph_com_'):-5].split('_src_')
            graph_com_id = int(fields[0])
            if graph_com_id != int(com_id):
                continue
            src_tw_id = fields[1].split('_seed_')[0]
            avg_nar_vec, path_lens, l_root_leaf_js = avg_leaf_nar_and_path_dist_by_sim_graph(dirpath + filename, graph_com_id, src_tw_id, nar_len)
            path_len_mean = None
            path_len_std = None
            if len(path_lens) > 0:
                path_len_mean = np.mean(path_lens)
                path_len_std = np.std(path_lens)
            else:
                path_lens = None
            root_leaf_js_mean = None
            root_leaf_js_std = None
            if len(l_root_leaf_js) > 0:
                root_leaf_js_mean = np.mean(l_root_leaf_js)
                root_leaf_js_std = np.std(l_root_leaf_js)
            else:
                l_root_leaf_js = None
            l_ready_recs.append((src_tw_id, avg_nar_vec, path_lens, path_len_mean, path_len_std, l_root_leaf_js,
                                 root_leaf_js_mean, root_leaf_js_std))

    df_out = pd.DataFrame(l_ready_recs, columns=['src_tw_id', 'avg_nar_vec', 'path_lens', 'path_len_mean',
                                                 'path_len_std', 'root_leaf_nar_js', 'root_leaf_nar_js_mean',
                                                 'root_leaf_nar_js_std'])
    logging.debug('[sim_avg_leaf_nar_and_path_dist_by_com] prelim df_out done with %s recs.' % str(len(df_out)))

    d_dups = dict()
    df_out_dup = df_out.duplicated(['src_tw_id'], keep=False)
    for idx, dup_rec in df_out_dup.iteritems():
        rec = df_out.iloc[idx]
        src_tw_id = rec['src_tw_id']
        if dup_rec and src_tw_id not in d_dups:
            d_dups[src_tw_id] = {'ids': [idx], 'recs': [rec]}
            continue
        if src_tw_id in d_dups:
            d_dups[src_tw_id]['ids'].append(idx)
            d_dups[src_tw_id]['recs'].append(rec)
    logging.debug('[sim_avg_leaf_nar_and_path_dist_by_com] %s src_tw_ids have duplicates, %s duplicates in total.'
                  % (len(d_dups), sum([len(d_dups[key]['ids']) for key in d_dups])))

    l_new_recs = []
    for src_tw_id in d_dups:
        avg_nar_vec = np.zeros(nar_len)
        avg_nar_vec_cnt = 0
        path_lens = []
        root_leaf_nar_js = []
        for rec in d_dups[src_tw_id]['recs']:
            dup_avg_nar_vec = rec['avg_nar_vec']
            if dup_avg_nar_vec is not None:
                avg_nar_vec += dup_avg_nar_vec
                avg_nar_vec_cnt += 1
            dup_path_lens = rec['path_lens']
            if dup_path_lens is not None:
                path_lens += dup_path_lens
            dup_root_leaf_nar_js = rec['root_leaf_nar_js']
            if dup_root_leaf_nar_js is not None:
                root_leaf_nar_js += dup_root_leaf_nar_js

        if np.count_nonzero(avg_nar_vec) <= 0:
            avg_nar_vec = None
        elif avg_nar_vec_cnt > 0:
            avg_nar_vec = avg_nar_vec / avg_nar_vec_cnt
        path_len_mean = None
        path_len_std = None
        if path_lens is None or len(path_lens) <= 0:
            path_lens = None
        else:
            path_len_mean = np.mean(path_lens)
            path_len_std = np.std(path_lens)
        root_leaf_nar_js_mean = None
        root_leaf_nar_js_std = None
        if root_leaf_nar_js is None or len(root_leaf_nar_js) <= 0:
            root_leaf_nar_js = None
        else:
            root_leaf_nar_js_mean = np.mean(path_lens)
            root_leaf_nar_js_std = np.std(path_lens)
        l_new_recs.append((src_tw_id, avg_nar_vec, path_lens, path_len_mean, path_len_std, root_leaf_nar_js,
                           root_leaf_nar_js_mean, root_leaf_nar_js_std))

        for idx in d_dups[src_tw_id]['ids']:
            df_out = df_out.drop(idx)
    df_new = pd.DataFrame(l_new_recs, columns=['src_tw_id', 'avg_nar_vec', 'path_lens', 'path_len_mean',
                                                 'path_len_std', 'root_leaf_nar_js', 'root_leaf_nar_js_mean',
                                                 'root_leaf_nar_js_std'])
    df_out = df_out.append(df_new)
    df_out.reset_index(drop=True, inplace=True)

    df_out.to_pickle(global_settings.g_microsim_sim_graph_data_by_com_file_format.format(str(com_id)))
    logging.debug('[sim_avg_leaf_nar_and_path_dist_by_com] All done for com %s with %s recs' % (com_id, len(df_out)))


def compare_msg_prop_graph_data_with_sim_graph_data_by_com(com_id):
    logging.debug('[compare_msg_prop_graph_data_with_sim_graph_data_by_com] Starts with com %s...' % str(com_id))

    df_msg_prop_graph_data = pd.read_pickle(global_settings.g_microsim_msg_prop_graph_data_by_com_file_format.format(str(com_id)))
    df_msg_prop_graph_data = df_msg_prop_graph_data.set_index('src_tw_id')
    logging.debug('[compare_msg_prop_graph_data_with_sim_graph_data_by_com] Load df_msg_prop_graph_data with %s recs.'
                  % str(len(df_msg_prop_graph_data)))
    df_sim_graph_data = pd.read_pickle(global_settings.g_microsim_sim_graph_data_by_com_file_format.format(str(com_id)))
    df_sim_graph_data = df_sim_graph_data.set_index('src_tw_id')
    logging.debug('[compare_msg_prop_graph_data_with_sim_graph_data_by_com] Load df_sim_graph_data with %s recs.'
                  % str(len(df_sim_graph_data)))

    l_metric_recs = []
    for src_tw_id, rec in df_sim_graph_data.iterrows():
        if not src_tw_id in df_msg_prop_graph_data.index:
            logging.error('[compare_msg_prop_graph_data_with_sim_graph_data_by_com] %s not in df_msg_prop_graph_data'
                          % src_tw_id)
        sim_avg_nar_vec = rec['avg_nar_vec']
        sim_path_lens = rec['path_lens']
        if sim_path_lens is not None and np.sum(sim_path_lens) > 0:
            sim_path_lens = sim_path_lens / np.sum(sim_path_lens)
        gt_avg_nar_vec = df_msg_prop_graph_data.loc[src_tw_id]['avg_nar_vec']
        gt_path_lens = df_msg_prop_graph_data.loc[src_tw_id]['path_lens']
        if gt_path_lens is not None and np.sum(gt_path_lens) > 0:
            gt_path_lens = gt_path_lens / np.sum(gt_path_lens)

        if sim_avg_nar_vec is None \
                or gt_avg_nar_vec is None \
                or np.count_nonzero(sim_avg_nar_vec) == 0 \
                or np.count_nonzero(gt_avg_nar_vec) == 0:
            js_div = None
        else:
            js_div = jensenshannon(sim_avg_nar_vec, gt_avg_nar_vec)

        if sim_path_lens is None \
                or gt_path_lens is None \
                or len(sim_path_lens) == 0 \
                or len(gt_path_lens) == 0:
            ws = None
        else:
            ws = wasserstein_distance(sim_path_lens, gt_path_lens)
        l_metric_recs.append((src_tw_id, js_div, ws))
    df_out = pd.DataFrame(l_metric_recs, columns=['src_tw_id', 'nar_js_div', 'path_len_ws'])
    df_out.to_pickle(global_settings.g_microsim_analysis_sim_vs_gt_by_com_file_format.format(str(com_id)))
    logging.debug('[compare_msg_prop_graph_data_with_sim_graph_data_by_com] Output g_microsim_analysis_sim_vs_gt_by_com_file for com %s with %s recs.'
                  % (com_id, len(df_out)))


def draw_compare_msg_prop_graph_data_with_sim_graph_data_by_com(com_id):
    logging.debug('[draw_compare_msg_prop_graph_data_with_sim_graph_data_by_com] Starts with com %s' % str(com_id))
    df_metrics = pd.read_pickle(global_settings.g_microsim_analysis_sim_vs_gt_by_com_file_format.format(str(com_id)))
    df_metrics = df_metrics.set_index('src_tw_id')
    logging.debug('[draw_compare_msg_prop_graph_data_with_sim_graph_data_by_com] Load g_microsim_analysis_sim_vs_gt_by_com_file with %s recs.'
                  % str(len(df_metrics)))

    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))

    fig_id = 0
    axes[fig_id].grid(True)
    axes[fig_id].set_title("Simulation vs Ground Truth: Narratives JS-Div for Community {0}".format(str(com_id)), fontsize=20)
    axes[fig_id].set_xticks([k for k in range(len(df_metrics.index))])
    axes[fig_id].set_xticklabels(list(df_metrics.index), rotation=45)
    markerline, stemlines, baselineaxes = axes[fig_id].stem(df_metrics['nar_js_div'].values.tolist(), use_line_collection=True)
    markerline.set_markerfacecolor('b')

    fig_id = 1
    axes[fig_id].grid(True)
    axes[fig_id].set_title("Simulation vs Ground Truth: Root-to-Leaf Path Length Wasserstein Distances for Community {0}".format(str(com_id)), fontsize=20)
    axes[fig_id].set_xticks([k for k in range(len(df_metrics.index))])
    axes[fig_id].set_xticklabels(list(df_metrics.index), rotation=45)
    markerline, stemlines, baselineaxes = axes[fig_id].stem(df_metrics['path_len_ws'].values.tolist(), use_line_collection=True)
    markerline.set_markerfacecolor('r')

    plt.tight_layout(pad=3.0)
    plt.savefig(global_settings.g_microsim_analysis_results_folder + 'sim_vs_gt_com_{0}.png'.format(str(com_id)), format='PNG')
    plt.clf()
    plt.close()


def analysis_summary(com_id):
    logging.debug('[analysis_summary] Starts with com %s' % str(com_id))

    df_gt = pd.read_pickle(global_settings.g_microsim_ana_sum_gt_alone_by_com_folder_format.format(str(com_id))
                           + 'msg_prop_graph_data_com_{0}.pickle'.format(str(com_id)))
    df_gt = df_gt.set_index('src_tw_id')
    logging.debug('[analysis_summary] Load df_gt with %s recs.' % str(len(df_gt)))

    l_df_sim = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_microsim_ana_sum_sim_alone_by_com_folder_format.format(str(com_id))):
        for filename in filenames:
            if filename[:14] != 'sim_graph_data' or filename[-7:] != '.pickle':
                continue
            df_sim = pd.read_pickle(dirpath + filename)
            df_sim = df_sim.set_index('src_tw_id')
            l_df_sim.append(df_sim)
            logging.debug('[analysis_summary] Load %s df_sim with %s recs' % (len(l_df_sim), len(df_sim)))

    l_gt_sim_alone_recs = []
    for src_tw_id, gt_rec in df_gt.iterrows():
        gt_root_leaf_nar_js_mean = gt_rec['root_leaf_nar_js_mean']
        gt_root_leaf_nar_js_std = gt_rec['root_leaf_nar_js_std']
        gt_path_len_mean = gt_rec['path_len_mean']
        gt_path_len_std = gt_rec['path_len_std']

        l_sim_root_leaf_nar_js = []
        l_sim_path_lens = []
        for df_sim in l_df_sim:
            if src_tw_id not in df_sim.index:
                continue
            sim_rec = df_sim.loc[src_tw_id]
            sim_root_leaf_nar_js = sim_rec['root_leaf_nar_js']
            sim_path_lens = sim_rec['path_lens']
            if sim_root_leaf_nar_js is not None:
                l_sim_root_leaf_nar_js += sim_root_leaf_nar_js
            if sim_path_lens is not None:
                l_sim_path_lens += sim_path_lens
        if len(l_sim_root_leaf_nar_js) <= 0:
            sim_root_leaf_nar_js_mean = None
            sim_root_leaf_nar_js_std = None
        else:
            sim_root_leaf_nar_js_mean = np.mean(l_sim_root_leaf_nar_js)
            sim_root_leaf_nar_js_std = np.std(l_sim_root_leaf_nar_js)
        if len(l_sim_path_lens) <= 0:
            sim_path_len_mean = None
            sim_path_len_std = None
        else:
            sim_path_len_mean = np.mean(l_sim_path_lens)
            sim_path_len_std = np.mean(l_sim_path_lens)
        l_gt_sim_alone_recs.append((src_tw_id, gt_path_len_mean, gt_path_len_std, gt_root_leaf_nar_js_mean,
                                    gt_root_leaf_nar_js_std, sim_path_len_mean, sim_path_len_std,
                                    sim_root_leaf_nar_js_mean, sim_root_leaf_nar_js_std))
    df_gt_sim_alone = pd.DataFrame(l_gt_sim_alone_recs, columns=['src_tw_id', 'gt_path_len_mean', 'gt_path_len_std',
                                                                 'gt_root_leaf_nar_js_mean', 'gt_root_leaf_nar_js_std',
                                                                 'sim_path_len_mean', 'sim_path_len_std',
                                                                 'sim_root_leaf_nar_js_mean', 'sim_root_leaf_nar_js_std'])
    df_gt_sim_alone = df_gt_sim_alone.set_index('src_tw_id')
    df_gt_sim_alone.to_csv(global_settings.g_microsim_ana_sum_gt_sim_alone_table_file_by_com_file_format.format(str(com_id)))
    logging.debug('[analysis_summary] Output df_gt_sim_alone with %s recs.' % str(len(df_gt_sim_alone)))

    l_df_sim_vs_gt = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_microsim_ana_sum_sim_vs_gt_by_com_folder_format.format(str(com_id))):
        for filename in filenames:
            if filename[:9] != 'sim_vs_gt' or filename[-7:] != '.pickle':
                continue
            df_sim_vs_gt = pd.read_pickle(dirpath + filename)
            df_sim_vs_gt = df_sim_vs_gt.set_index('src_tw_id')
            l_df_sim_vs_gt.append(df_sim_vs_gt)
            logging.debug('[analysis_summary] Load %s df_sim_vs_gt with %s recs'
                          % (len(l_df_sim_vs_gt), len(df_sim_vs_gt)))

    l_sim_vs_gt_recs = []
    l_nar_js_div = []
    l_path_len_ws = []
    for src_tw_id, gt_rec in df_gt.iterrows():
        for df_sim_vs_gt in l_df_sim_vs_gt:
            if src_tw_id not in df_sim_vs_gt.index:
                continue
            sim_vs_gt_rec = df_sim_vs_gt.loc[src_tw_id]
            nar_js_div = sim_vs_gt_rec['nar_js_div']
            path_len_ws = sim_vs_gt_rec['path_len_ws']
            if nar_js_div is not None and np.isfinite(nar_js_div):
                l_nar_js_div.append(nar_js_div)
            if path_len_ws is not None and np.isfinite(path_len_ws):
                l_path_len_ws.append(path_len_ws)
        if len(l_nar_js_div) <= 0:
            nar_js_mean = None
            nar_js_std = None
        else:
            nar_js_mean = np.mean(l_nar_js_div)
            nar_js_std = np.std(l_nar_js_div)
        if len(l_path_len_ws) <= 0:
            path_len_ws_mean = None
            path_len_ws_std = None
        else:
            path_len_ws_mean = np.mean(l_path_len_ws)
            path_len_ws_std = np.std(l_path_len_ws)
        l_sim_vs_gt_recs.append((src_tw_id, path_len_ws_mean, path_len_ws_std, nar_js_mean, nar_js_std))
    df_sim_vs_gt = pd.DataFrame(l_sim_vs_gt_recs, columns=['src_tw_id', 'path_len_ws_mean', 'path_len_ws_std',
                                                           'nar_js_mean', 'nar_js_std'])
    df_sim_vs_gt = df_sim_vs_gt.set_index('src_tw_id')
    df_sim_vs_gt.to_csv(global_settings.g_microsim_ana_sum_sim_vs_gt_table_file_by_com_file_format.format(str(com_id)))
    logging.debug('[analysis_summary] Output df_sim_vs_gt with %s recs.' % str(len(df_sim_vs_gt)))


def simperiod_analysis_summary():
    logging.debug('[simperiod_analysis] Starts...')

    d_periods = dict()
    for (dirpath, dirname, filenames) in walk(global_settings.g_microsim_ana_simperiod_folder):
        for filename in filenames:
            if filename[:9] != 'SimPeriod' or filename[-13:] != 'NarrTable.csv':
                continue
            period = filename[9:18]
            d_f1 = dict()
            with open(dirpath + filename, 'r') as in_fd:
                csv_reader = csv.reader(in_fd, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        nar = row[0]
                        if row[8] == '':
                            f1 = 0.0
                        else:
                            f1 = float(row[8])
                        d_f1[nar] = f1
                in_fd.close()
            d_periods[period] = d_f1

    l_periods = sorted(list(d_periods.keys()))
    l_nars = list(d_periods[l_periods[0]].keys())

    l_ready_recs = []
    for nar in l_nars:
        nar_rec = tuple([nar] + [d_periods[period][nar] for period in l_periods])
        l_ready_recs.append(nar_rec)

    df_out = pd.DataFrame(l_ready_recs, columns=['nar']+l_periods)
    df_out.to_pickle(global_settings.g_microsim_ana_simperiod_sum_file)
    logging.debug('[simperiod_analysis] All done.')


def simperiod_analysis_draw():
    logging.debug('[simperiod_analysis_draw] Starts...')

    df_simperiod = pd.read_pickle(global_settings.g_microsim_ana_simperiod_sum_file)
    df_simperiod = df_simperiod.set_index('nar')

    l_nars = list(df_simperiod.index)
    l_periods = list(df_simperiod.columns)

    mat_simperiod = np.zeros((len(l_nars), len(l_periods)), dtype=np.float32)
    for nar_idx, nar in enumerate(l_nars):
        for period_idx, period in enumerate(l_periods):
            mat_simperiod[nar_idx][period_idx] = df_simperiod.loc[nar][period]

    d_sorted_nars = dict()
    for period in l_periods:
        df_period = df_simperiod[[period]]
        df_period = df_period.sort_values(by=[period], ascending=False)
        d_sorted_nars[period] = df_period

    l_draw_data = []
    # l_rev_periods = reversed(l_periods)
    for period in l_periods:
        l_nar_idx = [l_nars.index(nar) for nar in list(d_sorted_nars[period].index)]
        l_draw_data.append(l_nar_idx)

    l_draw_f1 = []
    for period in l_periods:
        l_f1 = list(d_sorted_nars[period].values)
        l_draw_f1.append(l_f1)

    # fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 15))
    # cur_ax = axes
    # cur_ax.set_title("F1 Scores per Narrative", fontsize=15)
    # cur_ax.set_xticks([k for k in range(len(l_periods))])
    # cur_ax.set_xticklabels(l_periods, rotation=45, ha='center')
    # cur_ax.set_yticks([k for k in range(len(l_nars))])
    # cur_ax.set_yticklabels(l_nars)
    # pos = cur_ax.imshow(mat_simperiod, cmap='hot')
    # divider = make_axes_locatable(cur_ax)
    # cax = divider.append_axes("right", size="10%", pad=0.1)
    # fig.colorbar(pos, ax=cur_ax, cax=cax)
    # fig.align_labels(axs=cur_ax)
    # plt.show()
    # plt.clf()
    # plt.close()

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(35, 6))
    cur_ax = axes
    cur_ax.set_title("Sorted F1 Scores per Narrative", fontsize=20)
    cur_ax.set_yticks([k + 0.5 for k in range(len(l_periods))])
    cur_ax.set_yticklabels(l_periods, va='center')
    cur_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    l_colors = []
    for i in range(len(l_nars)):
        l_colors.append(np.random.rand(3,))
    newcmp = ListedColormap(l_colors, name='my_cmp')
    x = [i for i in range(len(l_nars) + 1)]
    y = [i for i in range(len(l_periods) + 1)]
    cur_ax.pcolormesh(x, y, l_draw_data, cmap=newcmp, edgecolors='black')
    for i in range(len(l_nars)):
        for j in range(len(l_periods)):
            cur_ax.text(i + 0.5, j + 0.5, '%.2f' % l_draw_f1[j][i],
                     horizontalalignment='center',
                     verticalalignment='center',)
    l_patches = []
    for i in range(len(l_nars)):
        nar_patch = mpatches.Patch(color=l_colors[i], label=l_nars[i])
        l_patches.append(nar_patch)
    cur_ax.legend(handles=l_patches, loc='lower left', ncol=math.ceil(len(l_nars) / 5), bbox_to_anchor=(0, -0.3))
    fig.align_labels(axs=cur_ax)
    plt.show()
    plt.clf()
    plt.close()

    print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'msg_prop_graphs':
        com_id = sys.argv[2]
        build_out_to_in_msg_prop_graphs_by_com(com_id)

    elif cmd == 'msg_prop_graph_data':
        com_id = sys.argv[2]
        nar_len = 48
        gt_avg_leaf_nar_and_path_dist_by_com(com_id, nar_len)

    elif cmd == 'sim_graphs':
        com_id = sys.argv[2]
        build_sim_graphs_by_com(com_id)

    elif cmd == 'sim_graph_data':
        com_id = sys.argv[2]
        nar_len = 48
        sim_avg_leaf_nar_and_path_dist_by_com(com_id, nar_len)

    elif cmd == 'analysis':
        com_id = sys.argv[2]
        # compare_msg_prop_graph_data_with_sim_graph_data_by_com(com_id)
        # draw_compare_msg_prop_graph_data_with_sim_graph_data_by_com(com_id)
        analysis_summary(com_id)

    elif cmd == 'simperiod':
        simperiod_analysis_summary()
        simperiod_analysis_draw()