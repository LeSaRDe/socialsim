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
import copy

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
from scipy.spatial.distance import jensenshannon

import global_settings
sys.path.insert(1, global_settings.g_lexvec_model_folder)
import model as lexvec


"""
new_tw_data
"""
def extract_new_tw_data():
    logging.debug('[extract_new_tw_data] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = """select tw_id, usr_id, tw_datetime from cp4.mf3jh_ven_tw_en_all where tw_type='n'"""
    tw_db_cur.execute(tw_db_sql_str)
    l_recs = tw_db_cur.fetchall()
    logging.debug('[extract_new_tw_data] Fetch %s new tw data in %s secs.' % (len(l_recs), time.time() - timer_start))

    df_new_tw_data = pd.DataFrame(l_recs, columns=['tw_id', 'usr_id', 'tw_datetime'])
    df_new_tw_data.to_pickle(global_settings.g_tw_new_tw_data)
    logging.debug('[extract_new_tw_data] All done in %s secs.' % str(time.time() - timer_start))


def collect_new_tws_by_community():
    logging.debug('[collect_tws_by_community] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_sql_str = """select id, community from cp4.sna"""
    tw_db_cur.execute(tw_sql_str)
    l_sna_recs = tw_db_cur.fetchall()

    d_com_usr = dict()
    d_usr_com = dict()
    for sna_rec in l_sna_recs:
        usr_id = sna_rec[0]
        if sna_rec[1] is None:
            continue
        com_id = int(sna_rec[1])
        d_usr_com[usr_id] = com_id
        if com_id not in d_com_usr:
            d_com_usr[com_id] = [usr_id]
        else:
            d_com_usr[com_id].append(usr_id)
    logging.debug('[collect_tws_by_community] d_com_usr and d_usr_com done with %s communities and %s usrs in %s secs.'
                  % (len(d_com_usr), len(d_usr_com), time.time() - timer_start))

    tw_sql_str = """select tw_id, tw_type, usr_id, tw_datetime from cp4.mf3jh_ven_tw_en_all"""
    tw_db_cur.execute(tw_sql_str)
    l_tw_recs = tw_db_cur.fetchall()

    d_com_tw = {com_id: [] for com_id in d_com_usr}
    cnt = 0
    for tw_rec in l_tw_recs:
        cnt += 1
        if cnt % 100000 == 0 and cnt >= 100000:
            logging.debug('[collect_tws_by_community] scan %s tw_recs in %s secs.' % (cnt, time.time() - timer_start))
        tw_id = tw_rec[0]
        tw_type = tw_rec[1]
        if tw_type != 'n':
            continue
        usr_id = tw_rec[2]
        tw_datetime = tw_rec[3]
        if usr_id not in d_usr_com:
            continue
        com_id = d_usr_com[usr_id]
        d_com_tw[com_id].append((tw_id, usr_id, tw_datetime))
    logging.debug('[collect_tws_by_community] d_com_tw done in %s secs.' % str(time.time() - timer_start))
    out_str = ''
    for com_id in d_com_tw:
        out_str += str(com_id) + ':' + str(len(d_com_tw[com_id])) + ', '
    out_str = out_str[:-2]
    logging.debug('[collect_tws_by_community] d_com_tw stats = %s.' % out_str)

    l_com_tw_df_recs = []
    for com_id in d_com_tw:
        l_tw_dt_df_recs = d_com_tw[com_id]
        if len(l_tw_dt_df_recs) <= 0:
            continue
        df_tw_dt = pd.DataFrame(l_tw_dt_df_recs, columns=['tw_id', 'usr_id', 'tw_datetime'])
        l_com_tw_df_recs.append((com_id, df_tw_dt))
    if len(l_com_tw_df_recs) <= 0:
        logging.error('[collect_tws_by_community] No available com_tw_df_recs.')
    else:
        df_com_tw = pd.DataFrame(l_com_tw_df_recs, columns=['com_id', 'tw_dt'])
        df_com_tw.to_pickle(global_settings.g_tw_new_tw_by_com_file)

    tw_db_cur.close()
    tw_db_conn.close()
    logging.debug('[collect_tws_by_community] All done. Output df_com_tw in %s secs.' % str(time.time() - timer_start))


def split_new_tw_data(df_new_tw_data, com_id, day_delta, dt_start, dt_end):
    '''
    day_delta, dt_start, dt_end should be strings
    for intervals, (int_start, int_end]
    '''
    logging.debug('[split_new_tw_data] Starts with day_delta=%s, dt_start=%s, dt_end=%s'
                  % (str(day_delta), str(dt_start), str(dt_end)))
    timer_start = time.time()

    # df_new_tw_data = pd.read_pickle(global_settings.g_tw_new_tw_data)
    df_new_tw_data = df_new_tw_data.sort_values(by='tw_datetime')
    logging.debug('[split_new_tw_data] Load g_tw_new_tw_data with %s recs in %s secs.'
                  % (len(df_new_tw_data), time.time() - timer_start))

    datetime_fmt = '%Y%m%d%H%M%S'
    day_delta = timedelta(days=int(day_delta))
    int_start = datetime.strptime(dt_start, datetime_fmt)
    int_end = int_start + day_delta
    if int_end > datetime.strptime(dt_end, datetime_fmt):
        int_end = datetime.strptime(dt_end, datetime_fmt)
    l_ready_rec = []
    for rec in df_new_tw_data.values:
        tw_datetime = rec[2]
        tw_dt = datetime.strptime(tw_datetime, datetime_fmt)
        if tw_dt <= int_start:
            continue
        if tw_dt > int_end:
            if len(l_ready_rec) > 0:
                out_str = datetime.strftime(int_start, datetime_fmt) + '#' + datetime.strftime(int_end, datetime_fmt)
                df_out = pd.DataFrame(l_ready_rec, columns=['tw_id', 'usr_id', 'tw_datetime'])
                df_out.to_pickle(global_settings.g_tw_new_tw_data_time_int_file_format.format(str(com_id), out_str))
                logging.debug('[split_new_tw_data] Output %s with %s split date in %s secs.'
                              % (out_str, len(df_out), time.time() - timer_start))
            l_ready_rec = []
            int_start = int_end
            if int_start > datetime.strptime(dt_end, datetime_fmt):
                logging.debug('[split_new_tw_data] All done in %s secs.' % str(time.time() - timer_start))
                return
            int_end += day_delta
            if int_end > datetime.strptime(dt_end, datetime_fmt):
                int_end = datetime.strptime(dt_end, datetime_fmt)
            if tw_dt > int_end:
                logging.debug('[split_new_tw_data] All done in %s secs.' % str(time.time() - timer_start))
                return
            else:
                l_ready_rec.append(rec)
        else:
            l_ready_rec.append(rec)
    if len(l_ready_rec) > 0:
        out_str = datetime.strftime(int_start, datetime_fmt) + '#' + datetime.strftime(int_end, datetime_fmt)
        df_out = pd.DataFrame(l_ready_rec, columns=['tw_id', 'usr_id', 'tw_datetime'])
        df_out.to_pickle(global_settings.g_tw_new_tw_data_time_int_file_format.format(str(com_id), out_str))
        logging.debug('[split_new_tw_data] Output %s with %s split date in %s secs.'
                      % (out_str, len(df_out), time.time() - timer_start))
    logging.debug('[split_new_tw_data] All done in %s secs.' % str(time.time() - timer_start))
    return


def split_new_tw_data_for_communities(day_delta, dt_start, dt_end):
    logging.debug('[split_new_tw_data_for_communities] Starts...')
    timer_start = time.time()

    df_new_tw_by_com = pd.read_pickle(global_settings.g_tw_new_tw_by_com_file)
    logging.debug('[split_new_tw_data_for_communities] Load g_tw_new_tw_by_com_file with %s recs in %s secs.'
                  % (len(df_new_tw_by_com), time.time() - timer_start))

    d_tw_by_com_stats = dict()
    for new_tw_by_com_rec in df_new_tw_by_com.values:
        com_id = int(new_tw_by_com_rec[0])
        df_tw_rec = new_tw_by_com_rec[1]
        d_tw_by_com_stats[com_id] = len(df_tw_rec)
        split_new_tw_data(df_tw_rec, com_id, day_delta, dt_start, dt_end)
    with open(global_settings.g_tw_new_tw_by_com_stats_file, 'w+') as out_fd:
        json.dump(d_tw_by_com_stats, out_fd)
        out_fd.close()
    logging.debug('[split_new_tw_data_for_communities] All done in %s secs.' % str(time.time() - timer_start))


"""
time_int_diff
"""
def phrase_cluster_distributions_and_embeds_for_time_ints_single_proc(l_tasks, n_clusters, p_id):
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: Starts with %s new_tw_data_int_files'
                  % (p_id, str(len(l_tasks))))
    timer_start = time.time()

    df_tw_phid = pd.read_pickle(global_settings.g_tw_phrase_extraction_tw_to_phids_file)
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: Load g_tw_phrase_extraction_tw_to_phids_file with %s tws in %s secs.'
                  % (p_id, len(df_tw_phid), time.time() - timer_start))

    df_pc_onehot = pd.read_pickle(global_settings.g_tw_phrase_cluster_onehot_embeds_all_file_format.format(str(n_clusters)))
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: Load g_tw_phrase_cluster_onehot_embeds_all_file with %s phrase embeds in %s secs.'
                  % (p_id, len(df_pc_onehot), time.time() - timer_start))

    df_ph_embed = pd.read_pickle(global_settings.g_tw_raw_phrases_embeds)
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: Load g_tw_raw_phrases_embeds with %s phrase embeds in %s secs.'
                  % (p_id, len(df_ph_embed), time.time() - timer_start))

    d_new_tw_tw_ids = dict()
    cnt = 0
    for int_id, com_id, new_tw_data_time_int_file_path in l_tasks:
        df_new_tw_data_int = pd.read_pickle(new_tw_data_time_int_file_path)
        cnt += len(df_new_tw_data_int)
        l_task_tw_ids = df_new_tw_data_int['tw_id'].to_list()
        if int_id not in d_new_tw_tw_ids:
            d_new_tw_tw_ids[int_id] = dict()
        d_new_tw_tw_ids[int_id][com_id] = l_task_tw_ids
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: Load %s new tw data.'
                  % (p_id, cnt))

    s_all_task_tw_ids = []
    for int_id in d_new_tw_tw_ids:
        for com_id in d_new_tw_tw_ids[int_id]:
            s_all_task_tw_ids += d_new_tw_tw_ids[int_id][com_id]
    s_all_task_tw_ids = set(s_all_task_tw_ids)
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: %s task tw_ids.'
                  % (p_id, len(s_all_task_tw_ids)))

    d_tw_to_phids = {tw_id: None for tw_id in s_all_task_tw_ids}
    d_phid_to_onehot = dict()
    d_phid_to_embed = dict()
    for tw_phid_rec in df_tw_phid.values:
        tw_id = tw_phid_rec[0]
        raw_phs = tw_phid_rec[1]
        if tw_id in d_tw_to_phids:
            d_tw_to_phids[tw_id] = raw_phs
            for phid in d_tw_to_phids[tw_id]:
                if phid not in d_phid_to_onehot:
                    d_phid_to_onehot[int(phid)] = None
                    d_phid_to_embed[int(phid)] = None
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: %s task phids in %s secs.'
                  % (p_id, len(d_phid_to_onehot), time.time() - timer_start))

    for phid_onehot_rec in df_pc_onehot.values:
        phid = int(phid_onehot_rec[0])
        if phid not in d_phid_to_onehot:
            continue
        ph_onehot = np.asarray(phid_onehot_rec[1], dtype=np.float32)
        d_phid_to_onehot[phid] = ph_onehot
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: done ph_onehots in %s secs.'
                  % (p_id, time.time() - timer_start))

    for phid_embed_rec in df_ph_embed.values:
        phid = int(phid_embed_rec[0])
        if phid not in d_phid_to_embed:
            continue
        ph_embed = np.asarray(phid_embed_rec[1], dtype=np.float32)
        d_phid_to_embed[phid] = ph_embed
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: done ph_embeds in %s secs.'
                  % (p_id, time.time() - timer_start))

    for int_id in d_new_tw_tw_ids:
        new_tw_pc_sum_vec = np.zeros(n_clusters, dtype=np.float32)
        new_tw_embed_sum_vec = np.zeros(300, dtype=np.float32)

        for com_id in d_new_tw_tw_ids[int_id]:
            embed_cnt = 0
            l_tw_ids = d_new_tw_tw_ids[int_id][com_id]
            eff_tw_cnt = 0
            for tw_id in l_tw_ids:
                l_phids = d_tw_to_phids[tw_id]
                if l_phids is not None:
                    eff_tw_cnt += 1
                    for phid in l_phids:
                        ph_onehot = d_phid_to_onehot[int(phid)]
                        new_tw_pc_sum_vec += ph_onehot
                        ph_embed = d_phid_to_embed[int(phid)]
                        new_tw_embed_sum_vec += ph_embed
                        embed_cnt += 1
                else:
                    logging.error('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: tw_id %s has None l_phids'
                                  % (p_id, tw_id))
            new_tw_pc_sum_vec = new_tw_pc_sum_vec / np.sum(new_tw_pc_sum_vec)
            new_tw_embed_sum_vec = new_tw_embed_sum_vec / embed_cnt
            d_new_tw_tw_ids[int_id][com_id] = [new_tw_pc_sum_vec.tolist(), new_tw_embed_sum_vec.tolist(), eff_tw_cnt]
        logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] Proc %s: new_tw_pc_sum_vec and new_tw_embed_sum_vec for %s is done in %s secs.'
                      % (p_id, int_id, time.time() - timer_start))

    with open(global_settings.g_tw_new_tw_time_int_pc_dists_and_embeds_int_file_format.format(str(p_id)), 'w+') as out_fd:
        json.dump(d_new_tw_tw_ids, out_fd)
        out_fd.close()
    logging.debug('[phrase_cluster_distributions_for_time_ints_single_proc] All done in %s sec.'
                  % str(time.time() - timer_start))


def phrase_cluster_distributions_and_embeds_for_time_ints_multiproc(n_clusters, num_procs, job_id):
    logging.debug('[phrase_cluster_distributions_for_time_ints_multiproc] Starts...')
    timer_start = time.time()

    l_task = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_new_tw_data_time_interval_folder):
        for filename in filenames:
            if re.fullmatch(r'com_\d+_new_tw_data_.*_\d{14}#\d{14}.pickle', filename) is None:
                continue
            l_fields = filename.split('_')
            com_id = l_fields[1]
            int_id = l_fields[6][:-7]
            l_task.append((int_id, com_id, dirpath + filename))
    num_tasks = len(l_task)
    logging.debug('[phrase_cluster_distributions_for_time_ints_multiproc] %s tasks.' % str(len(l_task)))

    batch_size = math.ceil(num_tasks / int(num_procs))
    l_batches = []
    for i in range(0, num_tasks, batch_size):
        l_batches.append(l_task[i:i + batch_size])
    logging.debug('[phrase_cluster_distributions_for_time_ints_multiproc] %s procs.' % len(l_batches))

    l_procs = []
    p_id = 0
    for batch in l_batches:
        t = multiprocessing.Process(target=phrase_cluster_distributions_and_embeds_for_time_ints_single_proc,
                                    args=(batch, n_clusters, str(n_clusters) + '_' + str(job_id) + '_' + str(p_id)))
        t.name = 't_mul_task_' + str(job_id) + '_' + str(p_id)
        t.start()
        l_procs.append(t)
        p_id += 1

    while len(l_procs) > 0:
        for p in l_procs:
            if p.is_alive():
                p.join(1)
            else:
                l_procs.remove(p)
                logging.debug('[phrase_cluster_distributions_for_time_ints_multiproc] Proc %s is finished.' % p.name)

    logging.debug('[phrase_cluster_distributions_for_time_ints_multiproc] All done in %s sec for %s tasks.'
                  % (time.time() - timer_start, len(l_batches)))


def phrase_cluster_distributions_and_embeds_for_time_ints_to_out(n_clusters):
    logging.debug('[phrase_cluster_distributions_and_embeds_for_time_ints_to_out] Starts...')
    l_int_dists_json = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_new_tw_time_int_pc_dists_int_folder):
        for filename in filenames:
            if filename[:40] != 'new_tw_time_int_pc_dists_and_embeds_int_' or filename[-5:] != '.json':
                continue
            with open(dirpath + filename, 'r') as in_fd:
                json_int_dists_n_embeds = json.load(in_fd)
                l_int_dists_json.append(json_int_dists_n_embeds)
                in_fd.close()

    l_ints = []
    l_com_ids = []
    for json_int in l_int_dists_json:
        l_ints += list(json_int.keys())
        for t_int in json_int:
            l_com_ids += list(json_int[t_int].keys())
    l_ints = sorted(list(set(l_ints)))
    l_com_ids = sorted([int(com_id) for com_id in set(l_com_ids)])
    logging.debug('[phrase_cluster_distributions_and_embeds_for_time_ints_to_out] %s time intervals and %s coms.'
                  % (len(l_ints), len(l_com_ids)))

    d_int_vecs = {t_int: [None] * len(l_com_ids) for t_int in l_ints}
    for json_int in l_int_dists_json:
        for t_int in json_int:
            for com_id in json_int[t_int]:
                new_tw_pc_dist = np.asarray(json_int[t_int][com_id][0], dtype=np.float32)
                new_tw_embed = np.asarray(json_int[t_int][com_id][1], dtype=np.float32)
                tw_cnt = int(json_int[t_int][com_id][2])
                com_id_idx = l_com_ids.index(int(com_id))
                d_int_vecs[t_int][com_id_idx] = (new_tw_pc_dist, new_tw_embed, tw_cnt)

    l_ready_recs = []
    for t_int in d_int_vecs:
        rec = (t_int,) + tuple(d_int_vecs[t_int])
        l_ready_recs.append(rec)
    df_out = pd.DataFrame(l_ready_recs, columns=['t_int'] + [str(com_id) for com_id in l_com_ids])
    df_out.to_pickle(global_settings.g_tw_new_tw_time_int_pc_dists_and_embeds_file_format.format(str(n_clusters)))
    logging.debug('[phrase_cluster_distributions_and_embeds_for_time_ints_to_out] Output g_tw_new_tw_time_int_pc_dists_and_embeds_file done.')


# def phrase_embeds_for_time_ints_to_out(embed_dim):
#     logging.debug('[phrase_embeds_for_time_ints_to_out] Starts...')
#     l_int_embeds_df = []
#     for (dirpath, dirname, filenames) in walk(global_settings.g_tw_new_tw_time_int_pc_dists_int_folder):
#         for filename in filenames:
#             if filename[:23] != 'new_tw_time_int_embeds_' or filename[-7:] != '.pickle':
#                 continue
#             df_embeds = pd.read_pickle(dirpath + filename)
#             l_int_embeds_df.append(df_embeds)
#     df_out = pd.concat(l_int_embeds_df)
#     df_out = df_out.sort_values(by='t_int')
#     df_out.to_pickle(global_settings.g_tw_new_tw_time_int_embeds_file_format.format(embed_dim))
#     logging.debug('[phrase_embeds_for_time_ints_to_out] Output g_tw_new_tw_time_int_embeds_file done.')


def get_community_usr_cnts():
    logging.debug('[get_community_usr_cnts] Starts...')
    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_sql_str = """select community, count(id) from cp4.sna group by community"""
    tw_db_cur.execute(tw_sql_str)
    l_sna_recs = tw_db_cur.fetchall()
    d_com_usr_cnt = dict()
    for sna_rec in l_sna_recs:
        com_id = sna_rec[0]
        usr_cnt = sna_rec[1]
        d_com_usr_cnt[com_id] = int(usr_cnt)
    tw_db_cur.close()
    tw_db_conn.close()

    with open(global_settings.g_tw_community_usr_cnts_file, 'w+') as out_fd:
        json.dump(d_com_usr_cnt, out_fd)
        out_fd.close()
    logging.debug('[get_community_usr_cnts] All done.')


def compare_phrase_cluster_distributions_for_time_ints(n_clusters):
    logging.debug('[compare_phrase_cluster_distributions_for_time_ints]')
    timer_start = time.time()

    with open(global_settings.g_tw_community_usr_cnts_file, 'r') as in_fd:
        d_com_usr_cnt = json.load(in_fd)
        in_fd.close()

    df_dist_n_embed = pd.read_pickle(global_settings.g_tw_new_tw_time_int_pc_dists_and_embeds_file_format.format(str(n_clusters)))
    l_com_ids = df_dist_n_embed.columns.tolist()[1:]
    l_dist_diff_mean_std = []
    l_embed_diff_mean_std = []
    d_tw_cnt_mean_std = dict()
    for com_id in l_com_ids:
        l_dist_diffs = []
        l_embed_diffs = []
        l_dist_n_embed = df_dist_n_embed[com_id].values
        if l_dist_n_embed is None:
            continue
        l_tw_cnts = [item[2] for item in l_dist_n_embed.tolist() if item is not None]
        for i in range(len(l_dist_n_embed) - 1):
            if l_dist_n_embed[i] is None:
                continue
            else:
                dist_vec_i, embed_vec_i, _ = l_dist_n_embed[i]
            j = i + 1
            if l_dist_n_embed[j] is None:
                continue
            else:
                dist_vec_j, embed_vec_j, _ = l_dist_n_embed[j]
            js = jensenshannon(dist_vec_i, dist_vec_j)
            cos = cosine(embed_vec_i, embed_vec_j)
            l_dist_diffs.append(js)
            l_embed_diffs.append(cos)
        if len(l_dist_diffs) <= 0 or len(l_embed_diffs) <= 0:
            continue
        dist_diff_mean = np.mean(l_dist_diffs)
        dist_diff_std = np.std(l_dist_diffs)
        l_dist_diff_mean_std.append((dist_diff_mean, dist_diff_std, len(l_dist_diffs), com_id))
        embed_diff_mean = np.mean(l_embed_diffs)
        embed_diff_std = np.std(l_embed_diffs)
        l_embed_diff_mean_std.append((embed_diff_mean, embed_diff_std, len(l_embed_diffs), com_id))
        tw_cnt_mean = np.mean(l_tw_cnts)
        tw_cnt_std = np.std(l_tw_cnts)
        d_tw_cnt_mean_std[com_id] = (tw_cnt_mean, tw_cnt_std)

    l_dist_diff_mean_std = sorted(l_dist_diff_mean_std, key=lambda k: k[2], reverse=True)
    l_embed_diff_mean_std = sorted(l_embed_diff_mean_std, key=lambda k: k[2], reverse=True)
    plt.figure(figsize=(40, 30))
    plt.subplot(5, 1, 1)
    plt.grid(True)
    plt.title("Mean and Std of JS-Div Between Consecutive Weeks (per Community)")
    plt.xticks(rotation=90)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.tight_layout()
    plt.errorbar([item[3] for item in l_dist_diff_mean_std],
                 [item[0] for item in l_dist_diff_mean_std],
                 [item[1] for item in l_dist_diff_mean_std],
                 fmt='o',
                 # ecolor='blue',
                 # markersize=5,
                 marker='o',
                 mfc='red',
                 capsize=2,
                 capthick=1)

    plt.subplot(5, 1, 2)
    plt.grid(True)
    plt.title("Mean and Std of Cosine Distance Between Consecutive Weeks (per Community)")
    plt.xticks(rotation=90)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.tight_layout()
    plt.errorbar([item[3] for item in l_embed_diff_mean_std],
                 [item[0] for item in l_embed_diff_mean_std],
                 [item[1] for item in l_embed_diff_mean_std],
                 fmt='o',
                 # ecolor='blue',
                 # markersize=5,
                 marker='o',
                 mfc='red',
                 capsize=2,
                 capthick=1)

    plt.subplot(5, 1, 3)
    plt.grid(True)
    plt.title("Num of Consecutive Weeks (per Community)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.stem([item[3] for item in l_dist_diff_mean_std], [item[2] for item in l_dist_diff_mean_std])

    plt.subplot(5, 1, 4)
    plt.grid(True)
    plt.title("Mean and Std of Effective New Tweet Counts (per Community)")
    plt.xticks(rotation=90)
    plt.yticks([i for i in range(0, 50000, 2000)])
    plt.tight_layout()
    plt.errorbar([item[3] for item in l_embed_diff_mean_std],
                 [d_tw_cnt_mean_std[item[3]][0] for item in l_dist_diff_mean_std],
                 [d_tw_cnt_mean_std[item[3]][1] for item in l_dist_diff_mean_std],
                 fmt='o',
                 # ecolor='blue',
                 # markersize=5,
                 marker='o',
                 mfc='red',
                 capsize=2,
                 capthick=1)

    plt.subplot(5, 1, 5)
    plt.grid(True)
    plt.title("User Counts (per Community)")
    plt.xticks(rotation=90)
    plt.yticks([i for i in range(0, 80000, 2000)])
    plt.tight_layout()
    plt.stem([item[3] for item in l_dist_diff_mean_std], [d_com_usr_cnt[item[3]] for item in l_dist_diff_mean_std])

    plt.show()
    print()


    # l_int_dist_pc_vecs = df_int_dists['pc_vec'].to_list()
    # num_ints = len(l_int_dist_pc_vecs)
    #
    # df_int_embeds = pd.read_pickle(global_settings.g_tw_new_tw_time_int_embeds_file_format.format(str(embed_dim)))
    # l_int_embeds = df_int_embeds['embed'].to_list()
    #
    # mat_dist_js = np.zeros((num_ints, num_ints), dtype=np.float32)
    # mat_embed_cos = np.zeros((num_ints, num_ints), dtype=np.float32)
    # for i in range(0, num_ints - 1):
    #     for j in range(i, num_ints):
    #         if i == j:
    #             mat_dist_js[i][j] == 0.0
    #             mat_embed_cos[i][j] == 0.0
    #         else:
    #             js = jensenshannon(l_int_dist_pc_vecs[i], l_int_dist_pc_vecs[j])
    #             mat_dist_js[i][j] = js
    #             mat_dist_js[j][i] = js
    #             cos = cosine(l_int_embeds[i], l_int_embeds[j])
    #             mat_embed_cos[i][j] = cos
    #             mat_embed_cos[j][i] = cos
    # logging.debug('[compare_phrase_cluster_distributions_for_time_ints] mat_dist_js and mat_embed_cos done in %s secs.'
    #               % str(time.time() - timer_start))

    # plt.subplot(1, 2, 1)
    # plt.imshow(mat_dist_js, cmap='hot')
    # plt.colorbar()
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(mat_embed_cos, cmap='hot')
    # plt.colorbar()
    #
    # # plt.tight_layout()
    # plt.show()
    # plt.clf()


def new_tw_time_series_by_com(t_start_str, t_end_str, day_delta):
    logging.debug('[new_tw_time_series_by_com] Starts...')
    timer_start = time.time()

    datetime_fmt = '%Y%m%d%H%M%S'
    t_start = datetime.strptime(t_start_str, datetime_fmt)
    t_end = datetime.strptime(t_end_str, datetime_fmt)
    num_ints = math.ceil(((t_end - t_start).days if t_start + timedelta(days=int((t_end - t_start).days)) >= t_end else (t_end - t_start).days + 1) / day_delta)
    l_ts = [t_start + timedelta(days=int_id * day_delta) for int_id in range(num_ints)]
    l_ts = [(int_s, int_s + timedelta(days=day_delta)) if int_s + timedelta(days=day_delta) <= t_end else (int_s, t_end) for int_s in l_ts]
    l_ts = [(datetime.strftime(item[0], datetime_fmt), datetime.strftime(item[1], datetime_fmt)) for item in l_ts]
    logging.debug('[new_tw_time_series_by_com] %s time intervals.' % str(len(l_ts)))

    df_new_tw_by_com = pd.read_pickle(global_settings.g_tw_new_tw_by_com_file)
    logging.debug('[new_tw_time_series_by_com] Load g_tw_new_tw_by_com_file with %s recs.' % str(len(df_new_tw_by_com)))

    l_ready_recs = []
    cnt = 0
    for idx1, new_tw_by_com_rec in df_new_tw_by_com.iterrows():
        com_id = new_tw_by_com_rec['com_id']
        df_tw = new_tw_by_com_rec['tw_dt']
        df_tw = df_tw.sort_values(by='tw_datetime')
        d_ts = {t_int: [] for t_int in l_ts}
        for idx2, tw_dt_rec in df_tw.iterrows():
            tw_id = tw_dt_rec['tw_id']
            usr_id = tw_dt_rec['usr_id']
            tw_datetime = tw_dt_rec['tw_datetime']
            if tw_datetime > t_end_str:
                break
            for int_s, int_e in l_ts:
                if int_s <= tw_datetime <= int_e:
                    d_ts[(int_s, int_e)].append((tw_id, usr_id, tw_datetime))
                    break
        l_ready_recs.append((com_id, d_ts))
        cnt += 1
        if cnt % 100 == 0 and cnt >= 100:
            logging.debug('[new_tw_time_series_by_com] %s communities done in %s secs.'
                          % (cnt, time.time() - timer_start))
    logging.debug('[new_tw_time_series_by_com] All %s communities done in %s secs.'
                  % (cnt, time.time() - timer_start))
    l_ready_recs = sorted(l_ready_recs, key=lambda k: sum([len(item) for item in k[1].values()]), reverse=True)
    df_out = pd.DataFrame(l_ready_recs, columns=['com_id', 'ts_data'])
    df_out.to_pickle(global_settings.g_tw_new_tw_time_series_data_by_com_file_format.
                     format(t_start_str + '#' + t_end_str + '#' + str(day_delta)))
    logging.debug('[new_tw_time_series_by_com] All done.')


def new_tw_time_series_pconehot_nar_stance_by_com(n_clusters, t_start_str, t_end_str, day_delta,
                                                  tw_nar_len, tw_stance_len):
    logging.debug('[new_tw_time_series_pconehot_nar_stance_by_com] Starts...')
    timer_start = time.time()

    df_tw_pc_onehot = pd.read_pickle(global_settings.g_tw_pc_onehot_file_format.format(str(n_clusters)))
    df_tw_pc_onehot = df_tw_pc_onehot.set_index('tw_id')
    logging.debug('[new_tw_time_series_pconehot_nar_stance_by_com] Load g_tw_pc_onehot_file with %s recs in %s secs.'
                  % (len(df_tw_pc_onehot), time.time() - timer_start))

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = """select trg_tw_id, trg_l_nars, trg_l_stances from cp4.mf3jh_udt_tw_src_trg_data 
    where trg_tw_type = 'n' and trg_tw_datetime >= '{0}' and trg_tw_datetime <= '{1}'""".format(t_start_str, t_end_str)
    tw_db_cur.execute(sql_str)
    l_tw_recs = tw_db_cur.fetchall()
    d_tw_nar_stance = dict()
    for tw_rec in l_tw_recs:
        tw_id = tw_rec[0]
        tw_l_nars = tw_rec[1]
        tw_l_stances = tw_rec[2]
        d_tw_nar_stance[tw_id] = (tw_l_nars, tw_l_stances)
    tw_db_cur.close()
    tw_db_conn.close()
    logging.debug('[new_tw_time_series_pconehot_nar_stance_by_com] Load %s tw nars and stances in %s secs.'
                  % (len(d_tw_nar_stance), time.time() - timer_start))

    df_new_tw_ts = pd.read_pickle(global_settings.g_tw_new_tw_time_series_data_by_com_file_format
                                  .format(t_start_str + '#' + t_end_str + '#' + str(day_delta)))
    logging.debug('[new_tw_time_series_pconehot_nar_stance_by_com] Load g_tw_new_tw_time_series_data_by_com_file with %s recs in %s secs.'
                  % (len(df_new_tw_ts), time.time() - timer_start))

    l_ready_recs = []
    cnt = 0
    for _, rec in df_new_tw_ts.iterrows():
        com_id = rec['com_id']
        d_ts = rec['ts_data']
        l_ts_pconehot_nar_stance_recs = []
        for t_int in d_ts:
            l_tw_recs = d_ts[t_int]
            int_pc_onehot = np.zeros(n_clusters, dtype=np.float32)
            int_nar_vec = np.zeros(tw_nar_len, dtype=np.float32)
            int_stance_vec = np.zeros(tw_stance_len, dtype=np.float32)
            stance_cnt = 0
            for tw_rec in l_tw_recs:
                tw_id = tw_rec[0]
                if tw_id in df_tw_pc_onehot.index:
                    tw_pc_onehot = df_tw_pc_onehot.loc[tw_id]['pc_onehot']
                    int_pc_onehot += tw_pc_onehot
                if tw_id in d_tw_nar_stance:
                    tw_l_nars, tw_l_stances = d_tw_nar_stance[tw_id]
                    if tw_l_nars is not None and len(tw_l_nars) != 0:
                        tw_nar_vec = np.asarray([1.0 if i in tw_l_nars else 0 for i in range(tw_nar_len)],
                                                dtype=np.float32)
                        int_nar_vec += tw_nar_vec
                    if tw_l_stances is not None and len(tw_l_stances) == tw_stance_len:
                        tw_stance_vec = np.asarray(tw_l_stances, dtype=np.float32)
                        int_stance_vec += tw_stance_vec
                        stance_cnt += 1
            sum_int_pc_onehot = sum(int_pc_onehot)
            sum_int_nar_vec = sum(int_nar_vec)
            if sum_int_pc_onehot > 0:
                int_pc_onehot = int_pc_onehot / sum_int_pc_onehot
            if sum_int_nar_vec > 0:
                int_nar_vec = int_nar_vec / sum_int_nar_vec
            if stance_cnt > 0:
                int_stance_vec = int_stance_vec / stance_cnt
            l_ts_pconehot_nar_stance_recs.append((t_int[0], t_int[1], int_pc_onehot, int_nar_vec, int_stance_vec))
        df_ts_pconehot_nar_stance_recs = pd.DataFrame(l_ts_pconehot_nar_stance_recs,
                                                      columns=['int_s', 'int_e', 'pcvec', 'nar', 'stance'])
            # d_ts[t_int] = (int_pc_onehot, int_nar_vec, int_stance_vec)
        l_ready_recs.append((com_id, df_ts_pconehot_nar_stance_recs))
        cnt += 1
        if cnt % 100 == 0 and cnt >= 100:
            logging.debug('[new_tw_time_series_pconehot_nar_stance_by_com] %s com pc_onehots done in %s secs.'
                          % (cnt, time.time() - timer_start))
    logging.debug('[new_tw_time_series_pconehot_nar_stance_by_com] All %s com pc_onehots done in %s secs.'
                  % (cnt, time.time() - timer_start))

    df_out = pd.DataFrame(l_ready_recs, columns=['com_id', 'df_ts_data'])
    df_out.to_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format.
                     format(t_start_str + '#' + t_end_str + '#' + str(day_delta)))
    logging.debug('[new_tw_time_series_pconehot_nar_stance_by_com] All done in %s secs.' % str(time.time() - timer_start))


def new_tw_gen_model_make_train_sets(new_tw_ts_name, pcvec_dim, max_ts_len):
    logging.debug('[new_tw_gen_model_make_train_sets] Starts...')

    df_new_tw_ts_by_com = pd.read_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format
                                         .format(new_tw_ts_name))
    # df_new_tw_ts_by_com = df_new_tw_ts_by_com.set_index('com_id')
    logging.debug('[new_tw_gen_model_make_train_sets] Load g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file with %s recs.'
                  % str(len(df_new_tw_ts_by_com)))

    df_event_info = pd.read_pickle(global_settings.g_events_info_file)
    # df_event_info = df_event_info.set_index('event_id')
    d_event_info = dict()
    for _, event_rec in df_event_info.iterrows():
        event_id = event_rec['event_id']
        pcvec = event_rec['pcvec']
        d_event_info[event_id] = pcvec
    logging.debug('[new_tw_gen_model_make_train_sets] Load g_events_info_file with %s recs.' % str(len(df_event_info)))

    d_ts_data_by_com = dict()
    for _, new_tw_by_com in df_new_tw_ts_by_com.iterrows():
        l_ts_new_tw_event = []
        com_id = new_tw_by_com['com_id']
        df_ts_data = new_tw_by_com['df_ts_data']
        for __, ts_rec in df_ts_data.iterrows():
            int_s = ts_rec['int_s']
            int_e = ts_rec['int_e']
            new_tw_pcvec = ts_rec['pcvec']
            if new_tw_pcvec is None:
                new_tw_pcvec = np.zeros(pcvec_dim, dtype=np.float32)
            pot_event_id = int_e[:8]
            if pot_event_id in d_event_info:
                event_pcvec = d_event_info[pot_event_id]
            else:
                event_pcvec = np.zeros(pcvec_dim, dtype=np.float32)
            l_ts_new_tw_event.append((int_s, new_tw_pcvec, event_pcvec))
        l_ts_new_tw_event = sorted(l_ts_new_tw_event, key=lambda item: item[0])
        d_ts_data_by_com[com_id] = l_ts_new_tw_event
    logging.debug('[new_tw_gen_model_make_train_sets] d_ts_data_by_com done.')

    l_train_set_by_com_recs = []
    for com_id in d_ts_data_by_com:
        l_ts_new_tw_event = d_ts_data_by_com[com_id]
        l_train_recs = []
        skip_cnt = 0
        for ts_len in range(2, max_ts_len + 1):
            for i in range(len(l_ts_new_tw_event) - ts_len + 1):
                ts_data = l_ts_new_tw_event[i:i + ts_len]
                gt_rec = ts_data[-1][1]
                if np.count_nonzero(gt_rec) == 0:
                    skip_cnt += 1
                    continue
                gt_rec = copy.deepcopy(ts_data[-1][1])
                input_recs = [(item[1], item[2]) for item in ts_data]
                input_recs[-1] = (np.zeros(pcvec_dim, dtype=np.float32), ts_data[-1][2])
                l_train_recs.append((input_recs, gt_rec))
        df_train_recs = pd.DataFrame(l_train_recs, columns=['inputs', 'gt'])
        l_train_set_by_com_recs.append((com_id, df_train_recs))
    df_train_set = pd.DataFrame(l_train_set_by_com_recs, columns=['com_id', 'df_train_recs'])
    df_train_set.to_pickle(global_settings.g_tw_new_tw_model_train_sets)
    logging.debug('[new_tw_gen_model_make_train_sets] All done.')


def new_tw_gen_model_make_test_sets(new_tw_ts_1_name, new_tw_ts_2_name, event_info_1_name, event_info_2_name,
                                    init_t_start, init_t_end, pred_t_start, pred_t_end, ts_len, num_pred):
    logging.debug('[new_tw_gen_model_make_test_sets] Starts...')

    d_new_tw_by_com = dict()
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_new_tw_folder):
        for filename in filenames:
            if filename[:37] != 'new_tw_ts_pconehot_nar_stance_by_com_' or filename[-7:] != '.pickle':
                continue
            df_new_tw = pd.read_pickle(dirpath + filename)
            for _, new_tw_rec in df_new_tw.iterrows():
                com_id = new_tw_rec['com_id']
                df_ts_data = new_tw_rec['df_ts_data']
                df_ts_data = df_ts_data.drop([len(df_ts_data) - 1])
                if com_id not in d_new_tw_by_com:
                    d_new_tw_by_com[com_id] = df_ts_data
                else:
                    d_new_tw_by_com[com_id].append(df_ts_data)


    df_new_tw_ts_1 = pd.read_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format.format(new_tw_ts_1_name))
    df_new_tw_ts_1 = df_new_tw_ts_1.set_index('com_id')
    logging.debug('[new_tw_gen_model_make_test_sets] Load df_new_tw_ts_1 with %s recs.'
                  % str(len(df_new_tw_ts_1)))
    df_new_tw_ts_2 = pd.read_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format.format(new_tw_ts_2_name))
    df_new_tw_ts_2 = df_new_tw_ts_2.set_index('com_id')
    logging.debug('[new_tw_gen_model_make_test_sets] Load df_new_tw_ts_2 with %s recs.'
                  % str(len(df_new_tw_ts_2)))

    df_event_info_1 = pd.read_pickle(global_settings.g_events_info_file_format.format(event_info_1_name))
    logging.debug('[new_tw_gen_model_make_test_sets] Load df_event_info_1 with %s recs.' % str(len(df_event_info_1)))
    df_event_info_2 = pd.read_pickle(global_settings.g_events_info_file_format.format(event_info_2_name))
    logging.debug('[new_tw_gen_model_make_test_sets] Load df_event_info_1 with %s recs.' % str(len(df_event_info_2)))

    for com_id, new_tw_recs_by_com in df_new_tw_ts_1.iterrows():
        l_init_pcvec = []
        df_ts_data_1 = new_tw_recs_by_com['df_ts_data']
        for _, ts_rec in df_ts_data_1.iterrows():
            int_s = ts_rec['int_s']
            pcvec = ts_rec['pcvec']
            if init_t_start <= int_s <= init_t_end:
                l_init_pcvec.append(pcvec)
            elif int_s > init_t_end:
                break
        df_ts_data_2 = df_new_tw_ts_2.loc[com_id]['df_ts_data']





def new_tw_gen_model(is_train, d1_d, d2_d, lstm_d, d3_d, skip_com_ids=None, infer_inputs=None, infer_com_id=None):
    import tensorflow as tf

    def scaled_dot_product_attention(q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def scale_vec(input_vec, scale_factor):
        input_vec_scaled = ((input_vec - np.min(input_vec)) / (np.max(input_vec) - np.min(input_vec))) * 2 * scale_factor - scale_factor
        input_vec_tan = np.tan(input_vec_scaled * np.pi / (2 * (scale_factor + 0.01)))
        return input_vec_tan

    class NewTwGen_LSTMCELL(tf.keras.layers.LSTMCell):
        def __init__(self, *args, **kwargs):
            d1_d = kwargs['d1_d']
            d2_d = kwargs['d2_d']
            del kwargs['d1_d']
            del kwargs['d2_d']
            super(NewTwGen_LSTMCELL, self).__init__(*args, **kwargs)
            self.h_e_weights = tf.Variable([[1., 1.]], dtype=tf.float32)
            self.dense_1 = tf.keras.layers.Dense(d1_d, activation='selu')
            self.dense_2 = tf.keras.layers.Dense(d2_d, activation='selu')
            # self.cnt = 0

        def build(self, input_shape):
            tf.keras.layers.LSTMCell.build(self, (input_shape[0], int(input_shape[1] / 2)))

        def call(self, inputs, states, training=None):
            input_vec, ext_vec = tf.split(inputs, num_or_size_splits=2, axis=1)
            d1_out = self.dense_1(input_vec)
            d2_out = self.dense_2(ext_vec)
            ext_outputs, [ext_hidden_states, ext_cell_states] = tf.keras.layers.LSTMCell.call(self, inputs=d2_out, states=states)
            input_outputs, [input_hidden_states, input_cell_states] = tf.keras.layers.LSTMCell.call(self, inputs=d1_out, states=states)
            forward_hidden_states = tf.matmul(self.h_e_weights, [ext_hidden_states[0], input_hidden_states[0]])
            return (input_outputs, [forward_hidden_states, input_cell_states])
            # input_outputs, [input_hidden_states, input_cell_states] = tf.keras.layers.LSTMCell.call(self, inputs=input_vec, states=states)
            # return (input_outputs, [input_hidden_states, input_cell_states])

    class NewTwGen(tf.keras.Model):
        def __init__(self, d1_d, d2_d, lstm_d, d3_d):
            super(NewTwGen, self).__init__()
            self.lstmcell = NewTwGen_LSTMCELL(lstm_d, d1_d=d1_d, d2_d=d2_d)
            self.lstm = tf.keras.layers.RNN(self.lstmcell)
            # return_sequences=True,
            # return_state=True)
            self.dense_3 = tf.keras.layers.Dense(d3_d, activation='sigmoid')

        def call(self, inputs, training=False):
            # input_vec, ext_vec = tf.split(inputs, num_or_size_splits=2, axis=1)
            # lstm_inputs = tf.concat([input_vec, ext_vec], -1)
            lstm_out = self.lstm(inputs)
            d3_out = self.dense_3(lstm_out)
            ret = tf.math.softmax(d3_out)
            return ret

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

    def load_all_train_test_sets():
        logging.debug('[load_all_train_test_sets] Starts...')
        df_all_new_tw_train_sets = pd.read_pickle(global_settings.g_tw_new_tw_model_train_sets)
        logging.debug('[load_all_train_test_sets] Load g_tw_new_tw_model_train_sets with %s com train sets.'
                      % str(len(df_all_new_tw_train_sets)))
        df_all_new_tw_train_sets = df_all_new_tw_train_sets.set_index('com_id')
        return df_all_new_tw_train_sets

    def load_train_set_by_com(com_id, df_all_new_tw_train_sets, batch_size, strategy):
        df_train_recs = df_all_new_tw_train_sets.loc[com_id]['df_train_recs']
        inputs = df_train_recs['inputs'].values.tolist()
        gt = df_train_recs['gt'].values.tolist()
        # input_set = tf.data.Dataset.from_tensor_slices((inputs, gt))
        # input_set = input_set.shuffle(len(df_train_recs))
        # input_set = input_set.batch(batch_size)
        # input_set = input_set.prefetch(tf.data.experimental.AUTOTUNE)
        # input_set = strategy.experimental_distribute_dataset(input_set)
        input_set = zip(inputs, gt)
        return input_set

    def train_step(new_tw_gen_ins, ts_idx, new_tw_hist, true_new_tw, optimizer, loss_obj):
        with tf.GradientTape() as gt:
            new_tw_hist = [tf.concat(input_tup, axis=-1) for input_tup in new_tw_hist]
            new_tw_hist = tf.stack(new_tw_hist)
            new_tw_hist = tf.reshape(new_tw_hist, (1, new_tw_hist.shape[0], new_tw_hist.shape[1]))
            pred_new_tw = new_tw_gen_ins(new_tw_hist, training=True)
            loss = loss_obj(true_new_tw, pred_new_tw)
            grads = gt.gradient(loss, new_tw_gen_ins.trainable_variables)
            optimizer.apply_gradients(zip(grads, new_tw_gen_ins.trainable_variables))
        return loss

    # def test_step(user_belief_ins, src_ce, trg_ce, loss_obj):
    #     pred_trg_ce = resp_infer(user_belief_ins, src_ce)
    #     loss = loss_obj(trg_ce, pred_trg_ce)
    #     return loss

    # @tf.function
    def train_distributed(d1_d, d2_d, lstm_d, d3_d, com_id, opt_ins, epoch_cnt, loss_obj, train_loss_met, test_loss_met,
                          batch_size, df_all_new_tw_train_sets):
        # TODO
        # [Strategy modification]
        # [WARNING!] strategy has a very significant overhead when running on Rivanna.
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        new_tw_gen_ins = NewTwGen(d1_d, d2_d, lstm_d, d3_d)
        for epoch in range(epoch_cnt):
            '''Training'''
            train_loss_met.reset_states()
            train_set = load_train_set_by_com(com_id, df_all_new_tw_train_sets, batch_size, strategy)
            cnt = 0
            for ts_idx, (src_ce, trg_ce) in enumerate(train_set):
                # TODO
                # [Strategy modification]
                loss = strategy.run(train_step, args=(new_tw_gen_ins, ts_idx, src_ce, trg_ce, opt_ins, loss_obj))
                # loss = train_step(input_vecs, embed_ae, opt_ins, loss_obj, loss_obj_2)
                train_loss_met(loss)
                cnt += 1
            logging.debug('[train_distributed] Com %s Epoch:%s Rec:%s Training done. Loss:%s Time:%s'
                  % (com_id, epoch, cnt, train_loss_met.result().numpy(), time.time() - timer_start))

            # '''Testing'''
            # test_loss_met.reset_states()
            # cnt = 0
            # for src_ce, trg_ce in test_set:
            #     test_loss = strategy.run(test_step, args=(user_belief_ins, src_ce, trg_ce, loss_obj))
            #     test_loss_met(test_loss)
            #     cnt += 1
            # print('[train_distributed] Epoch:%s Rec:%s Testing done. Loss:%s Time:%s'
            #       % (epoch, cnt, test_loss_met.result().numpy(), time.time() - timer_start))
        logging.debug('[train_distributed] Com %s All done. Train loss:%s Time:%s'
              % (com_id, train_loss_met.result().numpy(), time.time() - timer_start))
        return new_tw_gen_ins

    def new_tw_infer(new_tw_gen_ins, inputs):
        pred_new_tw = new_tw_gen_ins(inputs, training=False)
        return pred_new_tw


    '''Function starts here'''
    if is_train:
        df_all_new_tw_train_sets = load_all_train_test_sets()
        l_com_ids = list(df_all_new_tw_train_sets.index)
        if skip_com_ids is not None:
            l_com_ids = [com_id for com_id in l_com_ids if int(com_id) not in skip_com_ids]
        # new_tw_gen_ins = NewTwGen(d1_d, d2_d, lstm_d, d3_d)
        learn_rate = CustomSchedule(d1_d)
        opt_ins = tf.keras.optimizers.Adam(learn_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # opt_ins = tf.keras.optimizers.SGD(learning_rate=learn_rate, momentum=0.9)

        batch_size = 1
        epoch_cnt = 1000
        # ts_id = 'all_' + str(n_clusters)
        # ckpt_fmt = '_{0}'

        timer_start = time.time()
        loss_obj = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        train_loss_met = tf.keras.metrics.Mean()
        test_loss_met = tf.keras.metrics.Mean()

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            for com_id in l_com_ids:
                new_tw_gen_ins = train_distributed(d1_d, d2_d, lstm_d, d3_d, com_id, opt_ins, epoch_cnt, loss_obj,
                                                   train_loss_met, test_loss_met, batch_size, df_all_new_tw_train_sets)
                save_model_path = global_settings.g_tw_new_tw_model_folder + 'new_tw_com_' + str(com_id) + '/'
                if not os.path.exists(save_model_path):
                    os.mkdir(save_model_path)
                tf.saved_model.save(new_tw_gen_ins, save_model_path)
                logging.debug('[new_tw_gen_model] new tw gen model for com %s is saved.' % str(com_id))
    else:
        save_model_path = global_settings.g_tw_new_tw_model_folder + 'new_tw_com_' + str(infer_com_id) + '/'
        new_tw_gen_ins = tf.saved_model.load(save_model_path)
        pred_new_tw = new_tw_infer(new_tw_gen_ins, infer_inputs)
        return pred_new_tw


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'new_tw_data':
        # extract_new_tw_data()
        # collect_new_tws_by_community()
        day_delta = '1'
        dt_start = '20190131235959'
        dt_end = '20190208000000'
        # com_id = '100'
        # df_new_tw_data = pd.read_pickle(global_settings.g_tw_new_tw_data)
        # split_new_tw_data(df_new_tw_data, com_id, day_delta, dt_start, dt_end)
        # split_new_tw_data_for_communities(day_delta, dt_start, dt_end)
        new_tw_time_series_by_com(dt_start, dt_end, int(day_delta))
        tw_nar_len = 48
        tw_stance_len = 3
        new_tw_time_series_pconehot_nar_stance_by_com(global_settings.g_num_phrase_clusters,
                                                      dt_start,
                                                      dt_end,
                                                      int(day_delta),
                                                      tw_nar_len,
                                                      tw_stance_len)

    elif cmd == 'time_int_diff':
        # num_procs = sys.argv[2]
        # job_id = sys.argv[3]
        # phrase_cluster_distributions_and_embeds_for_time_ints_multiproc(global_settings.g_num_phrase_clusters,
        #                                                                 num_procs,
        #                                                                 job_id)
        # phrase_cluster_distributions_and_embeds_for_time_ints_to_out(global_settings.g_num_phrase_clusters)
        # get_community_usr_cnts()
        compare_phrase_cluster_distributions_for_time_ints(global_settings.g_num_phrase_clusters)

    elif cmd == 'new_tw_gen':
        # new_tw_ts_name = '20181223235959#20190201000000#1'
        # pcvec_dim = global_settings.g_num_phrase_clusters
        # max_ts_len = 7
        # new_tw_gen_model_make_train_sets(new_tw_ts_name, pcvec_dim, max_ts_len)
        is_train = True
        d1_d = 150
        d2_d = 150
        lstm_d = 150
        d3_d = 150
        skip_com_ids = [1, 1285, 1286, 1287, 1288, 1289, 1290, 1294, 151, 153, 155, 156, 159, 160, 164, 165, 162, 437,
                        438, 5, 6, 604, 605, 606, 609, 624, 9]
        new_tw_gen_model(is_train, d1_d, d2_d, lstm_d, d3_d, skip_com_ids, infer_inputs=None, infer_com_id=None)

    elif cmd == 'new_tw_infer':
        is_train = False
        d1_d = 150
        d2_d = 150
        lstm_d = 150
        d3_d = 150
        infer_com_id = 1285
        new_tw_gen_model(is_train, d1_d, d2_d, lstm_d, d3_d, infer_inputs=None, infer_com_id=infer_com_id)