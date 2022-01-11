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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.mplot3d.art3d as art3d
import scipy.stats as stats
from sklearn.preprocessing import normalize
import psycopg2
from scipy.spatial.distance import jensenshannon

import global_settings


g_com_file_folder = '/scratch/mf3jh/data/Sammple_Communities_29July2020/'
g_com_file_table_list = [('cp4.comm_1404_nodes', 'Community_1404_nodes.csv'),
                         ('cp4.comm_1404_edges', 'Community_1404_edges.csv'),
                         ('cp4.comm_151_nodes', 'Community_151_nodes.csv'),
                         ('cp4.comm_151_edges', 'Community_151_edges.csv'),
                         ('cp4.partisan_nodes', 'partisan_nodes_29July2020.csv'),
                         ('cp4.partisan_edges', 'partisan_edges_29July2020.csv')]
g_com_table_schema = \
    {
        'cp4.comm_1404_nodes': """create table if not exists cp4.comm_1404_nodes (pajek_id integer primary key, 
                                                                                  labels char(22), 
                                                                                  community_member integer, 
                                                                                  community_degree integer)""",
        'cp4.comm_1404_edges': """create table if not exists cp4.comm_1404_edges (sender integer, 
                                                                                  recver integer, 
                                                                                  weight real, 
                                                                                  community_members integer,
                                                                                  primary key (sender, recver))""",
        'cp4.comm_151_nodes': """create table if not exists cp4.comm_151_nodes (pajek_id integer primary key, 
                                                                                labels char(22), 
                                                                                community_degree integer,
                                                                                community_kcore integer,
                                                                                community_member integer,
                                                                                inner_core integer
                                                                                )""",
        'cp4.comm_151_edges': """create table if not exists cp4.comm_151_edges (sender integer, 
                                                                                recver integer, 
                                                                                weight real, 
                                                                                community_members integer,
                                                                                core_network integer,
                                                                                primary key (sender, recver))""",
        'cp4.partisan_nodes': """create table if not exists cp4.partisan_nodes (pajek_id integer primary key,
                                                                                id integer,
                                                                                labels char(22),
                                                                                community integer,
                                                                                proval real,
                                                                                stance integer,
                                                                                bridge integer,
                                                                                core integer)""",
        'cp4.partisan_edges': """create table if not exists cp4.partisan_edges (sender integer, 
                                                                                recver integer, 
                                                                                weight real,
                                                                                primary key (sender, recver))"""
    }
g_table_insert_fmt = \
    {
        'cp4.comm_1404_nodes': """insert into cp4.comm_1404_nodes (pajek_id, labels, community_member, community_degree) 
                                  values (%s,%s,%s,%s)""",
        'cp4.comm_1404_edges': """insert into cp4.comm_1404_edges (sender, recver, weight, community_members) 
                                  values (%s,%s,%s,%s)""",
        'cp4.comm_151_nodes': """insert into cp4.comm_151_nodes (pajek_id, labels, community_degree, community_kcore,
                                 community_member, inner_core) values (%s,%s,%s,%s,%s,%s)""",
        'cp4.comm_151_edges': """insert into cp4.comm_151_edges (sender, recver, weight, community_members, core_network) 
                                 values (%s,%s,%s,%s,%s)""",
        'cp4.partisan_nodes': """insert into cp4.partisan_nodes (pajek_id, id, labels, community, proval, stance, 
                                 bridge, core) values (%s,%s,%s,%s,%s,%s,%s,%s)""",
        'cp4.partisan_edges': """insert into cp4.partisan_edges (sender, recver, weight) values (%s,%s,%s)"""
    }
g_table_field_type = \
    {
        'cp4.comm_1404_nodes': [int, str, int, int],
        'cp4.comm_1404_edges': [int, int, float, int],
        'cp4.comm_151_nodes': [int, str, int, int, int, int],
        'cp4.comm_151_edges': [int, int, float, int, int],
        'cp4.partisan_nodes': [int, int, str, int, float, int, int, int],
        'cp4.partisan_edges': [int, int, float]
    }


def com_file_to_db():
    logging.debug('[com_file_to_db] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()

    for table_data in g_com_file_table_list:
        table_name = table_data[0]
        data_file = g_com_file_folder + table_data[1]
        sql_str = g_com_table_schema[table_name]
        tw_db_cur.execute(sql_str)
        tw_db_conn.commit()
        logging.debug('[com_file_to_db] create table %s' % table_name)

        sql_str = g_table_insert_fmt[table_name]
        cnt = 0
        with open(data_file, 'r') as in_fd:
            csv_reader = csv.reader(in_fd, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    l_field_types = g_table_field_type[table_name]
                    l_vals = []
                    for i in range(len(l_field_types)):
                        l_vals.append(l_field_types[i](row[i]))
                    l_vals = tuple(l_vals)
                    tw_db_cur.execute(sql_str, l_vals)
                    cnt += 1
            in_fd.close()
        tw_db_conn.commit()
        logging.debug('[com_file_to_db] %s with %s recs is committed in %s secs.'
                      % (table_name, cnt, time.time() - timer_start))

    tw_db_cur.close()
    tw_db_conn.close()
    logging.debug('[com_file_to_db] All done in %s secs.' % str(time.time() - timer_start))


def stance_stats():
    logging.debug('[stance_stats] Starts...')

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = """select trg_l_stances from cp4.mf3jh_udt_tw_src_trg_data"""
    tw_db_cur.execute(sql_str)

    l_recs = tw_db_cur.fetchall()
    cnt_total = 0
    cnt_am = 0
    cnt_pm = 0
    cnt_un = 0
    for rec in l_recs:
        cnt_total += 1
        if rec is None or rec[0] is None:
            continue
        am_score = rec[0][0]
        pm_score = rec[0][1]
        un_score = rec[0][2]
        if am_score >= 0.5 and pm_score < 0.5 and un_score < 0.5:
            cnt_am += 1
        elif pm_score >= 0.5 and am_score < 0.5 and un_score < 0.5:
            cnt_pm += 1
        elif un_score >= 0.5 and am_score < 0.5 and pm_score < 0.5:
            cnt_un += 1
    tw_db_cur.close()
    tw_db_conn.close()
    logging.debug('[stance_stats] cnt_total=%s, cnt_am=%s, cnt_pm=%s, cnt_un=%s' % (cnt_total, cnt_am, cnt_pm, cnt_un))


def compare_new_tw_ts_to_events(com_id, df_com_new_tw_ts):
    logging.debug('[compare_new_tw_ts_to_events] Starts...')
    timer_start = time.time()

    df_event_info = pd.read_pickle(global_settings.g_events_info_file)
    df_event_info = df_event_info.sort_values('event_id')
    # df_event_info = df_event_info.set_index('event_id')
    logging.debug('[compare_new_tw_ts_to_events] Load %s events in %s secs.'
                  % (len(df_event_info), time.time() - timer_start))

    d_comp_rets = dict()
    for _, event_rec in df_event_info.iterrows():
        event_id = event_rec['event_id']
        event_pcvec = event_rec['pcvec']
        event_nar = event_rec['nar']
        event_stance = event_rec['stance']
        l_rets = []
        for com_new_tw_ts_id, new_tw_ts_rec in df_com_new_tw_ts.iterrows():
            int_s = new_tw_ts_rec['int_s']
            if event_id > int_s[:8]:
                l_rets.append((int_s, -0.1, -0.1))
                continue
            int_e = new_tw_ts_rec['int_e']
            new_tw_pcvec = new_tw_ts_rec['pcvec']
            new_tw_nar = new_tw_ts_rec['nar']
            new_tw_stance = new_tw_ts_rec['stance']
            if np.count_nonzero(event_pcvec) == 0 or np.count_nonzero(new_tw_pcvec) == 0:
                js_pcvec = -0.1
            else:
                js_pcvec = jensenshannon(event_pcvec, new_tw_pcvec)
            if not np.isfinite(js_pcvec):
                raise Exception('[compare_new_tw_ts_to_events] Invalid js_pcvec for event %s and int_s %s.'
                                % (event_id, int_s))
            if np.count_nonzero(event_nar) == 0 or np.count_nonzero(new_tw_nar) == 0:
                js_nar = -0.1
            else:
                js_nar = jensenshannon(event_nar, new_tw_nar)
            if not np.isfinite(js_nar):
                raise Exception('[compare_new_tw_ts_to_events] Invalid js_nar for event %s and int_s %s.'
                                % (event_id, int_s))
            l_rets.append((int_s, js_pcvec, js_nar))
        d_comp_rets[event_id] = l_rets
    logging.debug('[compare_new_tw_ts_to_events] All done for %s' % str(com_id))
    return d_comp_rets


def pairwise_compare_new_tw_ts(com_id, df_com_new_tw_ts):
    logging.debug('[pairwise_compare_new_tw_ts] Starts...')

    dim = max(df_com_new_tw_ts.index) + 1
    mat_pcvec_js = np.zeros((dim, dim))
    mat_nar_js = np.zeros((dim, dim))
    l_ints = df_com_new_tw_ts['int_s']
    for i in range(dim - 1):
        new_tw_ts_rec_i = df_com_new_tw_ts.iloc[i]
        new_tw_pcvec_i = new_tw_ts_rec_i['pcvec']
        new_tw_nar_i = new_tw_ts_rec_i['nar']
        for j in range(i + 1, dim):
            new_tw_ts_rec_j = df_com_new_tw_ts.iloc[j]
            new_tw_pcvec_j = new_tw_ts_rec_j['pcvec']
            new_tw_nar_j = new_tw_ts_rec_j['nar']

            if np.count_nonzero(new_tw_pcvec_i) == 0 or np.count_nonzero(new_tw_pcvec_j) == 0:
                js_pcvec = -0.1
            else:
                js_pcvec = jensenshannon(new_tw_pcvec_i, new_tw_pcvec_j)
            mat_pcvec_js[i][j] = js_pcvec
            mat_pcvec_js[j][i] = js_pcvec

            if np.count_nonzero(new_tw_nar_i) == 0 or np.count_nonzero(new_tw_nar_j) == 0:
                js_nar = -0.1
            else:
                js_nar = jensenshannon(new_tw_nar_i, new_tw_nar_j)
            mat_nar_js[i][j] = js_nar
            mat_nar_js[j][i] = js_nar
    logging.debug('[pairwise_compare_new_tw_ts] All done for %s' % str(com_id))
    return l_ints, mat_pcvec_js, mat_nar_js


def new_tw_ts_by_com_analysis_single_proc(l_tasks, new_tw_ts_data_name, p_id):
    logging.debug('[new_tw_ts_by_com_analysis_single_proc] Proc %s: Starts with %s tasks...' % (p_id, len(l_tasks)))

    df_new_tw_ts_data = \
        pd.read_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format.
                       format(new_tw_ts_data_name))
    logging.debug('[new_tw_ts_by_com_analysis_single_proc] Proc %s: Load g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file for %s with %s recs.' %
                  (p_id, new_tw_ts_data_name, len(df_new_tw_ts_data)))

    df_new_tw_ts_data = df_new_tw_ts_data.set_index('com_id')
    for com_id in l_tasks:
    # for com_id in df_new_tw_ts_data.index:
        df_com_new_tw_ts = df_new_tw_ts_data.loc[com_id]['df_ts_data']
        d_comp_rets = compare_new_tw_ts_to_events(com_id, df_com_new_tw_ts)
        l_pw_ints, mat_pcvec_js, mat_nar_js = pairwise_compare_new_tw_ts(com_id, df_com_new_tw_ts)

        num_fig = len(d_comp_rets) * 2
        # l_fig_w = [4] * (len(d_comp_rets) * 2) + [1, 1]
        l_fig_w = [5]
        l_fig_h = [1] * (len(d_comp_rets) * 2)
        gs_kw = dict(width_ratios=l_fig_w, height_ratios=l_fig_h)
        fig, axes = plt.subplots(ncols=1, nrows=num_fig, gridspec_kw=gs_kw, figsize=(20, 150))
        l_event_ids = sorted(list(d_comp_rets.keys()))
        for i in range(len(l_event_ids)):
            event_id = l_event_ids[i]
            l_event_comp_rets = d_comp_rets[event_id]
            l_ints = [ret[0] for ret in l_event_comp_rets]
            l_event_pcvec_js = [ret[1] for ret in l_event_comp_rets]
            l_event_nar_js = [ret[2] for ret in l_event_comp_rets]

            fig_id = i * 2
            axes[fig_id].grid(True)
            axes[fig_id].set_title("Event %s vs New Tweet Time Series of Community %s: Phrase Space Embedding JS-Div"
                                   % (event_id, com_id), fontsize=20)
            axes[fig_id].set_xticks([k for k in range(len(l_ints))])
            axes[fig_id].set_xticklabels(l_ints, rotation=45)
            axes[fig_id].set_ylim(-0.15, 1.0)
            axes[fig_id].stem(l_event_pcvec_js, use_line_collection=True)

            fig_id += 1
            axes[fig_id].grid(True)
            axes[fig_id].set_title("Event %s vs New Tweet Time Series of Community %s: Narrative Vector JS-Div"
                                   % (event_id, com_id), fontsize=20)
            axes[fig_id].set_xticks([k for k in range(len(l_ints))])
            axes[fig_id].set_xticklabels(l_ints, rotation=45)
            axes[fig_id].set_ylim(-0.15, 1.0)
            axes[fig_id].stem(l_event_nar_js, use_line_collection=True)

        # plt.subplots_adjust(hspace=0.5)
        # plt.autoscale(enable=True)
        plt.tight_layout(pad=3.0)
        plt.savefig(global_settings.g_community_analysis_folder + 'new_tw_ts_vs_events_by_com_%s.png'
                    % str(com_id), format='PNG')
        plt.clf()
        plt.close()
        logging.debug('[new_tw_ts_by_com_analysis_single_proc] Proc %s: new_tw_ts_vs_events_by_com for %s done.'
                      % (p_id, com_id))

        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(15, 20))
        fig_id = 0
        axes[fig_id].grid(False)
        axes[fig_id].set_title("Community %s New Tweet Time Series Pairwise Phrase Space Embedding JS-Div"
                               % str(com_id), fontsize=20)
        axes[fig_id].set_xticks([k for k in range(len(l_pw_ints))])
        axes[fig_id].set_xticklabels(l_pw_ints, rotation=45)
        axes[fig_id].set_yticks([k for k in range(len(l_pw_ints))])
        axes[fig_id].set_yticklabels(l_pw_ints)
        pos = axes[fig_id].imshow(mat_pcvec_js, vmin=-0.15, vmax=1.0, cmap='hot')
        divider = make_axes_locatable(axes[fig_id])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(pos, ax=axes[fig_id], cax=cax)

        fig_id += 1
        axes[fig_id].grid(False)
        axes[fig_id].set_title("Community %s New Tweet Time Series Pairwise Narrative Vector JS-Div"
                               % str(com_id), fontsize=20)
        axes[fig_id].set_xticks([k for k in range(len(l_pw_ints))])
        axes[fig_id].set_xticklabels(l_pw_ints, rotation=45)
        axes[fig_id].set_yticks([k for k in range(len(l_pw_ints))])
        axes[fig_id].set_yticklabels(l_pw_ints)
        pos = axes[fig_id].imshow(mat_nar_js, vmin=-0.15, vmax=1.0, cmap='hot')
        divider = make_axes_locatable(axes[fig_id])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(pos, ax=axes[fig_id], cax=cax)

        # plt.subplots_adjust(hspace=0.5)
        # plt.autoscale(enable=True)
        plt.tight_layout(pad=3.0)
        plt.savefig(global_settings.g_community_analysis_folder + 'new_tw_ts_pw_by_com_%s.png' % str(com_id), format='PNG')
        plt.clf()
        plt.close()
        logging.debug('[new_tw_ts_by_com_analysis_single_proc] Proc %s: Figure for %s done.' % (p_id, com_id))

    logging.debug('[new_tw_ts_by_com_analysis_single_proc] Proc %s: All done.' % str(p_id))


def new_tw_ts_by_com_analysis_multiproc(num_procs, new_tw_ts_data_name):
    logging.debug('[new_tw_ts_by_com_analysis_multiproc] Starts...')
    timer_start = time.time()

    df_new_tw_ts_data = \
        pd.read_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format.
                       format(new_tw_ts_data_name))
    df_new_tw_ts_data = df_new_tw_ts_data.set_index('com_id')
    num_tasks = len(df_new_tw_ts_data.index)
    logging.debug('[new_tw_ts_by_com_analysis_multiproc] Load g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file for %s with %s recs.' %
                  (new_tw_ts_data_name, len(df_new_tw_ts_data)))

    l_task = list(df_new_tw_ts_data.index)
    batch_size = math.ceil(num_tasks / int(num_procs))
    l_batches = []
    for i in range(0, num_tasks, batch_size):
        l_batches.append(l_task[i:i + batch_size])
    logging.debug('[new_tw_ts_by_com_analysis_multiproc] %s procs.' % len(l_batches))

    l_procs = []
    p_id = 0
    for batch in l_batches:
        t = multiprocessing.Process(target=new_tw_ts_by_com_analysis_single_proc,
                                    args=(batch, new_tw_ts_data_name, str(p_id)))
        t.name = 't_mul_task_' + str(p_id)
        t.start()
        l_procs.append(t)
        p_id += 1

    while len(l_procs) > 0:
        for p in l_procs:
            if p.is_alive():
                p.join(1)
            else:
                l_procs.remove(p)
                logging.debug('[new_tw_ts_by_com_analysis_multiproc] Proc %s is finished.' % p.name)

    logging.debug('[new_tw_ts_by_com_analysis_multiproc] All done in %s sec for %s tasks.'
                  % (time.time() - timer_start, len(l_batches)))


def pairwise_compare_communities():
    logging.debug('[pairwise_compare_communities] Starts...')
    df_new_tw_ts_data = \
        pd.read_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format.
                       format(new_tw_ts_data_name))
    d_com_ts_data = dict()
    for idx, com_ts_data in df_new_tw_ts_data.iterrows():
        com_id = com_ts_data['com_id']
        df_com_ts_data = com_ts_data['df_ts_data']
        d_com_int_pcvec_nar = dict()
        for _, com_int_data in df_com_ts_data.iterrows():
            int_s = com_int_data['int_s']
            int_pcvec = com_int_data['pcvec']
            int_nar = com_int_data['nar']
            d_com_int_pcvec_nar[int_s] = (int_pcvec, int_nar)
        d_com_ts_data[com_id] = d_com_int_pcvec_nar
    l_tasks = list(df_new_tw_ts_data['com_id'])
    num_tasks = len(l_tasks)
    logging.debug('[pairwise_compare_communities] Load g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file for %s with %s recs.' %
                  (new_tw_ts_data_name, len(df_new_tw_ts_data)))

    l_int_starts = list(df_new_tw_ts_data.iloc[0]['df_ts_data']['int_s'])

    l_int_com_comp = []
    for int_s in l_int_starts:
        mat_js_pcvec = np.zeros((num_tasks, num_tasks), dtype=np.float32)
        mat_js_nar = np.zeros((num_tasks, num_tasks), dtype=np.float32)
        for i in range(num_tasks - 1):
            com_pcvec_i = d_com_ts_data[l_tasks[i]][int_s][0]
            com_nar_i = d_com_ts_data[l_tasks[i]][int_s][1]
            # df_com_int_data_i = df_new_tw_ts_data.loc[df_new_tw_ts_data.index[i]]['df_ts_data']
            # com_pcvec_i = df_com_int_data_i.loc[int_s]['pcvec'][0]
            # com_nar_i = df_com_int_data_i.loc[int_s]['nar'][0]
            for j in range(i, num_tasks):
                com_pcvec_j = d_com_ts_data[l_tasks[j]][int_s][0]
                com_nar_j = d_com_ts_data[l_tasks[j]][int_s][1]
                # df_com_int_data_j = df_new_tw_ts_data.loc[df_new_tw_ts_data.index[j]]['df_ts_data']
                # com_pcvec_j = df_com_int_data_j.loc[int_s]['pcvec'][0]
                # com_nar_j = df_com_int_data_j.loc[int_s]['nar'][0]
                if i == j:
                    js_pcvec = 0.0
                    js_nar = 0.0
                else:
                    if np.count_nonzero(com_pcvec_i) == 0 or np.count_nonzero(com_pcvec_j) == 0:
                        js_pcvec = -0.1
                    else:
                        js_pcvec = jensenshannon(com_pcvec_i, com_pcvec_j)
                    if np.count_nonzero(com_nar_i) == 0 or np.count_nonzero(com_nar_j) == 0:
                        js_nar = -0.1
                    else:
                        js_nar = jensenshannon(com_nar_i, com_nar_j)
                mat_js_pcvec[i][j] = js_pcvec
                mat_js_pcvec[j][i] = js_pcvec
                mat_js_nar[i][j] = js_nar
                mat_js_nar[j][i] = js_nar
        l_int_com_comp.append((int_s, mat_js_pcvec, mat_js_nar))
    logging.debug('[pairwise_compare_communities] l_int_com_comp done.')

    df_com_comp = pd.DataFrame(l_int_com_comp, columns=['int_s', 'mat_pcvec_js', 'mat_nar_js'])
    df_com_comp.to_pickle(global_settings.g_new_tw_ts_pw_com_comparison_data_format.format(new_tw_ts_data_name))
    logging.debug('[pairwise_compare_communities] All done.')


def draw_pairwise_compare_communities(new_tw_ts_data_name, sub_len):
    logging.debug('[draw_pairwise_compare_communities] Starts...')

    df_com_comp = pd.read_pickle(global_settings.g_new_tw_ts_pw_com_comparison_data_format.
                                 format(dt_start, dt_end, str(day_delta)))
    logging.debug('[draw_pairwise_compare_communities] Load g_new_tw_ts_pw_com_comparison_data with %s recs.'
                  % str(len(df_com_comp)))

    df_new_tw_ts_data = \
        pd.read_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format.
                       format(new_tw_ts_data_name))
    l_coms = list(df_new_tw_ts_data['com_id'])[:sub_len]
    num_coms = len(l_coms)
    logging.debug('[draw_pairwise_compare_communities] Load g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file with %s recs.'
                  % str(len(df_new_tw_ts_data)))

    for idx, int_com_comp in df_com_comp.iterrows():
        int_s = int_com_comp[0]
        mat_js_pcvec = int_com_comp[1][:sub_len, :sub_len]
        mat_js_nar = int_com_comp[2][:sub_len, :sub_len]

        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(35, 60))

        fig_id = 0
        axes[fig_id].grid(False)
        axes[fig_id].set_title("Time Interval %s Community New Tweet Time Series Pairwise Phrase Space Embedding JS-Div"
                               % str(int_s), fontsize=20)
        axes[fig_id].set_xticks([k for k in range(num_coms)])
        axes[fig_id].set_xticklabels(l_coms, rotation=45)
        axes[fig_id].set_yticks([k for k in range(num_coms)])
        axes[fig_id].set_yticklabels(l_coms)
        pos = axes[fig_id].imshow(mat_js_pcvec, vmin=-0.15, vmax=1.0, cmap='hot')
        divider = make_axes_locatable(axes[fig_id])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(pos, ax=axes[fig_id], cax=cax)

        fig_id += 1
        axes[fig_id].grid(False)
        axes[fig_id].set_title("Time Interval %s Community New Tweet Time Series Pairwise Narrative Vector JS-Div"
                               % str(int_s), fontsize=20)
        axes[fig_id].set_xticks([k for k in range(num_coms)])
        axes[fig_id].set_xticklabels(l_coms, rotation=45)
        axes[fig_id].set_yticks([k for k in range(num_coms)])
        axes[fig_id].set_yticklabels(l_coms)
        pos = axes[fig_id].imshow(mat_js_nar, vmin=-0.15, vmax=1.0, cmap='hot')
        divider = make_axes_locatable(axes[fig_id])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(pos, ax=axes[fig_id], cax=cax)

        plt.tight_layout(pad=3.0)
        plt.savefig(global_settings.g_community_analysis_folder + 'new_tw_ts_com_pw_comparisons_{0}_{1}.png'
                    .format(new_tw_ts_data_name, int_s), format='PNG')
        plt.clf()
        plt.close()
        logging.debug('[pairwise_compare_communities] %s is done.' % int_s)
    logging.debug('[pairwise_compare_communities] All done.')


def find_corelated_communities(new_tw_ts_data_name, sub_len, pcvec_js_upbd, nar_js_upbd):
    logging.debug('[find_corelated_communities] Starts...')

    df_com_comp = pd.read_pickle(global_settings.g_new_tw_ts_pw_com_comparison_data_format.
                                 format(dt_start, dt_end, str(day_delta)))
    logging.debug('[find_corelated_communities] Load g_new_tw_ts_pw_com_comparison_data with %s recs.'
                  % str(len(df_com_comp)))

    df_new_tw_ts_data = \
        pd.read_pickle(global_settings.g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format.
                       format(new_tw_ts_data_name))
    l_coms = list(df_new_tw_ts_data['com_id'])[:sub_len]
    num_coms = len(l_coms)
    logging.debug('[find_corelated_communities] Load g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file with %s recs.'
                  % str(len(df_new_tw_ts_data)))

    tensor_pcvec_js = np.stack(list(df_com_comp['mat_pcvec_js'].values))[:, :sub_len, :sub_len]
    tensor_nar_js = np.stack(list(df_com_comp['mat_nar_js'].values))[:, :sub_len, :sub_len]

    l_pcvec_js_stats = []
    l_nar_js_stats = []
    conf_max = 0
    for i in range(num_coms - 1):
        com_i = l_coms[i]
        for j in range(i + 1, num_coms):
            com_j = l_coms[j]
            l_eff_pcvec_js = np.asarray([ele for ele in tensor_pcvec_js[:, i, j] if ele >= 0])
            l_eff_nar_js = np.asarray([ele for ele in tensor_nar_js[:, i, j] if ele >= 0])
            if len(l_eff_pcvec_js) > 0:
                mean_pcvec_js = np.mean(l_eff_pcvec_js)
                std_pcvec_js = np.std(l_eff_pcvec_js)
                l_pcvec_js_stats.append((str(com_i) + '#' + str(com_j), mean_pcvec_js, std_pcvec_js, len(l_eff_pcvec_js)))
                if len(l_eff_pcvec_js) > conf_max:
                    conf_max = len(l_eff_pcvec_js)
            if len(l_eff_nar_js) > 0:
                mean_nar_js = np.mean(l_eff_nar_js)
                std_nar_js = np.std(l_eff_nar_js)
                l_nar_js_stats.append((str(com_i) + '#' + str(com_j), mean_nar_js, std_nar_js, len(l_eff_nar_js)))
                if len(l_nar_js_stats) > conf_max:
                    conf_max = len(l_nar_js_stats)

    l_pcvec_js_stats = [ele for ele in l_pcvec_js_stats if ele[1] <= pcvec_js_upbd]
    l_nar_js_stats = [ele for ele in l_nar_js_stats if ele[1] <= nar_js_upbd]

    l_pcvec_js = sorted(l_pcvec_js_stats, key=lambda item: item[1])
    l_nar_js = sorted(l_nar_js_stats, key=lambda item: item[1])

    l_intersect = []
    d_nar_js = {item[0]: (item[1], item[2], item[3]) for item in l_nar_js}
    for item in l_pcvec_js:
        if item[0] in d_nar_js:
            l_intersect.append((item[0], item[1], item[2], item[3], d_nar_js[item[0]][0], d_nar_js[item[0]][1], d_nar_js[item[0]][2]))

    graph_sim = nx.Graph()
    for item in l_intersect:
        com_pair = item[0]
        com_fields = com_pair.split('#')
        com_i = com_fields[0]
        com_j = com_fields[1]
        avg_pcvec_js = item[1]
        avg_nar_js = item[2]
        weight = (avg_pcvec_js + avg_nar_js) / 2.0
        graph_sim.add_edge(com_i, com_j, weight=weight, avg_pcvec_js=avg_pcvec_js, avg_nar_js=avg_nar_js)

    fig_suffix = new_tw_ts_data_name + '#' + str(sub_len) + '#' + str(pcvec_js_upbd) + '#' + str(nar_js_upbd)
    fig, axes = plt.subplots(ncols=1, nrows=6, figsize=(60, 30))
    fig_id = 0
    axes[fig_id].grid(True)
    axes[fig_id].set_title("Avg. Community New Tweet Time Series Pairwise Phrase Space Embedding JS-Div <= {0}"
                           .format(str(pcvec_js_upbd)), fontsize=20)
    axes[fig_id].set_xticks([k for k in range(len(l_pcvec_js))])
    axes[fig_id].set_xticklabels([item[0] for item in l_pcvec_js], rotation=90)
    # axes[fig_id].set_ylim(0, pcvec_js_upbd + 0.05)
    # markerline, stemlines, baselineaxes = axes[fig_id].stem([item[1] for item in l_pcvec_js], use_line_collection=True)
    # markerline.set_markerfacecolor('b')
    axes[fig_id].errorbar([k for k in range(len(l_pcvec_js))],
                          [item[1] for item in l_pcvec_js],
                          [item[2] for item in l_pcvec_js],
                          fmt='o',
                          # ecolor='blue',
                          # markersize=5,
                          marker='o',
                          mfc='b',
                          capsize=2,
                          capthick=1)

    fig_id = 1
    axes[fig_id].grid(True)
    axes[fig_id].set_title("Support for Above", fontsize=20)
    axes[fig_id].set_xticks([k for k in range(len(l_pcvec_js))])
    axes[fig_id].set_xticklabels([item[0] for item in l_pcvec_js], rotation=90)
    # axes[fig_id].set_ylim(0, pcvec_js_upbd + 0.05)
    markerline, stemlines, baselineaxes = axes[fig_id].stem([item[3] for item in l_pcvec_js], use_line_collection=True)
    markerline.set_markerfacecolor('b')
    # axes[fig_id].errorbar([k for k in range(len(l_pcvec_js))],
    #                       [item[1] for item in l_pcvec_js],
    #                       [item[2] for item in l_pcvec_js],
    #                       fmt='o',
    #                       # ecolor='blue',
    #                       # markersize=5,
    #                       marker='o',
    #                       mfc='b',
    #                       capsize=2,
    #                       capthick=1)

    fig_id = 2
    axes[fig_id].grid(True)
    axes[fig_id].set_title("Avg. Community New Tweet Time Series Pairwise Narrative Vector JS-Div <= {0}"
                           .format(str(nar_js_upbd)), fontsize=20)
    axes[fig_id].set_xticks([k for k in range(len(l_nar_js))])
    axes[fig_id].set_xticklabels([item[0] for item in l_nar_js], rotation=90)
    # axes[fig_id].set_ylim(0, nar_js_upbd + 0.05)
    # markerline, stemlines, baselineaxes = axes[fig_id].stem([item[1] for item in l_nar_js], use_line_collection=True)
    # markerline.set_markerfacecolor('r')
    axes[fig_id].errorbar([k for k in range(len(l_nar_js))],
                          [item[1] for item in l_nar_js],
                          [item[2] for item in l_nar_js],
                          fmt='o',
                          # ecolor='blue',
                          # markersize=5,
                          marker='o',
                          mfc='r',
                          capsize=2,
                          capthick=1)

    fig_id = 3
    axes[fig_id].grid(True)
    axes[fig_id].set_title("Support for Above", fontsize=20)
    axes[fig_id].set_xticks([k for k in range(len(l_nar_js))])
    axes[fig_id].set_xticklabels([item[0] for item in l_nar_js], rotation=90)
    # axes[fig_id].set_ylim(0, pcvec_js_upbd + 0.05)
    markerline, stemlines, baselineaxes = axes[fig_id].stem([item[3] for item in l_nar_js], use_line_collection=True)
    markerline.set_markerfacecolor('b')

    fig_id = 4
    axes[fig_id].grid(True)
    axes[fig_id].set_title("Intersection of Above Community Pairs", fontsize=20)
    axes[fig_id].set_xticks([k for k in range(len(l_intersect))])
    axes[fig_id].set_xticklabels([item[0] for item in l_intersect], rotation=90)
    # axes[fig_id].set_ylim(0, max(pcvec_js_upbd, nar_js_upbd) + 0.05)
    # markerline, stemlines, baselineaxes = axes[fig_id].stem([item[1] for item in l_intersect], use_line_collection=True)
    # markerline.set_markerfacecolor('b')
    # markerline, stemlines, baselineaxes = axes[fig_id].stem([item[2] for item in l_intersect], use_line_collection=True)
    # markerline.set_markerfacecolor('r')
    axes[fig_id].errorbar([k for k in range(len(l_intersect))],
                          [item[1] for item in l_intersect],
                          [item[2] for item in l_intersect],
                          fmt='o',
                          # ecolor='blue',
                          # markersize=5,
                          marker='o',
                          mfc='b',
                          capsize=2,
                          capthick=1)
    axes[fig_id].errorbar([k for k in range(len(l_intersect))],
                          [item[4] for item in l_intersect],
                          [item[5] for item in l_intersect],
                          fmt='o',
                          # ecolor='blue',
                          # markersize=5,
                          marker='o',
                          mfc='r',
                          capsize=2,
                          capthick=1)

    fig_id = 5
    axes[fig_id].grid(True)
    axes[fig_id].set_title("Support for Above", fontsize=20)
    axes[fig_id].set_xticks([k for k in range(len(l_intersect))])
    axes[fig_id].set_xticklabels([item[0] for item in l_intersect], rotation=90)
    # axes[fig_id].set_ylim(0, pcvec_js_upbd + 0.05)
    markerline, stemlines, baselineaxes = axes[fig_id].stem([item[3] for item in l_intersect], use_line_collection=True)
    markerline.set_markerfacecolor('b')
    markerline, stemlines, baselineaxes = axes[fig_id].stem([item[6] for item in l_intersect], use_line_collection=True)
    markerline.set_markerfacecolor('r')

    plt.tight_layout(pad=3.0)
    plt.savefig(global_settings.g_community_analysis_folder + 'new_tw_ts_avg_com_pw_comparisons_{0}.png'
                .format(fig_suffix), format='PNG')
    plt.clf()
    plt.close()

    plt.figure(1, figsize=(50, 50), tight_layout={'pad': 1, 'w_pad': 50, 'h_pad': 50, 'rect': None})
    pos = nx.spring_layout(graph_sim, k=0.8)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.10
    plt.xlim(x_min - x_margin, x_max + x_margin)
    d_node_labels = {node[0]: node[0] for node in graph_sim.nodes(data=True)}
    nx.draw_networkx_nodes(graph_sim, pos, node_size=20)
    nx.draw_networkx_labels(graph_sim, pos, labels=d_node_labels, font_size=30, font_color='r', font_weight='semibold')
    num_edges = graph_sim.number_of_edges()
    edge_colors = range(2, num_edges + 2)
    l_edges = graph_sim.edges()
    drawn_edges = nx.draw_networkx_edges(graph_sim,
                                         pos,
                                         edgelist=l_edges,
                                         width=4,
                                         edge_color=edge_colors,
                                         edge_cmap=plt.get_cmap('Blues'),
                                         arrows=True,
                                         arrowsize=40)
    plt.savefig(global_settings.g_community_analysis_folder + 'new_tw_ts_avg_com_sim_graph_{0}.png'
                .format(fig_suffix), format="PNG")
    plt.clf()
    plt.close()

    logging.debug('[find_corelated_communities] All done.')


def baseline_js():
    l_means = np.random.rand(150)
    l_std = np.random.rand(150)
    l_samples = []
    for k in range(1000):
        sample = []
        for i in range(150):
            # sample.append(np.random.normal(l_means[i], l_std[i], 1)[0])
            sample.append(np.random.rand(1))
        sample = np.asarray(sample, dtype=np.float32)
        # sample = (sample - max(sample)) / (max(sample) - min(sample))
        sample = sample / np.sum(sample)
        l_samples.append(sample)

    l_js = []
    for i in range(999):
        sample_i = l_samples[i]
        for j in range(i+1, 1000):
            sample_j = l_samples[j]
            js = jensenshannon(sample_i, sample_j)
            l_js.append(js)

    print(np.mean(l_js), np.std(l_js))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'upload':
        com_file_to_db()

    elif cmd == 'stats':
        stance_stats()

    elif cmd == 'com_analysis':
        nar_len = 48
        com_id = 1285
        day_delta = '1'
        dt_start = '20181223235959'
        dt_end = '20190201000000'
        new_tw_ts_data_name = dt_start + '#' + dt_end + '#' + day_delta
        # num_procs = sys.argv[2]
        # new_tw_ts_by_com_analysis_multiproc(num_procs, new_tw_ts_data_name)
        # pairwise_compare_communities(new_tw_ts_data_name)
        sub_len = 200
        pcvec_js_upbd = 0.5
        nar_js_upbd = 0.5
        # draw_pairwise_compare_communities(new_tw_ts_data_name, sub_len)
        find_corelated_communities(new_tw_ts_data_name, sub_len, pcvec_js_upbd, nar_js_upbd)

    elif cmd == 'test':
        baseline_js()