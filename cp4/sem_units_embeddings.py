import logging
import sys
import os
from os import walk, path
import sqlite3
import json
import math
import time
import threading
import multiprocessing
import random

import numpy as np
import networkx as nx
import psycopg2
# import sshtunnel

import global_settings

sys.path.insert(1, global_settings.g_lexvec_model_folder)
import model as lexvec
from pygsp import graphs, filters

g_lexvec_model = None
g_embedding_len = 300


def load_lexvec_model():
    global g_lexvec_model, g_embedding_len
    # model_file = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.300d.W+C.pos.vectors'
    # g_lexvec_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    g_lexvec_model = lexvec.Model(global_settings.g_lexvec_vect_file_path)
    g_embedding_len = len(g_lexvec_model.word_rep('the'))
    logging.debug('[load_lexvec_model] The length of embeddings is %s' % g_embedding_len)


def phrase_embedding(phrase_str):
    '''
    Take a phrase (i.e. a sequence of tokens connected by whitespaces) and compute an embedding for it.
    This function acts as a wrapper of word embeddings. Refactor this function to adapt various word embedding models.
    :param
        phrase_str: An input phrase
    :return:
        An embedding for the phrase
    '''
    if global_settings.g_word_embedding_model == 'lexvec':
        if g_lexvec_model is None:
            raise Exception('g_lexvec_model is not loaded!')
        phrase_vec = np.zeros(g_embedding_len)
        l_words = [word.strip().lower() for word in phrase_str.split(' ')]
        for word in l_words:
            word_vec = g_lexvec_model.word_rep(word)
            phrase_vec += word_vec
        return phrase_vec


def avg_noun_phrases_vect(l_nps):
    if l_nps is None or len(l_nps) == 0:
        return np.zeros(g_embedding_len)
    else:
        l_np_vects = []
        for noun_phrase in l_nps:
            np_vect = phrase_embedding(noun_phrase.strip())
            l_np_vects.append(np_vect)
        return sum(l_np_vects) / len(l_np_vects)


def weighted_avg_cls_edge_vect(cls_graph):
    if cls_graph is None or len(cls_graph.nodes()) <= 0:
        return np.zeros(g_embedding_len)
    l_edge_vects = []
    for comp in nx.connected_components(cls_graph):
        sub_cls_graph = nx.subgraph(cls_graph, comp)
        if len(sub_cls_graph.nodes()) == 1:
            single_node = list(sub_cls_graph.nodes())[0]
            single_node_weight = 1.0
            single_node_fields = single_node.split('|')
            single_node_txt = single_node_fields[-1].strip()
            single_node_vect = phrase_embedding(single_node_txt)
            l_edge_vects.append(single_node_vect * single_node_weight)
            continue
        for edge in sub_cls_graph.edges():
            node_1 = edge[0]
            node_2 = edge[1]
            node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
            node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
            if node_1_pos + '_' + node_2_pos in global_settings.g_d_pos_pair_weights:
                node_1_fields = node_1.split('|')
                node_2_fields = node_2.split('|')
                node_1_txt = node_1_fields[-1].strip()
                node_2_txt = node_2_fields[-1].strip()
                node_1_vect = phrase_embedding(node_1_txt)
                node_2_vect = phrase_embedding(node_2_txt)
                edge_vect = (node_1_vect + node_2_vect) / 2
                w_edge_vect = edge_vect * global_settings.g_d_pos_pair_weights[node_1_pos + '_' + node_2_pos]
                l_edge_vects.append(w_edge_vect)
    if len(l_edge_vects) > 0:
        return sum(l_edge_vects) / len(l_edge_vects)
    else:
        return np.zeros(g_embedding_len)


def spectral_weighted_avg_cls_edge_vect(cls_graph):
    if cls_graph is None or len(cls_graph.nodes()) <= 0:
        return np.zeros(g_embedding_len)
    l_nodes, gsp_graph = build_gsp_graph(cls_graph)
    l_node_vects = []
    for node in l_nodes:
        node_fields = node.split('|')
        node_txt = node_fields[-1].strip()
        node_vect = phrase_embedding(node_txt)
        l_node_vects.append(node_vect)
    if gsp_graph.Ne == 0:
        return sum(l_node_vects) / len(l_node_vects)
    l_spectral_node_vects = []
    for node_idx, node in enumerate(l_nodes):
        pulse_sig = build_gsp_pulse_signal(l_nodes, node_idx)
        filtered_sig = heat_kernal_filtering(gsp_graph, pulse_sig)
        spectral_node_vect = np.zeros(g_embedding_len)
        for sig_idx, sig_val in enumerate(filtered_sig):
            if np.isfinite(sig_val):
                spectral_node_vect += sig_val * l_node_vects[sig_idx]
        spectral_node_vect = spectral_node_vect / len(filtered_sig)
        spectral_node_vect = (spectral_node_vect + l_node_vects[node_idx]) / 2
        l_spectral_node_vects.append(spectral_node_vect)
    l_edge_vects = []
    for comp in nx.connected_components(cls_graph):
        sub_cls_graph = nx.subgraph(cls_graph, comp)
        if len(sub_cls_graph.nodes()) == 1:
            single_node = list(sub_cls_graph.nodes())[0]
            single_node_weight = 1.0
            single_node_vect = l_spectral_node_vects[l_nodes.index(single_node)]
            l_edge_vects.append(single_node_vect * single_node_weight)
        for edge in sub_cls_graph.edges():
            node_1 = edge[0]
            node_2 = edge[1]
            node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
            node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
            if node_1_pos + '_' + node_2_pos in global_settings.g_d_pos_pair_weights:
                node_1_vect = l_spectral_node_vects[l_nodes.index(node_1)]
                node_2_vect = l_spectral_node_vects[l_nodes.index(node_2)]
                edge_vect = (node_1_vect + node_2_vect) / 2
                w_edge_vect = edge_vect * global_settings.g_d_pos_pair_weights[node_1_pos + '_' + node_2_pos]
                l_edge_vects.append(w_edge_vect)
    if len(l_edge_vects) > 0:
        ret_vect = sum(l_edge_vects) / len(l_edge_vects)
        if not np.isfinite(ret_vect).all():
            raise Exception('Invalid doc vect!')
        return ret_vect
    else:
        return np.zeros(g_embedding_len)


def build_gsp_graph(cls_graph):
    '''
    Convert a NetworkX core clause structure graph to a GSP graph.
    :param
        cls_graph: An core clause structure graph.
    :return:
        (l_nodes, gsp_graph): l_nodes is the node list which will be used to build signals on vertices, and gsp_graph
        is the GSP graph.
    '''
    for edge in cls_graph.edges():
        node_1 = edge[0]
        node_2 = edge[1]
        node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
        node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
        if node_1_pos + '_' + node_2_pos in global_settings.g_d_pos_pair_weights:
            edge_weight = global_settings.g_d_pos_pair_weights[node_1_pos + '_' + node_2_pos]
        else:
            edge_weight = 0.0
        cls_graph.edges[node_1, node_2]['weight'] = edge_weight
    l_nodes = list(cls_graph.nodes())
    adj_mat = nx.adjacency_matrix(cls_graph, l_nodes, weight='weight')
    gsp_graph = graphs.Graph(adj_mat, lap_type='normalized')
    gsp_graph.compute_laplacian(lap_type='normalized')
    gsp_graph.compute_fourier_basis()
    if gsp_graph.N > 1:
        gsp_graph.compute_differential_operator()
        gsp_graph.estimate_lmax()
    # else:
    #     raise Exception('[build_gsp_graph] cls_graph has no or only one node.')
    return l_nodes, gsp_graph


def build_gsp_pulse_signal(l_nodes, pulse_idx):
    signal = np.zeros(len(l_nodes))
    signal[pulse_idx] = 1.0
    return signal


def heat_kernal_filtering(gsp_graph, signal):
    # TODO
    # We may need to fine tune the parameter for the filter. So far we have observed that at 5 the filtered signal
    # has shown the evenly distributed heat values, and at 10 the values at somewhere else may be higher than the
    # pulse spot. So we set to 1, 3, 5.
    if gsp_graph.N == 1:
        return signal
    heat_kernel_filter = filters.Heat(gsp_graph, [1, 3, 5])
    filtered_signals = heat_kernel_filter.filter(signal, method='exact')
    ret_signal = filtered_signals[:, 0] + filtered_signals[:, 1] + filtered_signals[:, 2]
    if not np.isfinite(ret_signal).any():
        raise Exception('[heat_kernal_filtering] ret_signal is invalid.')
    return ret_signal


def doc_embedding(cls_graph, l_nps, embed_type, doc_id=None, doc_sim_db_cur=None):
    '''
    Take a core clause structure for a doc, and compute an embedding for the doc.
    Options:
    1. 'avg_edge_vect':
    Take an edge, compute the average vector of the two ends of this edge as the edge embedding. Then, compute the
    average vector of all edge embeddings as the embedding for the doc.
    2. 's_avg_edge_vect':
    We only consider a subset of types of edges (e.g. VERB_NOUN and NOUN_NOUN). Then, for these selected types of edges,
    we do 1.
    3. 'w_avg_edge_vect':
    We consider the weights of different types of edges listed in g_d_pos_pair_weights (i.e. for other edge types,
    the weights are all zero). Then, for these edges, we compute a weighted average vector as the embedding for the doc.
    4. 'p_avg_edge_vect':
    We utilize graph spectral analysis tricks to compute node embeddings. Then, we compute weighted average for the doc.
    NOTE:
    Ruling out single nodes doesn't help at all. It would worsen the performance.
    Weighting single NOUN and VERB nodes doesn't help at all. Worsen the performance.
    Vector concatenation doesn't work so well. Though, the POS pair weights do matter a lot.
    :param
        cls_graph: A core clause structure of a doc.
        embed_type: ['avg_node_vect'|'avg_edge_vect'|'s_avg_edge_vect'|'w_avg_edge_vect'|'sp_w_avg_edge_vect'|
                    'avg_nps_vect'|'avg_nps_vect_w_avg_edge_vect'|'avg_nps_vect_sp_w_avg_edge_vect']
        l_nps: The list of noun phrases.
    :return:
        A vector for the doc.
    '''
    if (cls_graph is None or len(cls_graph.nodes()) <= 0) and (l_nps is None or len(l_nps) <= 0):
        logging.debug('Trivial text exists!')
        return np.zeros(g_embedding_len)

    if embed_type == 'avg_node_vect':
        l_node_vects = []
        for node in cls_graph.nodes():
            node_fields = node.split('|')
            node_txt = node_fields[-1].strip()
            node_vect = phrase_embedding(node_txt)
            l_node_vects.append(node_vect)
        return sum(l_node_vects) / len(l_node_vects)
    elif embed_type == 'w_avg_edge_vect':
        return weighted_avg_cls_edge_vect(cls_graph)
    elif embed_type == 'sp_w_avg_edge_vect':
        return spectral_weighted_avg_cls_edge_vect(cls_graph)
    elif embed_type == 'avg_nps_vect':
        return avg_noun_phrases_vect(l_nps)
    elif embed_type == 'avg_nps_vect_w_avg_edge_vect':
        avg_nps_vect = avg_noun_phrases_vect(l_nps)
        w_avg_edge_vect = weighted_avg_cls_edge_vect(cls_graph)
        return avg_nps_vect + w_avg_edge_vect
    elif embed_type == 'avg_nps_vect_sp_w_avg_edge_vect':
        avg_nps_vect = avg_noun_phrases_vect(l_nps)
        sp_w_avg_edge_vect = spectral_weighted_avg_cls_edge_vect(cls_graph)
        return avg_nps_vect + sp_w_avg_edge_vect
    else:
        raise Exception('Unsupported embedding type %s.' % embed_type)


def tw_embeddings_tasks(num_jobs):
    tw_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = '''select tw_id from ven_tw_en'''
    tw_db_cur.execute(sql_str)
    l_tw_ids = [rec[0] for rec in tw_db_cur.fetchall()]
    batch_size = math.ceil(len(l_tw_ids) / num_jobs)
    l_jobs = []
    for i in range(0, len(l_tw_ids), batch_size):
        if i + batch_size < len(l_tw_ids):
            l_jobs.append(l_tw_ids[i:i + batch_size])
        else:
            l_jobs.append(l_tw_ids[i:])
    for idx, job in enumerate(l_jobs):
        with open(global_settings.g_tw_embed_task_file_format.format(str(idx)), 'w+') as out_fd:
            out_str = '\n'.join(job)
            out_fd.write(out_str)
            out_fd.close()
    print('[tw_embeddings_tasks] Job partitioning is done.')
    # return l_tw_ids


def tw_embeddings_retweet_makeup_tasks():
    tw_embed_db_conn = sqlite3.connect(global_settings.g_tw_embed_db)
    tw_embed_db_cur = tw_embed_db_conn.cursor()
    sql_str = '''select tw_id from ven_tw_en_embeds where avg_nps_vect is null'''
    tw_embed_db_cur.execute(sql_str)
    l_tw_ids = [rec[0] for rec in tw_embed_db_cur.fetchall()]
    return l_tw_ids


# def retrieve_tw_topic_vect_single_thread(l_tasks, t_id, output_folder):
# logging.debug('[retrieve_tw_topic_vect_single_thread] %s starts...' % t_id)
# out_fd = open(output_folder + 'topic_vect_{0}_embeds.txt'.format(t_id), 'w+')
# for data_file in global_settings.g_tw_raw_data_file_list:
#     data_path = global_settings.g_tw_raw_data_folder + data_file
#     with open(data_path, 'r') as in_fd:
#         for tw_ln in in_fd:
#             tw_json = json.loads(tw_ln)
#             tw_vect = np.zeros(g_embedding_len)
#             if ('topic_vector_en' not in tw_json['extension']) \
#                     and ('topic_vector_es' not in tw_json['extension']):
#                 continue
#             if ('topic_vector_en' in tw_json['extension']) \
#                     and ('topic_vector_es' in tw_json['extension']) \
#                     and (tw_json['extension']['topic_vector_en'] is None) \
#                     and (tw_json['extension']['topic_vector_es'] is None):
#                 continue


def compute_tw_embeddings_single_thread(l_tasks, l_embed_types, t_id, output_folder):
    logging.debug('[compute_tw_embeddings_single_thread] %s starts...' % t_id)
    timer_start = time.time()

    su_db_conn = sqlite3.connect(global_settings.g_tw_sem_units_db)
    su_db_cur = su_db_conn.cursor()
    sql_str = '''select cls_json_str, nps_str from ven_tw_sem_units where tw_id = ?'''

    d_out_fds = dict()
    for embed_type in l_embed_types:
        out_fd = open(output_folder + '{0}_{1}_embeds.txt'.format(embed_type, t_id), 'w+')
        d_out_fds[embed_type] = out_fd

    cnt = 0
    for tw_id in l_tasks:
        su_db_cur.execute(sql_str, (tw_id,))
        rec = su_db_cur.fetchone()
        if rec is None:
            continue
        cls_json_str = rec[0]
        cls_graph = None
        if cls_json_str is not None:
            cls_graph = nx.adjacency_graph(json.loads(cls_json_str))
        nps_str = rec[1]
        l_nps = None
        if nps_str is not None:
            l_nps = [noun_phrase.strip() for noun_phrase in nps_str.split('\n')]
        for embed_type in l_embed_types:
            tw_vect = doc_embedding(cls_graph, l_nps, embed_type)
            tw_vect_str = ','.join([str(ele) for ele in tw_vect])
            out_str = tw_id + ',' + tw_vect_str
            d_out_fds[embed_type].write(out_str)
            d_out_fds[embed_type].write('\n')
            cnt += 1
            if cnt % 1000 == 0 and cnt >= 1000:
                logging.debug('[compute_tw_embeddings_single_thread] Thread %s: %s embeddings have done in %s secs.'
                              % (t_id, cnt, time.time() - timer_start))
    for embed_type in d_out_fds:
        d_out_fds[embed_type].close()

    logging.debug('[compute_tw_embeddings_single_thread] Thread %s: %s tasks all done in %s secs.'
                  % (t_id, cnt, time.time() - timer_start))


def compute_tw_embeddings_multithreading(op_func, l_tasks, l_embed_types, num_threads, output_folder=None):
    timer_1 = time.time()
    logging.debug('[compute_tw_embeddings_multithreading] %s tasks in total.' % len(l_tasks))

    # en_topic_vect = False
    # if 'topic_vect' in l_embed_types:
    #     en_topic_vect = True
    #     l_embed_types = [embed_type for embed_type in l_embed_types if embed_type != 'topic_vect']
    #     num_threads -= 1

    batch_size = math.ceil(len(l_tasks) / num_threads)
    l_l_subtasks = []
    for i in range(0, len(l_tasks), batch_size):
        if i + batch_size < len(l_tasks):
            l_l_subtasks.append(l_tasks[i:i + batch_size])
        else:
            l_l_subtasks.append(l_tasks[i:])
    logging.debug('[compute_tw_embeddings_multithreading] %s threads.' % len(l_l_subtasks))

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_subtasks:
        t = threading.Thread(target=op_func, args=(l_each_batch, l_embed_types,
                                                   't_mul_task_' + str(t_id), output_folder))
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
                logging.debug('[compute_tw_embeddings_multithreading] Thread %s is finished.' % t.getName())

    logging.debug('Embeddings all done in %s secs.' % str(time.time() - timer_1))


def add_col_to_sem_units_embed_db(col_name, embed_db_conn, embed_db_cur):
    embed_sql_str = "alter table ven_tw_en_embeds add column {0} text"
    try:
        embed_db_cur.execute(embed_sql_str.format(col_name))
        embed_db_conn.commit()
    except:
        logging.debug('[add_col_to_sem_units_embed_db] %s already exists.' % col_name)
        return
    logging.debug('[add_col_to_sem_units_embed_db] Add col: %s' % col_name)


def final_output_sem_units_embeddings():
    embed_db_conn = sqlite3.connect(global_settings.g_tw_embed_db)
    embed_db_cur = embed_db_conn.cursor()
    embed_sql_str = '''create table if not exists ven_tw_en_embeds (tw_id text primary key)'''
    embed_db_cur.execute(embed_sql_str)
    embed_db_conn.commit()
    embed_sql_str = "insert into ven_tw_en_embeds (tw_id) values (?)"

    tw_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
    tw_db_cur = tw_db_conn.cursor()
    tw_sql_str = '''select tw_id, tw_type, tw_src_id from ven_tw_en'''
    tw_db_cur.execute(tw_sql_str)
    l_recs = tw_db_cur.fetchall()
    d_tw_ref = dict()
    for rec in l_recs:
        tw_id = rec[0]
        tw_type = rec[1]
        tw_src_id = rec[2]
        if tw_type == 't':
            d_tw_ref[tw_id] = tw_src_id
        else:
            d_tw_ref[tw_id] = []
        embed_db_cur.execute(embed_sql_str, (tw_id,))
    embed_db_conn.commit()
    tw_db_conn.close()
    for embed_type in global_settings.g_embedding_type:
        add_col_to_sem_units_embed_db(embed_type, embed_db_conn, embed_db_cur)

    for tw_id in d_tw_ref:
        if isinstance(d_tw_ref[tw_id], str):
            tw_src_id = d_tw_ref[tw_id]
            if tw_src_id in d_tw_ref:
                d_tw_ref[tw_src_id].append(tw_id)
            else:
                d_tw_ref[tw_id] = []

    embed_sql_str = "update ven_tw_en_embeds set {0} = ? where tw_id = ?"
    for embed_type in global_settings.g_embedding_type:
        cnt = 0
        timer_start = time.time()
        for (dirpath, dirname, filenames) in walk(global_settings.g_tw_embed_folder):
            for filename in filenames:
                if '_'.join(filename.split('_')[:-5]) != embed_type:
                    continue
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for ln in in_fd:
                        fields = ln.split(',', 1)
                        tw_id = fields[0].strip()
                        embed_str = fields[1].strip()
                        if tw_id not in d_tw_ref:
                            raise Exception('[final_output_sem_units_embeddings] Unknown tw_id %s.' % tw_id)
                        if isinstance(d_tw_ref[tw_id], list):
                            embed_db_cur.execute(embed_sql_str.format(embed_type), (embed_str, tw_id))
                            cnt += 1
                            if cnt % 10000 == 0 and cnt >= 10000:
                                embed_db_conn.commit()
                                logging.debug(
                                    '[final_output_sem_units_embeddings] %s embeds for %s have been output in %s secs.'
                                    % (cnt, embed_type, time.time() - timer_start))
                            for trg_tw_id in d_tw_ref[tw_id]:
                                embed_db_cur.execute(embed_sql_str.format(embed_type), (embed_str, trg_tw_id))
                                cnt += 1
                                if cnt % 10000 == 0 and cnt >= 10000:
                                    embed_db_conn.commit()
                                    logging.debug(
                                        '[final_output_sem_units_embeddings] %s embeds for %s have been output in %s secs.'
                                        % (cnt, embed_type, time.time() - timer_start))
        embed_db_conn.commit()
        logging.debug('[final_output_sem_units_embeddings] %s embeds for %s all done in %s secs.'
                      % (cnt, embed_type, time.time() - timer_start))
    embed_db_conn.close()


def update_sem_unit_embeds_for_retweet_makeups():
    embed_db_conn = sqlite3.connect(global_settings.g_tw_embed_db)
    embed_db_cur = embed_db_conn.cursor()
    embed_sql_str = "update ven_tw_en_embeds set {0} = ? where tw_id = ?"
    for embed_type in global_settings.g_embedding_type:
        cnt = 0
        timer_start = time.time()
        for (dirpath, dirname, filenames) in walk(global_settings.g_tw_embed_folder):
            for filename in filenames:
                if '_'.join(filename.split('_')[:-5]) != embed_type:
                    continue
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for ln in in_fd:
                        fields = ln.split(',', 1)
                        tw_id = fields[0].strip()
                        embed_str = fields[1].strip()
                        embed_db_cur.execute(embed_sql_str.format(embed_type), (embed_str, tw_id))
                        cnt += 1
                        if cnt % 10000 == 0 and cnt >= 10000:
                            embed_db_conn.commit()
                            logging.debug(
                                '[update_sem_unit_embeds_for_retweet_makeups] %s embeds for %s have been output in %s secs.'
                                % (cnt, embed_type, time.time() - timer_start))
        embed_db_conn.commit()
        logging.debug('[update_sem_unit_embeds_for_retweet_makeups] %s embeds for %s all done in %s secs.'
                      % (cnt, embed_type, time.time() - timer_start))
    embed_db_conn.close()


def update_sem_units_embeds_for_trivia():
    embed_db_conn = sqlite3.connect(global_settings.g_tw_embed_db)
    embed_db_cur = embed_db_conn.cursor()
    for embed_type in global_settings.g_embedding_type:
        cnt = 0
        embed_sql_str = "select tw_id from ven_tw_en_embeds where {0} is null".format(embed_type)
        embed_db_cur.execute(embed_sql_str)
        l_tw_ids = [rec[0].strip() for rec in embed_db_cur.fetchall()]
        embed_sql_str = "update ven_tw_en_embeds set {0} = ? where tw_id = ?".format(embed_type)
        for tw_id in l_tw_ids:
            embed_db_cur.execute(embed_sql_str, (np.zeros(g_embedding_len), tw_id))
            cnt += 1
            if cnt % 10000 == 0 and cnt >= 1000:
                embed_db_conn.commit()
                logging.debug('[update_sem_units_embeds_for_trivia] %s embeds for %s.' % (cnt, embed_type))
        embed_db_conn.commit()
        logging.debug('[update_sem_units_embeds_for_trivia] %s embeds for %s all done.' % (cnt, embed_type))


def compute_usr_avg_embeds_alt_single_thread(l_embed_recs, dt_start, dt_end, t_id):
    logging.debug('[compute_usr_avg_embeds_alt_single_thread] Thread %s with %s tasks (%s to %s) starts...'
                  % (t_id, len(l_embed_recs), dt_start, dt_end))

    # tw_db_conn = psycopg2.connect(host='postgis1',
    #                                  port=5432,
    #                                  dbname='socialsim',
    #                                  user=global_settings.g_postgis1_username,
    #                                  password=global_settings.g_postgis1_password)
    tw_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
    tw_db_cur = tw_db_conn.cursor()
    # tw_db_sql_str = """select usr_id from cp4.tw_statuses where id = '{0}' and created_at >= '{1}' and created_at <= '{2}'"""
    tw_db_sql_str = """select usr_id from ven_tw_en where tw_id = '{0}' and tw_datetime >= '{1}' and tw_datetime <= '{2}'"""

    timer_start = time.time()
    d_usr_avg_embeds = dict()
    embed_cnt = 0
    for embed_rec in l_embed_recs:
        tw_id = embed_rec[0]
        tw_db_cur.execute(tw_db_sql_str.format(tw_id, dt_start, dt_end))
        rec = tw_db_cur.fetchone()
        if rec is None:
            embed_cnt += 1
            continue
        usr_id = rec[0]
        embed_str = embed_rec[1]
        embed_vec = np.zeros(300)
        try:
            embed_vec = np.asarray([float(ele.strip()) for ele in embed_str.split(',')], dtype=np.float32)
        except:
            continue
        if usr_id in d_usr_avg_embeds:
            d_usr_avg_embeds[usr_id] = (d_usr_avg_embeds[usr_id][0] + embed_vec, d_usr_avg_embeds[usr_id][1] + 1)
        else:
            d_usr_avg_embeds[usr_id] = (embed_vec, 1)
        embed_cnt += 1
        if embed_cnt % 5000 == 0 and embed_cnt >= 5000:
            logging.debug('[compute_usr_avg_embeds_alt_single_thread] Thread %s: %s embed_recs processed in %s secs.'
                          % (t_id, embed_cnt, str(time.time() - timer_start)))
    logging.debug('[compute_usr_avg_embeds_alt_single_thread] Thread %s: All %s embed_recs processed in %s secs.'
                  % (t_id, embed_cnt, str(time.time() - timer_start)))

    for usr_id in d_usr_avg_embeds:
        d_usr_avg_embeds[usr_id] = {'emb': d_usr_avg_embeds[usr_id][0].tolist(), 'cnt': d_usr_avg_embeds[usr_id][1]}
    with open(global_settings.g_tw_embed_usr_avg_embeds_int_file_format.format(t_id), 'w+') as out_fd:
        json.dump(d_usr_avg_embeds, out_fd)
        out_fd.close()
    logging.debug('[compute_usr_avg_embeds_alt_single_thread] Thread %s: Output done in %s secs.'
                  % (t_id, str(time.time() - timer_start)))


# 12537978 embedding records in total
def compute_usr_avg_embeds_alt_multithread(op_func, num_threads, embed_rec_offset, task_size, embed_type,
                                           dt_start, dt_end, job_id):
    logging.debug('[compute_usr_avg_embeds_alt_multithread] Job %s: Starts...' % job_id)
    timer_start = time.time()
    embed_db_conn = psycopg2.connect(host='postgis1',
                                     port=5432,
                                     dbname='socialsim',
                                     user=global_settings.g_postgis1_username,
                                     password=global_settings.g_postgis1_password)
    embed_db_cur = embed_db_conn.cursor()
    embed_db_sql_str = """select id, {0} from cp4.tw_embeddings offset {1} fetch next {2} rows only"""

    batch_size = math.ceil(task_size / num_threads)
    l_batches = []
    offset = embed_rec_offset
    stride = batch_size
    for i in range(num_threads):
        embed_db_cur.execute(embed_db_sql_str.format(embed_type, offset, stride))
        rec = embed_db_cur.fetchall()
        l_batches.append(rec)
        offset += stride
    embed_db_cur.close()
    embed_db_conn.close()
    logging.debug('[compute_usr_avg_embeds_alt_multithread] Job %s: %s batches created in %s secs.'
                  % (job_id, len(l_batches), time.time() - timer_start))

    l_threads = []
    t_id = 0
    for each_batch in l_batches:
        t = threading.Thread(target=op_func, args=(each_batch, dt_start, dt_end, str(job_id) + '_' + str(t_id)))
        t.setName('t_mul_task_' + str(job_id) + '_' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('[compute_usr_avg_embeds_single_thread] Thread %s is finished.' % t.getName())

    logging.debug('[compute_usr_avg_embeds_single_thread] Job %s: All done in %s sec for %s tasks.'
                  % (job_id, time.time() - timer_start, task_size))


def compute_usr_avg_embeds_alt_wrapper(task_file_path, embed_type, dt_start, dt_end, job_id):
    with open(task_file_path, 'r') as in_fd:
        ln = in_fd.readline()
        fields = [item.strip() for item in ln.split('|')]
        offset = int(fields[0])
        task_size = int(fields[1])
        in_fd.close()
    compute_usr_avg_embeds_alt_multithread(compute_usr_avg_embeds_alt_single_thread,
                                           multiprocessing.cpu_count(),
                                           offset,
                                           task_size,
                                           embed_type,
                                           dt_start,
                                           dt_end,
                                           job_id)


def make_usr_avg_embeds_alt_task(total_embed_recs, num_jobs):
    offset = 0
    task_size = math.ceil(total_embed_recs / num_jobs)
    for i in range(num_jobs):
        with open(global_settings.g_tw_embed_usr_avg_embeds_alt_task_file_format.format(str(i)), 'w+') as out_fd:
            out_str = str(offset) + '|' + str(task_size)
            out_fd.write(out_str)
            out_fd.close()
        offset += task_size
    logging.debug('[make_usr_avg_embeds_alt_task] All done.')


def usr_avg_embeds_merge_int_results():
    logging.debug('[usr_avg_embeds_merge_int_results] Starts...')
    timer_start = time.time()
    d_usr_avg_embeds = dict()
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_embed_usr_avg_embeds_int_folder):
        for filename in filenames:
            if filename[:19] != 'usr_avg_embeds_int_' or filename[-5:] != '.json':
                continue
            with open(dirpath + '/' + filename, 'r') as in_fd:
                d_cur = json.load(in_fd)
                in_fd.close()
            for usr_id in d_cur:
                sum_embed = np.asarray(d_cur[usr_id]['emb'], dtype=np.float32)
                embed_cnt = d_cur[usr_id]['cnt']
                if sum_embed.shape != (300,):
                    raise Exception('[usr_avg_embeds_merge_int_results] usr_id = %s, sum_embed.shape = %s'
                                    % (usr_id, sum_embed.shape))
                if usr_id in d_usr_avg_embeds:
                    d_usr_avg_embeds[usr_id]['emb'] += sum_embed
                    d_usr_avg_embeds[usr_id]['cnt'] += embed_cnt
                else:
                    d_usr_avg_embeds[usr_id] = {'emb': sum_embed, 'cnt': embed_cnt}
    for usr_id in d_usr_avg_embeds:
        if d_usr_avg_embeds[usr_id]['cnt'] > 0:
            avg_embed = (d_usr_avg_embeds[usr_id]['emb'] / d_usr_avg_embeds[usr_id]['cnt']).tolist()
        else:
            avg_embed = [0.0] * 300
        d_usr_avg_embeds[usr_id] = avg_embed
    logging.debug('[usr_avg_embeds_merge_int_results] Merge done in %s secs.' % str(time.time() - timer_start))
    with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'w+') as out_fd:
        json.dump(d_usr_avg_embeds, out_fd)
        out_fd.close()
    logging.debug('[usr_avg_embeds_merge_int_results] All done in %s secs.' % str(time.time() - timer_start))


def get_activated_usr_ids(tb_name):
    db_conn = psycopg2.connect(host='postgis1',
                                 port=5432,
                                 dbname='socialsim',
                                 user=global_settings.g_postgis1_username,
                                 password=global_settings.g_postgis1_password)
    db_cur = db_conn.cursor()
    sql_str = """select distinct usr_id from cp4.{0}""".format(tb_name)
    db_cur.execute(sql_str)
    l_recs = db_cur.fetchall()
    if l_recs is not None:
        l_usr_ids = [rec[0].strip() for rec in l_recs]
    else:
        raise Exception('[get_activated_usr_ids] No available user in cp4.activated_users!')
    logging.debug('[get_activated_usr_ids] %s users in cp4.activated_users.' % len(l_usr_ids))
    db_cur.close()
    db_conn.close()

    with open(global_settings.g_tw_activated_usr_ids_file, 'w+') as out_fd:
        out_str = '\n'.join(l_usr_ids)
        out_fd.write(out_str)
        out_fd.close()
    logging.debug('[get_activated_usr_ids] All done.')


def load_activated_usrs():
    l_activated_usrs = []
    with open(global_settings.g_tw_activated_usr_ids_file, 'r') as in_fd:
        for ln in in_fd:
            l_activated_usrs.append(ln.strip())
        in_fd.close()
    logging.debug('[load_activated_usrs] num_act_usrs=%s' % len(l_activated_usrs))
    return l_activated_usrs


def get_all_usr_ids():
    db_conn = psycopg2.connect(host='postgis1',
                               port=5432,
                               dbname='socialsim',
                               user=global_settings.g_postgis1_username,
                               password=global_settings.g_postgis1_password)
    db_cur = db_conn.cursor()
    sql_str = """select distinct usr_id from cp4.tw_statuses"""
    db_cur.execute(sql_str)
    l_recs = db_cur.fetchall()
    if l_recs is not None:
        l_usr_ids = [rec[0].strip() for rec in l_recs]
    else:
        raise Exception('[get_all_usr_ids] No available user in cp4.tw_statuses!')
    logging.debug('[get_all_usr_ids] %s users in cp4.tw_statuses.' % len(l_usr_ids))
    db_cur.close()
    db_conn.close()

    with open(global_settings.g_tw_all_usr_ids_file, 'w+') as out_fd:
        out_str = '\n'.join(l_usr_ids)
        out_fd.write(out_str)
        out_fd.close()
    logging.debug('[get_all_usr_ids] All done.')


def load_all_usrs():
    l_all_usrs = []
    with open(global_settings.g_tw_all_usr_ids_file, 'r') as in_fd:
        for ln in in_fd:
            l_all_usrs.append(ln.strip())
        in_fd.close()
    logging.debug('[load_all_usrs] num_all_usrs=%s' % len(l_all_usrs))
    return l_all_usrs


def usr_avg_embeds_make_up_specified_usrs(l_usr_ids):
    num_input_usrs = len(l_usr_ids)
    with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'r') as in_fd:
        d_usr_avg_embeds = json.load(in_fd)
        in_fd.close()
    num_exist_usrs = len(d_usr_avg_embeds)
    for usr_id in l_usr_ids:
        if usr_id not in d_usr_avg_embeds:
            d_usr_avg_embeds[usr_id] = [0.0]*300
    num_all_usrs = len(d_usr_avg_embeds)
    with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'w+') as out_fd:
        json.dump(d_usr_avg_embeds, out_fd)
        out_fd.close()
    logging.debug('[usr_avg_embeds_make_up_specified_usrs] num_input_usrs=%s, num_embed_usrs=%s, num_all_usrs=%s'
                  % (num_input_usrs, num_exist_usrs, num_all_usrs))


def usr_avg_embeds_activated_usrs_only():
    s_activated_usrs = set(load_activated_usrs())
    with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'r') as in_fd:
        d_usr_avg_embeds = json.load(in_fd)
        in_fd.close()
    d_act_usr_avg_embeds = {usr_id: d_usr_avg_embeds[usr_id] for usr_id in s_activated_usrs}
    with open(global_settings.g_tw_embed_usr_avg_embeds_activated_usrs_only_output, 'w+') as out_fd:
        json.dump(d_act_usr_avg_embeds, out_fd)
        out_fd.close()
    logging.debug('[usr_avg_embeds_activated_usrs_only] usr_avg_embeds output for %s activated usrs.'
                  % len(d_act_usr_avg_embeds))


def usr_avg_embeds_stat():
    num_usrs = 0
    zero_cnt = 0
    inv_cnt = 0
    with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'r') as in_fd:
        d_usr_avg_embeds = json.load(in_fd)
        in_fd.close()
    num_usrs = len(d_usr_avg_embeds)
    for usr_id in d_usr_avg_embeds:
        embed = np.asarray([float(ele.strip()) for ele in d_usr_avg_embeds[usr_id].split(',')], dtype=np.float32)
        if np.array_equal(embed, np.zeros(300)):
            zero_cnt += 1
        if len(embed) != 300:
            inv_cnt += 1
    logging.debug('[usr_avg_embeds_stat] num_usrs=%s, zero_cnt=%s, inv_cnt=%s' % (num_usrs, zero_cnt, inv_cnt))

# # @profile
# def compute_usr_avg_embeds_single_thread(l_usr_ids, embed_type, t_id):
#     logging.debug('[compute_usr_avg_embeds_single_thread] Thread %s with %s tasks starts...' % (t_id, len(l_usr_ids)))
#     # embed_db_conn = sqlite3.connect(global_settings.g_tw_embed_db)
#     # embed_db_cur = embed_db_conn.cursor()
#     # embed_db_sql_str = '''select {0} from ven_tw_en_embeds where tw_id = ?'''
#
#     embed_db_conn = psycopg2.connect(host='postgis1',
#                                      port=5432,
#                                      dbname='socialsim',
#                                      user='mf3jh',
#                                      password='LSRDeae19830602!')
#     embed_db_cur = embed_db_conn.cursor()
#     embed_db_sql_str = """select {0} from cp4.tw_embeddings where id = '{1}'"""
#
#     tw_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
#     tw_db_cur = tw_db_conn.cursor()
#     tw_db_sql_str = '''select tw_id from ven_tw_en where usr_id = ?'''
#
#     timer_start = time.time()
#     d_usr_avg_embeds = dict()
#     usr_cnt = 0
#     # TEST ONLY STARTS
#     test_embed_cnt = 0
#     # TEST ONLY ENDS
#     for idx, usr_id in enumerate(l_usr_ids):
#         # TEST ONLY STARTS
#         # if idx > 100:
#         #     embed_db_cur.close()
#         #     embed_db_conn.close()
#         #     tw_db_conn.close()
#         #     return
#         # TEST ONLY ENDS
#         usr_avg_embed = np.zeros(300)
#         tw_db_cur.execute(tw_db_sql_str, (usr_id,))
#         l_recs = tw_db_cur.fetchall()
#         if l_recs is None:
#             d_usr_avg_embeds[usr_id] = ','.join([str(ele) for ele in usr_avg_embed])
#             continue
#         # logging.debug('[compute_usr_avg_embeds_single_thread] %s tw_ids for %s.' % (len(l_recs), usr_id))
#         l_tw_ids = [rec[0] for rec in l_recs]
#         sample_size = 20
#         if len(l_tw_ids) > sample_size:
#             l_tw_ids_sample = random.sample(l_tw_ids, sample_size)
#         else:
#             l_tw_ids_sample = l_tw_ids
#         embed_cnt = 0
#         for tw_id in l_tw_ids_sample:
#             embed_db_cur.execute(embed_db_sql_str.format(embed_type, tw_id))
#             embed_rec = embed_db_cur.fetchone()
#             if embed_rec is None:
#                 continue
#             if type(embed_rec[0]) != str:
#                 continue
#             embed = []
#             try:
#                 embed = [float(ele.strip()) for ele in embed_rec[0].split(',')]
#             except:
#                 continue
#             if len(embed) != 300:
#                 continue
#             usr_avg_embed += np.asarray(embed, dtype=np.float32)
#             embed_cnt += 1
#             # TEST ONLY STARTS
#             test_embed_cnt += 1
#             if test_embed_cnt >= 500:
#                 logging.debug('[compute_usr_avg_embeds_single_thread] test only %s secs.'
#                               % str(time.time() - timer_start))
#                 return
#             # TEST ONLY ENDS
#         if embed_cnt > 0:
#             usr_avg_embed = usr_avg_embed / embed_cnt
#         d_usr_avg_embeds[usr_id] = ','.join([str(ele) for ele in usr_avg_embed])
#         usr_cnt += 1
#         if usr_cnt % 500 == 0 and usr_cnt > 500:
#             logging.debug('[compute_usr_avg_embeds_single_thread] Thread %s: %s users are done in %s secs.'
#                           % (t_id, usr_cnt, time.time() - timer_start))
#     logging.debug('[compute_usr_avg_embeds_single_thread] Thread %s: All %s users are done in %s secs.'
#                   % (t_id, usr_cnt, time.time() - timer_start))
#     embed_db_cur.close()
#     embed_db_conn.close()
#     tw_db_conn.close()
#
#     with open(global_settings.g_tw_embed_usr_avg_embeds_int_file_format.format(t_id), 'w+') as out_fd:
#         json.dump(d_usr_avg_embeds, out_fd)
#         out_fd.close()
#     logging.debug('[compute_usr_avg_embeds_single_thread] Thread %s: usr_avg_embeds written.' % t_id)
#
#
# def compute_usr_avg_embeds_multithread(l_tasks, op_func, num_threads, embed_type, job_id):
#     timer_1 = time.time()
#     logging.debug('[compute_usr_avg_embeds_single_thread] %s tasks in total.' % len(l_tasks))
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
#         t = threading.Thread(target=op_func, args=(l_each_batch, embed_type, str(job_id) + '_' + str(t_id)))
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
#                 logging.debug('[compute_usr_avg_embeds_single_thread] Thread %s is finished.' % t.getName())
#
#     logging.debug('[compute_usr_avg_embeds_single_thread] All done in %s sec for %s tasks.'
#                   % (time.time() - timer_1, len(l_tasks)))
#
#
# def load_usr_avg_embeds_tasks(task_file_path):
#     l_usr_ids = []
#     with open(task_file_path, 'r') as in_fd:
#         for ln in in_fd:
#             l_usr_ids.append(ln.strip())
#         in_fd.close()
#     return l_usr_ids


############################################################
# TESTING ONLY STARTS
############################################################
def test_only_final_output_sem_units_embeddings():
    embed_db_conn = sqlite3.connect(global_settings.g_tw_embed_db)
    embed_db_cur = embed_db_conn.cursor()
    embed_sql_str = '''create table if not exists ven_tw_en_embeds (tw_id text primary key)'''
    embed_db_cur.execute(embed_sql_str)
    embed_db_conn.commit()
    # embed_sql_str = "insert into ven_tw_en_embeds (tw_id) values (?)"

    tw_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
    tw_db_cur = tw_db_conn.cursor()
    tw_sql_str = '''select tw_id, tw_type, tw_src_id from ven_tw_en'''
    tw_db_cur.execute(tw_sql_str)
    l_recs = tw_db_cur.fetchall()
    d_tw_ref = dict()
    for rec in l_recs:
        tw_id = rec[0]
        tw_type = rec[1]
        tw_src_id = rec[2]
        if tw_type == 't':
            d_tw_ref[tw_id] = tw_src_id
        else:
            d_tw_ref[tw_id] = []
        # embed_db_cur.execute(embed_sql_str, (tw_id,))
    # embed_db_conn.commit()
    tw_db_conn.close()

    for embed_type in global_settings.g_embedding_type:
        add_col_to_sem_units_embed_db(embed_type, embed_db_conn, embed_db_cur)

    for tw_id in d_tw_ref:
        if isinstance(d_tw_ref[tw_id], str):
            tw_src_id = d_tw_ref[tw_id]
            if tw_src_id in d_tw_ref:
                d_tw_ref[tw_src_id].append(tw_id)
            else:
                d_tw_ref[tw_id] = []

    # embed_sql_str = "update ven_tw_en_embeds set {0} = ? where tw_id = ?"
    embed_sql_str = "insert into ven_tw_en_embeds (tw_id, {0}) values (?, ?)"
    for embed_type in global_settings.g_embedding_type:
        cnt = 0
        timer_start = time.time()
        for (dirpath, dirname, filenames) in walk(global_settings.g_tw_embed_folder):
            for filename in filenames:
                if '_'.join(filename.split('_')[:-5]) != embed_type:
                    continue
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for ln in in_fd:
                        fields = ln.split(',', 1)
                        tw_id = fields[0].strip()
                        embed_str = fields[1].strip()
                        if tw_id not in d_tw_ref:
                            raise Exception('[final_output_sem_units_embeddings] Unknown tw_id %s.' % tw_id)
                        if isinstance(d_tw_ref[tw_id], list):
                            embed_db_cur.execute(embed_sql_str.format(embed_type), (tw_id, embed_str))
                            cnt += 1
                            if cnt % 10000 == 0 and cnt >= 10000:
                                embed_db_conn.commit()
                                logging.debug(
                                    '[final_output_sem_units_embeddings] %s embeds for %s have been output in %s secs.'
                                    % (cnt, embed_type, time.time() - timer_start))
                            for trg_tw_id in d_tw_ref[tw_id]:
                                embed_db_cur.execute(embed_sql_str.format(embed_type), (embed_str, trg_tw_id))
                                cnt += 1
                                if cnt % 10000 == 0 and cnt >= 10000:
                                    embed_db_conn.commit()
                                    logging.debug(
                                        '[final_output_sem_units_embeddings] %s embeds for %s have been output in %s secs.'
                                        % (cnt, embed_type, time.time() - timer_start))
        embed_db_conn.commit()
        logging.debug('[final_output_sem_units_embeddings] %s embeds for %s all done in %s secs.'
                      % (cnt, embed_type, time.time() - timer_start))
    embed_db_conn.close()


def test_invalid_embeds():
    db_conn = sqlite3.connect(global_settings.g_tw_embed_folder + 'tw_embed_test.db')
    db_cur = db_conn.cursor()
    sql_str = '''create table if not exists ven_tw_en_embeds (tw_id text primary key, avg_nps_vect text, avg_nps_vect_w_avg_edge_vect text, avg_nps_vect_sp_w_avg_edge_vect text)'''
    db_cur.execute(sql_str)
    db_conn.commit()
    sql_str = '''insert into ven_tw_en_embeds (tw_id, avg_nps_vect, avg_nps_vect_w_avg_edge_vect, avg_nps_vect_sp_w_avg_edge_vect) values (?, ?, ?, ?)'''
    l_tw_ids = ['6VCy9UN73ZTS-LSqY7dBLg', 'APgQ_sc_bImq4kc5Pswr2Q']

    orig_db_conn = sqlite3.connect(global_settings.g_tw_embed_db)
    orig_db_cur = orig_db_conn.cursor()
    orig_sql_str = '''select * from ven_tw_en_embeds where tw_id = ?'''
    for tw_id in l_tw_ids:
        orig_db_cur.execute(orig_sql_str, (tw_id,))
        rec = orig_db_cur.fetchone()
        db_cur.execute(sql_str, (tw_id, rec[1], rec[2], rec[3]))
    db_conn.commit()
    logging.debug('[test_invalid_embeds] test embed db is done.')
    db_conn.close()
    orig_db_conn.close()


def test_embed_extraction():
    db_conn = sqlite3.connect(global_settings.g_tw_embed_folder + 'tw_embed_test.db')
    db_cur = db_conn.cursor()
    embed_type = 'avg_nps_vect_sp_w_avg_edge_vect'
    sql_str = '''select {0} from ven_tw_en_embeds where tw_id = ?'''.format(embed_type)
    l_tw_ids = ['6VCy9UN73ZTS-LSqY7dBLg', 'APgQ_sc_bImq4kc5Pswr2Q']
    for tw_id in l_tw_ids:
        usr_avg_embed = np.zeros(300)
        db_cur.execute(sql_str, (tw_id,))
        rec = db_cur.fetchone()
        if type(rec[0]) != str:
            continue
        l_vals = rec[0].split(',')
        if len(l_vals) != 300:
            continue
        embed = [float(ele.strip()) for ele in rec[0].split(',')]
        if len(embed) != 300:
            continue
        usr_avg_embed += np.asarray(embed)


def convert_embed_strs_to_vecs():
    with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'r') as in_fd:
        d_embed_strs = json.load(in_fd)
        in_fd.close()
    for key in d_embed_strs:
        d_embed_strs[key] = [float(ele.strip()) for ele in d_embed_strs[key].split(',')]
    with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'w+') as out_fd:
        json.dump(d_embed_strs, out_fd)
        out_fd.close()
    logging.debug('[convert_embed_strs_to_vecs] All done.')


# def test_usr_avg_embeds(usr_id):
#     with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'r') as in_fd:

############################################################
# TESTING ONLY ENDS
############################################################

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    '''
    Step #1: Compute Embeddings
    '''
    # load_lexvec_model()
    # l_tasks = tw_embeddings_tasks()
    # l_tasks = tw_embeddings_retweet_makeup_tasks()
    # compute_tw_embeddings_multithreading(compute_tw_embeddings_single_thread,
    #                                      l_tasks,
    #                                      global_settings.g_embedding_type,
    #                                      multiprocessing.cpu_count(),
    #                                      global_settings.g_tw_embed_folder)

    '''Output Embeddings'''
    # final_output_sem_units_embeddings()
    # update_sem_unit_embeds_for_retweet_makeups()
    # update_sem_units_embeds_for_trivia()
    # test_only_final_output_sem_units_embeddings()
    # test_invalid_embeds()
    # test_embed_extraction()
    '''User Average Embeddings'''
    # make_usr_avg_embeds_alt_task(12537978, 20)
    # task_file_path = sys.argv[1]
    # job_id = sys.argv[2]
    # embed_type = 'avg_nps_vect_sp_w_avg_edge_vect'
    # dt_start = '2018-12-14'
    # dt_end = '2019-01-10'
    # dt_start = '20181224000000'
    # dt_end = '2019011823595959'
    # compute_usr_avg_embeds_alt_wrapper(task_file_path, embed_type, dt_start, dt_end, job_id)
    # usr_avg_embeds_merge_int_results()
    # get_activated_usr_ids('activated_users_int_dry_run_train_w_init')
    # usr_avg_embeds_make_up_specified_usrs(load_all_usrs())
    # usr_avg_embeds_activated_usrs_only()
    # get_all_usr_ids()
    # usr_avg_embeds_stat()
    # convert_embed_strs_to_vecs()
