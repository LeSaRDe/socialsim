import copy
import logging
import sys
sys.path.insert(1, '/home/mf3jh/workspace/lib/lexvec/lexvec/python/lexvec/')
import model as lexvec
import numpy as np
import networkx as nx
import os
import sqlite3
from pygsp import graphs, filters
import re
from os import walk, path
import time
import scipy.spatial.distance as scipyd
import json
from random import sample
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import multiprocessing
import threading
import math


############################################################
#   Effective Code Starts Here
############################################################
g_d_pos_pair_weights = {'VERB_NOUN': 0.648, 'NOUN_VERB': 0.648,
                        'NOUN_NOUN': 0.244,
                        'VERB_VERB': 0.040,
                        'NOUN_ADJ': 0.028, 'ADJ_NOUN': 0.028,
                        'VERB_ADV': 0.010, 'ADV_VERB': 0.010,
                        'NOUN_ADV': 0.010, 'ADV_NOUN': 0.010,
                        'VERB_ADJ': 0.011, 'ADJ_VERB': 0.011,
                        'ADP_NOUN': 0.005, 'NOUN_ADP': 0.005,
                        'PROPN_NOUN': 0.004, 'NOUN_PROPN': 0.004}

# DSCTP weights
# g_d_pos_pair_weights = {'NOUN_NOUN': 0.616,
#                         'VERB_NOUN': 0.164, 'NOUN_VERB': 0.164,
#                         'VERB_VERB': 0.107}
# made-up weights
# NOUN_NOUN only: NMI=0.59
# NOUN_VERB only: NMI=0.59
# VERB_VERB only: NMI=0.51
# NOUN_ADJ only: NMI=0.48
# ALL ZERO: NMI=0.50 => the Ven tw dataset sucks!
# g_d_pos_pair_weights = {'NOUN_NOUN': 0.5,
#                         'VERB_NOUN': 0.5, 'NOUN_VERB': 0.5}
# g_d_pos_pair_weights = {'NOUN_NOUN': 0.5, 'VERB_NOUN': 0.5}

# g_d_pos_pair_weights = {'NOUN_NOUN': 0.244,
#                         'VERB_NOUN': 0.648,
#                         'VERB_VERB': 0.040}

g_word_embedding_model = 'lexvec'
g_lexvec_model = None


def load_lexvec_model():
    global g_lexvec_model
    # model_file = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.300d.W+C.pos.vectors'
    # g_lexvec_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    model_file = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
    g_lexvec_model = lexvec.Model(model_file)


def phrase_embedding(phrase_str):
    '''
    Take a phrase (i.e. a sequence of tokens connected by whitespaces) and compute an embedding for it.
    This function acts as a wrapper of word embeddings. Refactor this function to adapt various word embedding models.
    :param
        phrase_str: An input phrase
    :return:
        An embedding for the phrase
    '''
    if g_word_embedding_model == 'lexvec':
        if g_lexvec_model is None:
            load_lexvec_model()
        phrase_vec = np.zeros(300)
        l_words = [word.strip().lower() for word in phrase_str.split(' ')]
        for word in l_words:
            word_vec = g_lexvec_model.word_rep(word)
            phrase_vec += word_vec
        return phrase_vec


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
        embed_type: ['avg_node_vect'|'avg_edge_vect'|'s_avg_edge_vect'|'w_avg_edge_vect'|'sp_avg_node_vect'|'sp_w_avg_edge_vect'|'avg_nps_vect']
        l_nps: The list of noun phrases.
    :return:
        A vector for the doc.
    '''
    if cls_graph is None:
        logging.error('Empty cls_graph exists!')
        if embed_type == 'pos_decomp':
            np.zeros(len(g_d_pos_pair_weights) * 300)
        return np.zeros(300)

    if len(cls_graph.nodes()) == 0:
        return np.zeros(300)

    if len(cls_graph.nodes()) == 1:
        only_node = list(cls_graph.nodes())[0]
        node_fields = only_node.split('|')
        node_txt = node_fields[-1].strip()
        doc_id = node_fields[1].strip()
        node_vect = phrase_embedding(node_txt)
        # logging.debug('%s cls_graph has only one node!' % doc_id)
        if embed_type == 'pos_decomp':
            node_pos = cls_graph.nodes(data=True)[only_node]['pos']
            if node_pos == 'NOUN':
                # node_vect = g_d_pos_pair_weights['NOUN_NOUN'] \
                #             * np.concatenate([node_vect, np.zeros(300), np.zeros(300)], axis=0)
                node_vect = np.concatenate([node_vect, np.zeros(300)], axis=0)
            elif node_pos == 'VERB':
                # node_vect = g_d_pos_pair_weights['VERB_VERB'] \
                #             * np.concatenate([np.zeros(300), np.zeros(300), node_vect], axis=0)
                node_vect = np.concatenate([np.zeros(300), node_vect], axis=0)
            else:
                np.zeros(len(g_d_pos_pair_weights) * 300)
        return node_vect

    if embed_type == 'avg_node_vect':
        l_node_vects = []
        for node in cls_graph.nodes():
            node_fields = node.split('|')
            node_txt = node_fields[-1].strip()
            node_vect = phrase_embedding(node_txt)
            l_node_vects.append(node_vect)
        return sum(l_node_vects) / len(l_node_vects)
    elif embed_type == 'avg_edge_vect':
        l_edge_vects = []
        for comp in nx.connected_components(cls_graph):
            sub_cls_graph = nx.subgraph(cls_graph, comp)
            if len(sub_cls_graph.nodes()) == 1:
                single_node = list(sub_cls_graph.nodes())[0]
                single_node_fields = single_node.split('|')
                single_node_txt = single_node_fields[-1].strip()
                single_node_vect = phrase_embedding(single_node_txt)
                l_edge_vects.append(single_node_vect)
                continue
            for edge in sub_cls_graph.edges():
                node_1 = edge[0]
                node_2 = edge[1]
                node_1_fields = node_1.split('|')
                node_2_fields = node_2.split('|')
                node_1_txt = node_1_fields[-1].strip()
                node_2_txt = node_2_fields[-1].strip()
                node_1_vect = phrase_embedding(node_1_txt)
                node_2_vect = phrase_embedding(node_2_txt)
                edge_vect = (node_1_vect + node_2_vect) / 2
                l_edge_vects.append(edge_vect)
        return sum(l_edge_vects) / len(l_edge_vects)
    elif embed_type == 's_avg_edge_vect':
        s_selected_pos_pairs = {'VERB_NOUN', 'NOUN_VERB', 'NOUN_NOUN', 'VERB_VERB'}
        l_edge_vects = []
        for comp in nx.connected_components(cls_graph):
            sub_cls_graph = nx.subgraph(cls_graph, comp)
            if len(sub_cls_graph.nodes()) == 1:
                single_node = list(sub_cls_graph.nodes())[0]
                single_node_fields = single_node.split('|')
                single_node_txt = single_node_fields[-1].strip()
                single_node_vect = phrase_embedding(single_node_txt)
                l_edge_vects.append(single_node_vect)
                continue
            for edge in sub_cls_graph.edges():
                node_1 = edge[0]
                node_2 = edge[1]
                node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
                node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
                if (node_1_pos + '_' + node_2_pos in s_selected_pos_pairs) \
                        or (node_2_pos + '_' + node_1_pos in s_selected_pos_pairs):
                    node_1_fields = node_1.split('|')
                    node_2_fields = node_2.split('|')
                    node_1_txt = node_1_fields[-1].strip()
                    node_2_txt = node_2_fields[-1].strip()
                    node_1_vect = phrase_embedding(node_1_txt)
                    node_2_vect = phrase_embedding(node_2_txt)
                    edge_vect = (node_1_vect + node_2_vect) / 2
                    l_edge_vects.append(edge_vect)
        if len(l_edge_vects) > 0:
            return sum(l_edge_vects) / len(l_edge_vects)
        else:
            return np.zeros(300)
    elif embed_type == 'w_avg_edge_vect':
        l_edge_vects = []
        for comp in nx.connected_components(cls_graph):
            sub_cls_graph = nx.subgraph(cls_graph, comp)
            if len(sub_cls_graph.nodes()) == 1:
                single_node = list(sub_cls_graph.nodes())[0]
                single_node_weight = 1.0
                # single_node_pos = cls_graph.nodes(data=True)[single_node]['pos']
                # if single_node_pos == 'NOUN':
                #     single_node_weight = g_d_pos_pair_weights['NOUN_NOUN']
                # elif single_node_pos == 'VERB':
                #     single_node_weight = g_d_pos_pair_weights['VERB_VERB']
                # else:
                #     continue
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
                if node_1_pos + '_' + node_2_pos in g_d_pos_pair_weights:
                    node_1_fields = node_1.split('|')
                    node_2_fields = node_2.split('|')
                    node_1_txt = node_1_fields[-1].strip()
                    node_2_txt = node_2_fields[-1].strip()
                    node_1_vect = phrase_embedding(node_1_txt)
                    node_2_vect = phrase_embedding(node_2_txt)
                    edge_vect = (node_1_vect + node_2_vect) / 2
                    w_edge_vect = edge_vect * g_d_pos_pair_weights[node_1_pos + '_' + node_2_pos]
                    l_edge_vects.append(w_edge_vect)
        if len(l_edge_vects) > 0:
            return sum(l_edge_vects) / len(l_edge_vects)
            # return sum(l_edge_vects)
        else:
            return np.zeros(300)
    elif embed_type == 'sp_avg_node_vect':
        l_nodes, gsp_graph = build_gsp_graph(cls_graph)
        l_node_vects = []
        for node in l_nodes:
            node_fields = node.split('|')
            node_txt = node_fields[-1].strip()
            node_vect = phrase_embedding(node_txt)
            l_node_vects.append(node_vect)
        if len(cls_graph.edges()) == 0:
            ret_vect = sum(l_node_vects) / len(l_node_vects)
            if not np.isfinite(ret_vect).all():
                raise Exception('Invalid doc vect!')
            return ret_vect
        l_spectral_node_vects = []
        for node_idx, node in enumerate(l_nodes):
            pulse_sig = build_gsp_pulse_signal(l_nodes, node_idx)
            filtered_sig = heat_kernal_filtering(gsp_graph, pulse_sig)
            spectral_node_vect = np.zeros(300)
            for sig_idx, sig_val in enumerate(filtered_sig):
                if np.isfinite(sig_val):
                    spectral_node_vect += sig_val * l_node_vects[sig_idx]
            spectral_node_vect = spectral_node_vect / len(filtered_sig)
            spectral_node_vect = (spectral_node_vect + l_node_vects[node_idx]) / 2
            l_spectral_node_vects.append(spectral_node_vect)
        ret_vect = sum(l_spectral_node_vects) / len(l_spectral_node_vects)
        if not np.isfinite(ret_vect).all():
            raise Exception('Invalid doc vect!')
        return ret_vect
    elif embed_type == 'sp_w_avg_edge_vect':
        l_nodes, gsp_graph = build_gsp_graph(cls_graph)
        l_node_vects = []
        for node in l_nodes:
            node_fields = node.split('|')
            node_txt = node_fields[-1].strip()
            node_vect = phrase_embedding(node_txt)
            l_node_vects.append(node_vect)
        if len(cls_graph.edges()) == 0:
            return sum(l_node_vects) / len(l_node_vects)
        l_spectral_node_vects = []
        for node_idx, node in enumerate(l_nodes):
            pulse_sig = build_gsp_pulse_signal(l_nodes, node_idx)
            filtered_sig = heat_kernal_filtering(gsp_graph, pulse_sig)
            spectral_node_vect = np.zeros(300)
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
                # single_node_pos = cls_graph.nodes(data=True)[single_node]['pos']
                # if single_node_pos == 'NOUN':
                #     single_node_weight = g_d_pos_pair_weights['NOUN_NOUN']
                # elif single_node_pos == 'VERB':
                #     single_node_weight = g_d_pos_pair_weights['VERB_VERB']
                # else:
                #     continue
                single_node_vect = l_spectral_node_vects[l_nodes.index(single_node)]
                l_edge_vects.append(single_node_vect * single_node_weight)
            for edge in sub_cls_graph.edges():
                node_1 = edge[0]
                node_2 = edge[1]
                node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
                node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
                if node_1_pos + '_' + node_2_pos in g_d_pos_pair_weights:
                    node_1_vect = l_spectral_node_vects[l_nodes.index(node_1)]
                    node_2_vect = l_spectral_node_vects[l_nodes.index(node_2)]
                    edge_vect = (node_1_vect + node_2_vect) / 2
                    w_edge_vect = edge_vect * g_d_pos_pair_weights[node_1_pos + '_' + node_2_pos]
                    l_edge_vects.append(w_edge_vect)
        if len(l_edge_vects) > 0:
            ret_vect = sum(l_edge_vects) / len(l_edge_vects)
            if not np.isfinite(ret_vect).all():
                raise Exception('Invalid doc vect!')
            return ret_vect
        else:
            return np.zeros(300)
    elif embed_type == 'pos_decomp':
        d_sel_pos_pair_edge_vects = {key: [] for key in g_d_pos_pair_weights}
        for edge in cls_graph.edges():
            node_1 = edge[0]
            node_2 = edge[1]
            node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
            node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
            pos_pair = node_1_pos + '_' + node_2_pos
            pos_pair_reverse = node_2_pos + '_' + node_1_pos
            if pos_pair in d_sel_pos_pair_edge_vects:
                actual_pos_pair = pos_pair
            elif pos_pair_reverse in d_sel_pos_pair_edge_vects:
                actual_pos_pair = pos_pair_reverse
            else:
                continue
            node_1_fields = node_1.split('|')
            node_2_fields = node_2.split('|')
            node_1_txt = node_1_fields[-1].strip()
            node_2_txt = node_2_fields[-1].strip()
            node_1_vect = phrase_embedding(node_1_txt)
            node_2_vect = phrase_embedding(node_2_txt)
            edge_vect = (node_1_vect + node_2_vect) / 2
            d_sel_pos_pair_edge_vects[actual_pos_pair].append(edge_vect)
        # final_vect = np.zeros(300)
        l_comp_vects = []
        for pos_pair in ['NOUN_NOUN', 'VERB_NOUN']:
            if len(d_sel_pos_pair_edge_vects[pos_pair]) > 0:
                # d_sel_pos_pair_edge_vects[pos_pair] = g_d_pos_pair_weights[pos_pair] * \
                #                                       sum(d_sel_pos_pair_edge_vects[pos_pair]) / \
                #                                       len(d_sel_pos_pair_edge_vects[pos_pair])
                d_sel_pos_pair_edge_vects[pos_pair] = sum(d_sel_pos_pair_edge_vects[pos_pair]) / len(d_sel_pos_pair_edge_vects[pos_pair])
            else:
                d_sel_pos_pair_edge_vects[pos_pair] = np.zeros(300)
            l_comp_vects.append(np.asarray(d_sel_pos_pair_edge_vects[pos_pair]))
        if len(l_comp_vects) > 0:
            final_vect = np.concatenate(l_comp_vects, axis=0)
        else:
            final_vect = np.zeros(len(d_sel_pos_pair_edge_vects) * 300)
        #     final_vect += d_sel_pos_pair_edge_vects[pos_pair]
        if final_vect.shape[0] != (len(d_sel_pos_pair_edge_vects) * 300):
            raise Exception('final_vect has a wrong shape!')
        return final_vect
    elif embed_type == 'avg_nps_vect':
        if len(l_nps) == 0:
            return np.zeros(300)
        else:
            l_np_vects = []
            for noun_phrase in l_nps:
                np_vect = phrase_embedding(noun_phrase.strip())
                l_np_vects.append(np_vect)
            return sum(l_np_vects) / len(l_np_vects)
    elif embed_type == 'avg_nps_vect_avg_node_vect':
        # sql_str = '''select avg_nps_vect, avg_node_vect from docs where tw_id = ?'''
        sql_str = '''select avg_nps_vect, avg_node_vect from docs where doc_id = ?'''
        doc_sim_db_cur.execute(sql_str, (doc_id,))
        rec = doc_sim_db_cur.fetchone()
        avg_nps_vect = np.asarray([float(ele.strip()) for ele in rec[0].split(',')])
        avg_node_vect = np.asarray([float(ele.strip()) for ele in rec[1].split(',')])
        return avg_nps_vect + avg_node_vect
    elif embed_type == 'avg_nps_vect_avg_edge_vect':
        # sql_str = '''select avg_nps_vect, avg_edge_vect from docs where tw_id = ?'''
        sql_str = '''select avg_nps_vect, avg_edge_vect from docs where doc_id = ?'''
        doc_sim_db_cur.execute(sql_str, (doc_id,))
        rec = doc_sim_db_cur.fetchone()
        avg_nps_vect = np.asarray([float(ele.strip()) for ele in rec[0].split(',')])
        avg_edge_vect = np.asarray([float(ele.strip()) for ele in rec[1].split(',')])
        return avg_nps_vect + avg_edge_vect
    elif embed_type == 'avg_nps_vect_s_avg_edge_vect':
        # sql_str = '''select avg_nps_vect, s_avg_edge_vect from docs where tw_id = ?'''
        sql_str = '''select avg_nps_vect, s_avg_edge_vect from docs where doc_id = ?'''
        doc_sim_db_cur.execute(sql_str, (doc_id,))
        rec = doc_sim_db_cur.fetchone()
        avg_nps_vect = np.asarray([float(ele.strip()) for ele in rec[0].split(',')])
        s_avg_edge_vect = np.asarray([float(ele.strip()) for ele in rec[1].split(',')])
        return avg_nps_vect + s_avg_edge_vect
    elif embed_type == 'avg_nps_vect_w_avg_edge_vect':
        sql_str = '''select avg_nps_vect, w_avg_edge_vect from ven_tw_en where tw_id = ?'''
        # sql_str = '''select avg_nps_vect, w_avg_edge_vect from docs where doc_id = ?'''
        doc_sim_db_cur.execute(sql_str, (doc_id,))
        rec = doc_sim_db_cur.fetchone()
        avg_nps_vect = np.asarray([float(ele.strip()) for ele in rec[0].split(',')])
        w_avg_edge_vect = np.asarray([float(ele.strip()) for ele in rec[1].split(',')])
        return avg_nps_vect + w_avg_edge_vect
    elif embed_type == 'avg_nps_vect_sp_avg_node_vect':
        # sql_str = '''select avg_nps_vect, sp_avg_node_vect from docs where tw_id = ?'''
        sql_str = '''select avg_nps_vect, sp_avg_node_vect from docs where doc_id = ?'''
        doc_sim_db_cur.execute(sql_str, (doc_id,))
        rec = doc_sim_db_cur.fetchone()
        avg_nps_vect = np.asarray([float(ele.strip()) for ele in rec[0].split(',')])
        sp_avg_node_vect = np.asarray([float(ele.strip()) for ele in rec[1].split(',')])
        return avg_nps_vect + sp_avg_node_vect
    elif embed_type == 'avg_nps_vect_sp_w_avg_edge_vect':
        sql_str = '''select avg_nps_vect, sp_w_avg_edge_vect from ven_tw_en where tw_id = ?'''
        # sql_str = '''select avg_nps_vect, sp_w_avg_edge_vect from docs where doc_id = ?'''
        doc_sim_db_cur.execute(sql_str, (doc_id,))
        rec = doc_sim_db_cur.fetchone()
        avg_nps_vect = np.asarray([float(ele.strip()) for ele in rec[0].split(',')])
        sp_w_avg_edge_vect = np.asarray([float(ele.strip()) for ele in rec[1].split(',')])
        return avg_nps_vect + sp_w_avg_edge_vect
    else:
        raise Exception('Unsupported embedding type %s.' % embed_type)


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
        if node_1_pos + '_' + node_2_pos in g_d_pos_pair_weights:
            edge_weight = g_d_pos_pair_weights[node_1_pos + '_' + node_2_pos]
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
    return ret_signal


############################################################
#   Effective Code Ends Here
############################################################


def build_txt_db(txt_set_name):
    src_db_folder = '/home/mf3jh/workspace/data/docsim/'
    src_db_path = src_db_folder + txt_set_name + '.db'
    src_db_conn = sqlite3.connect(src_db_path)
    src_db_cur = src_db_conn.cursor()
    src_sql_str = '''select doc_id, pre_ner from docs'''

    trg_db_path = src_db_folder + txt_set_name + '_doc_embed.db'
    trg_db_conn = sqlite3.connect(trg_db_path)
    trg_db_cur = trg_db_conn.cursor()
    trg_sql_str = '''create table if not exists docs (doc_id text primary key, raw_txt text)'''
    trg_db_cur.execute(trg_sql_str)

    trg_sql_str = '''insert into docs (doc_id, raw_txt) values (?, ?)'''
    src_db_cur.execute(src_sql_str)
    l_src_recs = src_db_cur.fetchall()
    for src_rec in l_src_recs:
        doc_id = src_rec[0].strip()
        if txt_set_name == '20news50short10':
            doc_id = re.sub(r'/', '_', doc_id)
        raw_txt = src_rec[1].strip()
        trg_db_cur.execute(trg_sql_str, (doc_id, raw_txt))
    trg_db_conn.commit()
    trg_db_conn.close()
    src_db_conn.close()


def alter_table_add_col(txt_set_name, col_name, col_type):
    db_folder = '/home/mf3jh/workspace/data/docsim/'
    db_path = db_folder + txt_set_name + '_doc_embed.db'
    if txt_set_name == 'tw_man_nar_data':
        db_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
        db_path = db_folder + txt_set_name + '.db'
    db_conn = sqlite3.connect(db_path)
    db_cur = db_conn.cursor()
    db_sql_str = 'alter table docs add column ' + col_name + ' ' + col_type
    db_cur.execute(db_sql_str)
    db_conn.close()


def update_doc_embeddings(txt_set_name, embedding_type):
    timer_start = time.time()
    db_folder = '/home/mf3jh/workspace/data/docsim/'
    db_path = db_folder + txt_set_name + '_doc_embed.db'
    db_conn = sqlite3.connect(db_path)
    db_cur = db_conn.cursor()
    db_update_sql_str = 'update docs set {0} = ? where doc_id = ?'.format(embedding_type)
    db_select_sql_str = '''select raw_txt from docs where doc_id = ?'''

    cnt = 0
    sem_unit_folder = '/home/mf3jh/workspace/data/docsim/sem_units_' + txt_set_name + '/'
    for (dirpath, dirname, filenames) in walk(sem_unit_folder):
        for filename in filenames:
            if filename[-14:] == '_cls_graph.gml':
                # TEST ONLY STARTS
                #if re.match(r'alt.atheism_54160.*.gml', filename) is None:
                #    continue
                # TEST ONLY ENDS
                doc_id = filename[:-14]
                db_cur.execute(db_select_sql_str, (doc_id,))
                doc_rec = db_cur.fetchone()
                if doc_rec is None:
                    logging.error('%s does not have record in %s.' % (doc_id, txt_set_name))
                    continue
                cls_graph = nx.read_gml(dirpath + '/' + filename)
                nps_path = dirpath + '/' + filename[:-14] + '_nps.txt'
                l_nps = []
                if path.exists(nps_path):
                    with open(nps_path, 'r') as in_fd:
                        for ln in in_fd:
                            l_nps.append(ln.strip())
                        in_fd.close()
                doc_vect = doc_embedding(cls_graph, l_nps, embedding_type, doc_id, db_cur)
                doc_vect_str = ','.join([str(ele) for ele in doc_vect])
                db_cur.execute(db_update_sql_str, (doc_vect_str, doc_id))
                cnt += 1
    db_conn.commit()
    db_conn.close()
    logging.debug('update_doc_embeddings: %s is done with %s on %s docs in %s secs.'
                  % (txt_set_name, embedding_type, cnt, (time.time() - timer_start)))


def doc_sim(txt_set_name, embed_type):
    timer_start = time.time()
    db_folder = '/home/mf3jh/workspace/data/docsim/'
    vect_db_path = db_folder + txt_set_name + '_doc_embed.db'
    vect_db_conn = sqlite3.connect(vect_db_path)
    vect_db_cur = vect_db_conn.cursor()
    vect_sql_str = '''select doc_id, %s from docs''' % embed_type
    sim_db_path = db_folder + txt_set_name + '.db'
    sim_db_conn = sqlite3.connect(sim_db_path)
    sim_db_cur = sim_db_conn.cursor()
    sim_sql_str = '''update docs_sim set %s_lexvec = ? WHERE doc_id_pair = ? or doc_id_pair = ?''' % embed_type

    if embed_type == 'gp_avg_vect':
        dsctp_int_folder = '/home/mf3jh/workspace/data/docsim/dsctp_int_rets/dsctp_int_rets/%s_nasari_30_rmswcbwexpws_w3-3/' \
                                % (txt_set_name)
        cnt = 0
        for (dirpath, dirname, filenames) in walk(dsctp_int_folder):
            for filename in filenames:
                if filename[-5:] == '.json':
                    doc_ids = [ele.strip() for ele in filename[:-5].split('#')]
                    doc_id_1 = doc_ids[0]
                    doc_id_2 = doc_ids[1]
                    if txt_set_name == '20news50short10':
                        doc_id_1 = re.sub(r'_', '/', doc_id_1)
                        doc_id_2 = re.sub(r'_', '/', doc_id_2)
                    doc_id_pair_1 = doc_id_1 + '#' + doc_id_2
                    doc_id_pair_2 = doc_id_2 + '#' + doc_id_1
                    with open(dirpath + '/' + filename, 'r') as in_fd:
                        int_ret_json = json.load(in_fd)
                        in_fd.close()
                    doc_sim = 0.0
                    for sent_pair_key in int_ret_json['sentence_pair']:
                        for cycle in int_ret_json['sentence_pair'][sent_pair_key]['cycles']:
                            l_s1_words = []
                            l_s2_words = []
                            for token in cycle:
                                if token[3] == 'L':
                                    token_lemma = token.split('#')[5].split(':')[0].strip()
                                    if token[:2] == 's1':
                                        l_s1_words.append(token_lemma)
                                    elif token[:2] == 's2':
                                        l_s2_words.append(token_lemma)
                            s1_embed = phrase_embedding(' '.join(l_s1_words))
                            s2_embed = phrase_embedding(' '.join(l_s2_words))
                            sim = 1.0 - scipyd.cosine(s1_embed, s2_embed)
                            doc_sim += sim
                    sim_db_cur.execute(sim_sql_str, (doc_sim, doc_id_pair_1, doc_id_pair_2))
                    cnt += 1
    elif embed_type == 'cls_phrases':
        sem_unit_folder_format = '/home/mf3jh/workspace/data/docsim/sem_units_%s/' % txt_set_name
        sim_docpair_query_sql_str = '''select doc_id_pair from docs_sim'''
        sim_db_cur.execute(sim_docpair_query_sql_str)
        l_docpairs = sim_db_cur.fetchall()
        cnt = 0
        for doc_pair in l_docpairs:
            doc_ids = [doc_id.strip() for doc_id in doc_pair[0].split('#')]
            doc_id_1 = doc_ids[0]
            doc_id_2 = doc_ids[1]
            if txt_set_name == '20news50short10':
                doc_id_1 = re.sub(r'/', '_', doc_id_1)
                doc_id_2 = re.sub(r'/', '_', doc_id_2)
            cls_graph_1 = nx.read_gml(sem_unit_folder_format + doc_id_1 + '_cls_graph.gml')
            cls_graph_2 = nx.read_gml(sem_unit_folder_format + doc_id_2 + '_cls_graph.gml')
            l_nn_1 = []
            l_vn_1 = []
            l_nn_2 = []
            l_vn_2 = []
            for edge in cls_graph_1.edges():
                node_1 = edge[0]
                node_2 = edge[1]
                node_1_pos = cls_graph_1.nodes(data=True)[node_1]['pos']
                node_2_pos = cls_graph_1.nodes(data=True)[node_2]['pos']
                if node_1_pos == 'NOUN' and node_2_pos == 'NOUN':
                    node_1_txt = node_1.split('|')[-1].strip()
                    node_2_txt = node_2.split('|')[-1].strip()
                    node_1_vect = phrase_embedding(node_1_txt)
                    node_2_vect = phrase_embedding(node_2_txt)
                    edge_vect = (node_1_vect + node_2_vect) / 2
                    l_nn_1.append(edge_vect)
                elif (node_1_pos == 'NOUN' and node_2_pos == 'VERB') or (node_1_pos == 'VERB' and node_2_pos == 'NOUN'):
                    node_1_txt = node_1.split('|')[-1].strip()
                    node_2_txt = node_2.split('|')[-1].strip()
                    node_1_vect = phrase_embedding(node_1_txt)
                    node_2_vect = phrase_embedding(node_2_txt)
                    edge_vect = (node_1_vect + node_2_vect) / 2
                    l_vn_1.append(edge_vect)
            for edge in cls_graph_2.edges():
                node_1 = edge[0]
                node_2 = edge[1]
                node_1_pos = cls_graph_2.nodes(data=True)[node_1]['pos']
                node_2_pos = cls_graph_2.nodes(data=True)[node_2]['pos']
                if node_1_pos == 'NOUN' and node_2_pos == 'NOUN':
                    node_1_txt = node_1.split('|')[-1].strip()
                    node_2_txt = node_2.split('|')[-1].strip()
                    node_1_vect = phrase_embedding(node_1_txt)
                    node_2_vect = phrase_embedding(node_2_txt)
                    edge_vect = (node_1_vect + node_2_vect) / 2
                    l_nn_2.append(edge_vect)
                elif (node_1_pos == 'NOUN' and node_2_pos == 'VERB') or (node_1_pos == 'VERB' and node_2_pos == 'NOUN'):
                    node_1_txt = node_1.split('|')[-1].strip()
                    node_2_txt = node_2.split('|')[-1].strip()
                    node_1_vect = phrase_embedding(node_1_txt)
                    node_2_vect = phrase_embedding(node_2_txt)
                    edge_vect = (node_1_vect + node_2_vect) / 2
                    l_vn_2.append(edge_vect)
            doc_sim = 0.0
            for nn_vect_1 in l_nn_1:
                for nn_vect_2 in l_nn_2:
                    sim = 1.0 - scipyd.cosine(nn_vect_1, nn_vect_2)
                    doc_sim += sim
            for vn_vect_1 in l_vn_1:
                for vn_vect_2 in l_vn_2:
                    sim = 1.0 - scipyd.cosine(vn_vect_1, vn_vect_2)
                    doc_sim += sim
            if txt_set_name == '20news50short10':
                doc_id_1 = re.sub(r'_', '/', doc_id_1)
                doc_id_2 = re.sub(r'_', '/', doc_id_2)
            doc_id_pair_1 = doc_id_1 + '#' + doc_id_2
            doc_id_pair_2 = doc_id_2 + '#' + doc_id_1
            sim_db_cur.execute(sim_sql_str, (doc_sim, doc_id_pair_1, doc_id_pair_2))
            cnt += 1
    elif embed_type == 'noun_phrases':
        '''
        BBC: sim mean = 0.10295296621839034
             sim sigma = 0.09312128915059538
        Reuters: sim mean = 0.11708250701067974
                 sim sigma = 0.10558014156887045
        '''
        l_sims = []
        sem_unit_folder_format = '/home/mf3jh/workspace/data/docsim/sem_units_%s/' % txt_set_name
        sim_docpair_query_sql_str = '''select doc_id_pair from docs_sim'''
        sim_db_cur.execute(sim_docpair_query_sql_str)
        l_docpairs = sim_db_cur.fetchall()
        cnt = 0
        for doc_pair in l_docpairs:
            doc_ids = [doc_id.strip() for doc_id in doc_pair[0].split('#')]
            doc_id_1 = doc_ids[0]
            doc_id_2 = doc_ids[1]
            if txt_set_name == '20news50short10':
                doc_id_1 = re.sub(r'/', '_', doc_id_1)
                doc_id_2 = re.sub(r'/', '_', doc_id_2)
            l_nn_1 = []
            l_nn_2 = []
            with open(sem_unit_folder_format + doc_id_1 + '_nps.txt', 'r') as in_fd:
                for noun_phrase in in_fd:
                    nn_1_vect = phrase_embedding(noun_phrase.strip())
                    l_nn_1.append(nn_1_vect)
                in_fd.close()
            with open(sem_unit_folder_format + doc_id_2 + '_nps.txt', 'r') as in_fd:
                for noun_phrase in in_fd:
                    nn_2_vect = phrase_embedding(noun_phrase.strip())
                    l_nn_2.append(nn_2_vect)
                in_fd.close()
            doc_sim = 0.0
            for nn_vect_1 in l_nn_1:
                for nn_vect_2 in l_nn_2:
                    sim = 1.0 - scipyd.cosine(nn_vect_1, nn_vect_2)
                    doc_sim += sim
                    l_sims.append(sim)
            if txt_set_name == '20news50short10':
                doc_id_1 = re.sub(r'_', '/', doc_id_1)
                doc_id_2 = re.sub(r'_', '/', doc_id_2)
            doc_id_pair_1 = doc_id_1 + '#' + doc_id_2
            doc_id_pair_2 = doc_id_2 + '#' + doc_id_1
            sim_db_cur.execute(sim_sql_str, (doc_sim, doc_id_pair_1, doc_id_pair_2))
            cnt += 1
        with open('/home/mf3jh/workspace/data/docsim/int_rets_noun_phrases_lexvec/%s_docsim_stats.txt' % txt_set_name, 'w+') as out_fd:
            out_str = '\n'.join([str(ele) for ele in l_sims])
            out_fd.write(out_str)
            out_fd.close()
    else:
        vect_db_cur.execute(vect_sql_str)
        l_vect_recs = vect_db_cur.fetchall()
        l_doc_vects = []
        for vect_rec in l_vect_recs:
            doc_id = vect_rec[0]
            if txt_set_name == '20news50short10':
                doc_id = re.sub(r'_', '/', doc_id)
            doc_vect = np.asarray([float(ele.strip()) for ele in vect_rec[1].split(',')])
            l_doc_vects.append((doc_id, doc_vect))
        vect_db_conn.close()

        # SIM STATS STARTS
        # l_nn_sims = []
        # l_vn_sims = []
        # SIM STATS ENDS

        cnt = 0
        for i in range(0, len(l_doc_vects)-1):
            for j in range(i+1, len(l_doc_vects)):
                doc_id_pair_1 = l_doc_vects[i][0] + '#' + l_doc_vects[j][0]
                doc_id_pair_2 = l_doc_vects[j][0] + '#' + l_doc_vects[i][0]
                if embed_type == 'pos_decomp':
                    doc_1_nn_vect = np.asarray(l_doc_vects[i][1][0:300])
                    doc_1_vn_vect = np.asarray(l_doc_vects[i][1][300:600])
                    # doc_1_vv_vect = np.asarray(l_doc_vects[i][1][600:])
                    doc_2_nn_vect = np.asarray(l_doc_vects[j][1][0:300])
                    doc_2_vn_vect = np.asarray(l_doc_vects[j][1][300:600])
                    # doc_2_vv_vect = np.asarray(l_doc_vects[j][1][600:])
                    nn_sim = 1.0 - scipyd.cosine(doc_1_nn_vect, doc_2_nn_vect)
                    if not np.isfinite(nn_sim):
                        nn_sim = 0.0
                    vn_sim = 1.0 - scipyd.cosine(doc_1_vn_vect, doc_2_vn_vect)
                    if not np.isfinite(vn_sim):
                        vn_sim = 0.0
                    # vv_sim = 1.0 - scipyd.cosine(doc_1_vv_vect, doc_2_vv_vect)
                    # if not np.isfinite(vv_sim):
                    #     vv_sim = 0.0

                    # SIM STATS STARTS
                    # l_nn_sims.append(nn_sim)
                    # l_vn_sims.append(vn_sim)
                    # SIM STATS ENDS

                    sim = g_d_pos_pair_weights['NOUN_NOUN'] * nn_sim \
                          + g_d_pos_pair_weights['VERB_NOUN'] * vn_sim
                    # sim = nn_sim + vn_sim + vv_sim
                else:
                    sim = 1.0 - scipyd.cosine(l_doc_vects[i][1], l_doc_vects[j][1])
                    if not np.isfinite(sim):
                        sim = 0.0
                sim_db_cur.execute(sim_sql_str, (sim, doc_id_pair_1, doc_id_pair_2))
                cnt += 1

        # SIM STATS STARTS
        # with open('sim_stats_nn.txt', 'w+') as out_fd:
        #     out_str = '\n'.join([str(ele) for ele in l_nn_sims])
        #     out_fd.write(out_str)
        #     out_fd.close()
        # with open('sim_stats_vn.txt', 'w+') as out_fd:
        #     out_str = '\n'.join([str(ele) for ele in l_vn_sims])
        #     out_fd.write(out_str)
        #     out_fd.close()
        # SIM STATS ENDS

    sim_db_conn.commit()
    sim_db_conn.close()
    logging.debug('doc_sim: %s is done with %s on %s doc pairs in %s secs.'
                  % (txt_set_name, embed_type, cnt, (time.time() - timer_start)))



def update_doc_embeddings_cp4(txt_set_name, embedding_type):
    timer_start = time.time()
    db_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    db_path = db_folder + 'tw_man_nar_data.db'
    db_conn = sqlite3.connect(db_path)
    db_cur = db_conn.cursor()
    db_update_sql_str = 'update docs set {0} = ? where tw_id = ?'.format(embedding_type)
    db_select_sql_str = '''select tw_id from docs'''
    db_cur.execute(db_select_sql_str)
    l_recs = db_cur.fetchall()
    sem_unit_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/sem_units_full/'
    s_doc_ids = set([])
    for rec in l_recs:
        doc_id = rec[0]
        s_doc_ids.add(doc_id)
    cnt = 0
    for (dirpath, dirname, filenames) in walk(sem_unit_folder):
        for filename in filenames:
            if filename[-14:] == '_cls_graph.gml':
                doc_id = filename[:-14].split('|')[0].strip()
                # TEST ONLY STARTS
                # if doc_id == 'Zq1j-BxBawVdAaV0esCp_A':
                #     print()
                # TEST ONLY ENDS
                if doc_id in s_doc_ids:
                    cls_graph = nx.read_gml(dirpath + '/' + filename)
                    nps_path = dirpath + '/' + filename[:-14] + '_nps.txt'
                    l_nps = []
                    if path.exists(nps_path):
                        with open(nps_path, 'r') as in_fd:
                            for ln in in_fd:
                                l_nps.append(ln.strip())
                            in_fd.close()
                    doc_vect = doc_embedding(cls_graph, l_nps, embedding_type, doc_id, db_cur)
                    doc_vect_str = ','.join([str(ele) for ele in doc_vect])
                    db_cur.execute(db_update_sql_str, (doc_vect_str, doc_id))
                    cnt += 1
    db_conn.commit()
    db_conn.close()
    logging.debug('update_doc_embeddings: %s is done with %s on %s docs in %s secs.'
                  % (txt_set_name, embedding_type, cnt, (time.time() - timer_start)))


def fetch_manual_narrative_labeled_tws():
    tw_raw_data_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/tw_raw_data/'
    tw_man_data_db_path = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/tw_man_nar_data.db'
    db_conn = sqlite3.connect(tw_man_data_db_path)
    db_cur = db_conn.cursor()
    sql_str = '''create table if not exists tw_man_nar (tw_id text primary key, man_nar text, lang text, ft_vec_en text, ft_vec_es text, topic_vec_en text, topic_vec_es text, single_nar real)'''
    db_cur.execute(sql_str)
    sql_str = '''insert into tw_man_nar (tw_id, man_nar, lang, ft_vec_en, ft_vec_es, topic_vec_en, topic_vec_es, single_nar) values (?,?,?,?,?,?,?,?)'''
    timer_start = time.time()
    cnt = 0
    for (dirpath, dirname, filenames) in walk(tw_raw_data_folder):
        for filename in filenames:
            if filename[-5:] == '.json':
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for tw_ln in in_fd:
                        tw_json = json.loads(tw_ln)
                        tw_id = tw_json['id_str_h']
                        tw_lang = tw_json['lang']
                        if 'extension' not in tw_json:
                            continue
                        if 'manual_narratives' not in tw_json['extension']:
                            continue
                        if (tw_json['extension']['manual_narratives'] is None) \
                                or (len(tw_json['extension']['manual_narratives']) == 0):
                            continue
                        if ('ft_vector_en' not in tw_json['extension']) \
                                and ('ft_vector_es' not in tw_json['extension']):
                            continue
                        if ('ft_vector_en' in tw_json['extension']) \
                                and ('ft_vector_es' in tw_json['extension']) \
                                and (tw_json['extension']['ft_vector_en'] is None) \
                                and (tw_json['extension']['ft_vector_es'] is None):
                            continue
                        if ('topic_vector_en' not in tw_json['extension']) \
                                and ('topic_vector_es' not in tw_json['extension']):
                            continue
                        if ('topic_vector_en' in tw_json['extension']) \
                                and ('topic_vector_es' in tw_json['extension']) \
                                and (tw_json['extension']['topic_vector_en'] is None) \
                                and (tw_json['extension']['topic_vector_es'] is None):
                            continue
                        if len(tw_json['extension']['manual_narratives']) == 1:
                            tw_single_nar = 1
                        else:
                            tw_single_nar = 0
                        tw_man_nar = '|'.join(tw_json['extension']['manual_narratives'])
                        tw_ft_vec_en = None
                        if 'ft_vector_en' in tw_json['extension'] and tw_json['extension']['ft_vector_en'] is not None:
                            tw_ft_vec_en = ','.join([str(ele) for ele in tw_json['extension']['ft_vector_en']])
                        tw_ft_vec_es = None
                        if 'ft_vector_es' in tw_json['extension'] and tw_json['extension']['ft_vector_es'] is not None:
                            tw_ft_vec_es = ','.join([str(ele) for ele in tw_json['extension']['ft_vector_es']])
                        tw_topic_vec_en = None
                        if 'topic_vector_en' in tw_json['extension'] \
                                and tw_json['extension']['topic_vector_en'] is not None:
                            tw_topic_vec_en = ','.join([str(ele) for ele in tw_json['extension']['topic_vector_en']])
                        tw_topic_vec_es = None
                        if 'topic_vector_es' in tw_json['extension'] \
                                and tw_json['extension']['topic_vector_es'] is not None:
                            tw_topic_vec_es = ','.join([str(ele) for ele in tw_json['extension']['topic_vector_es']])
                        db_cur.execute(sql_str, (tw_id, tw_man_nar, tw_lang, tw_ft_vec_en, tw_ft_vec_es,
                                                 tw_topic_vec_en, tw_topic_vec_es, tw_single_nar))
                        cnt += 1
                        if cnt % 1000 == 0 and cnt >= 1000:
                            db_conn.commit()
                            logging.debug('%s man-nar-tws have been written in %s secs.'
                                          % (cnt, (time.time() - timer_start)))
    db_conn.commit()
    logging.debug('%s man-nar-tws have been written in %s secs.'
                  % (cnt, (time.time() - timer_start)))
    db_conn.close()


def update_topic_vect_to_ven_tw_en_db():
    tw_work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    tw_raw_data_folder = tw_work_folder + 'tw_raw_data/'
    tw_db_path = tw_work_folder + 'ven_tw_en_v2-1.db'
    tw_db_conn = sqlite3.connect(tw_db_path)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = '''select tw_id from ven_tw_en'''
    tw_db_cur.execute(sql_str)
    s_tw_ids = set([rec[0].strip() for rec in tw_db_cur.fetchall()])
    sql_str = '''update ven_tw_en set topic_vect = ? where tw_id = ?'''
    timer_start = time.time()
    cnt = 0
    for (dirpath, dirname, filenames) in walk(tw_raw_data_folder):
        for filename in filenames:
            if filename[-5:] == '.json':
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for tw_ln in in_fd:
                        tw_json = json.loads(tw_ln)
                        tw_id = tw_json['id_str_h']
                        if tw_id not in s_tw_ids:
                            continue
                        tw_lang = tw_json['lang']
                        tw_topic_vec = None
                        tw_topic_vec_en = None
                        if 'topic_vector_en' in tw_json['extension'] \
                                and tw_json['extension']['topic_vector_en'] is not None:
                            tw_topic_vec_en = ','.join([str(ele) for ele in tw_json['extension']['topic_vector_en']])
                        tw_topic_vec_es = None
                        if 'topic_vector_es' in tw_json['extension'] \
                                and tw_json['extension']['topic_vector_es'] is not None:
                            tw_topic_vec_es = ','.join([str(ele) for ele in tw_json['extension']['topic_vector_es']])
                        if tw_lang == 'en':
                            if tw_topic_vec_en is not None:
                                tw_topic_vec = tw_topic_vec_en
                            elif tw_topic_vec_es is not None:
                                tw_topic_vec = tw_topic_vec_es
                        elif tw_lang == 'es':
                            if tw_topic_vec_es is not None:
                                tw_topic_vec = tw_topic_vec_es
                            elif tw_topic_vec_en is not None:
                                tw_topic_vec = tw_topic_vec_en
                        tw_db_cur.execute(sql_str, (tw_topic_vec, tw_id))
                        cnt += 1
                        if cnt % 10000 == 0 and cnt >= 10000:
                            tw_db_conn.commit()
                            logging.debug('%s topic vects are written in %s secs.' % (cnt, time.time() - timer_start))
    tw_db_conn.commit()
    logging.debug('%s topic vects are written in %s secs.' % (cnt, time.time() - timer_start))
    tw_db_conn.close()


g_cp4_ven_tw_categories = {'guaido/legitimate', 'international/respect_sovereignty', 'maduro/dictator',
                           'maduro/illegitimate', 'maduro/legitimate', 'military', 'other/restore_democracy',
                           'protests'}
g_d_cp4_ven_tw_categories = {'guaido/legitimate': 0, 'international/respect_sovereignty': 1, 'maduro/dictator': 2,
                           'maduro/illegitimate': 3, 'maduro/legitimate': 4, 'military': 5,
                             'other/restore_democracy': 6, 'protests': 7}


def create_cls_graph_exist_table():
    work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    db_path = work_folder + 'ven_tw_en_v2-1.db'
    db_conn = sqlite3.connect(db_path)
    db_cur = db_conn.cursor()
    sql_str = '''create table ven_tw_cls_exist (tw_id text primary key)'''
    db_cur.execute(sql_str)
    sql_str = '''insert into ven_tw_cls_exist (tw_id) values (?)'''

    sem_unit_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/sem_units_full/'
    cnt = 0
    for (dirpath, dirname, filenames) in walk(sem_unit_folder):
        for filename in filenames:
            if filename[-14:] == '_cls_graph.gml':
                tw_id = filename[:-14].split('|')[0].strip()
                db_cur.execute(sql_str, (tw_id,))
                cnt += 1
                if cnt % 5000 == 0 and cnt >= 5000:
                    db_conn.commit()
                    logging.debug('create_cls_graph_exist_table: %s recs are inserted.' % cnt)
    db_conn.commit()
    logging.debug('create_cls_graph_exist_table: %s recs are inserted.' % cnt)
    db_conn.close()


def cp4_ven_tw_sampling():
    work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    man_db_path = work_folder + 'tw_man_nar_data.db'
    man_db_conn = sqlite3.connect(man_db_path)
    man_db_cur = man_db_conn.cursor()
    man_db_sql_str = '''select docs.tw_id, tw_man_single_nar.man_nar from docs inner join tw_man_single_nar on docs.tw_id = tw_man_single_nar.tw_id'''

    cls_db_path = work_folder + 'ven_tw_en_v2-1.db'
    cls_db_conn = sqlite3.connect(cls_db_path)
    cls_db_cur = cls_db_conn.cursor()
    cls_db_sql_str = '''select tw_id from ven_tw_cls_exist where tw_id = ?'''

    d_nar = dict()
    man_db_cur.execute(man_db_sql_str)
    l_recs = man_db_cur.fetchall()
    for rec in l_recs:
        tw_id = rec[0]
        tw_nar = rec[1]
        cls_db_cur.execute(cls_db_sql_str, (tw_id,))
        cls_exist = cls_db_cur.fetchone()
        if cls_exist is None:
            continue
        if tw_nar not in d_nar:
            d_nar[tw_nar] = [tw_id]
        else:
            d_nar[tw_nar].append(tw_id)

    man_db_conn.close()
    cls_db_conn.close()

    l_sel_cats = [nar for nar in d_nar if len(d_nar[nar]) >= 100]
    d_samples = dict()
    for cat in l_sel_cats:
        l_tw_ids = d_nar[cat]
        l_sample_tw_ids = sample(l_tw_ids, 100)
        d_samples[cat] = l_sample_tw_ids

    with open(work_folder + 'tw_man_nar_8cat_samples.json', 'w+') as out_fd:
        json.dump(d_samples, out_fd)
        out_fd.close()


def create_docsim_table():
    work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    db_path = work_folder + 'tw_man_nar_data.db'
    db_conn = sqlite3.connect(db_path)
    db_cur = db_conn.cursor()
    db_sql_str = '''create table if not exists docs_sim_8cat (doc_pair text primary key)'''
    db_cur.execute(db_sql_str)
    db_sql_str = '''insert into docs_sim_8cat (doc_pair) values (?)'''

    tw_sample_json_path = work_folder + 'tw_man_nar_8cat_samples.json'
    with open(tw_sample_json_path, 'r') as in_fd:
        d_samples = json.load(in_fd)
        in_fd.close()
    l_sample_tw_ids = []
    for cat in d_samples:
        l_sample_tw_ids += d_samples[cat]

    for i in range(0, len(l_sample_tw_ids)-1):
        for j in range(i+1, len(l_sample_tw_ids)):
            db_cur.execute(db_sql_str, (l_sample_tw_ids[i] + '|' + l_sample_tw_ids[j],))
    db_conn.commit()
    db_conn.close()


def update_ft_vect_topic_vect():
    work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    docs_db_path = work_folder + 'tw_man_nar_data.db'
    docs_db_conn = sqlite3.connect(docs_db_path)
    docs_db_cur = docs_db_conn.cursor()
    docs_insert_sql_str = '''update docs set ft_vect = ?, topic_vect = ? where tw_id = ?'''
    docs_query_sql_str = '''select tw_id from docs'''
    man_sql_str = '''select lang, ft_vec_en, ft_vec_es, topic_vec_en, topic_vec_es from tw_man_single_nar where tw_id = ?'''

    docs_db_cur.execute(docs_query_sql_str)
    l_recs = docs_db_cur.fetchall()
    l_tw_ids = [rec[0].strip() for rec in l_recs]

    cnt = 0
    for tw_id in l_tw_ids:
        docs_db_cur.execute(man_sql_str, (tw_id,))
        rec = docs_db_cur.fetchone()
        lang = rec[0]
        ft_vec_en_str = rec[1]
        ft_vec_es_str = rec[2]
        topic_vec_en_str = rec[3]
        topic_vec_es_str = rec[4]
        ft_vect = None
        topic_vect = None
        if lang == 'en':
            if ft_vec_en_str is not None:
                ft_vect = ft_vec_en_str
            elif ft_vec_es_str is not None:
                ft_vect = ft_vec_es_str
            if topic_vec_en_str is not None:
                topic_vect = topic_vec_en_str
            elif topic_vec_es_str is not None:
                topic_vect = topic_vec_es_str
        elif lang == 'es':
            if ft_vec_es_str is not None:
                ft_vect = ft_vec_es_str
            elif ft_vec_en_str is not None:
                ft_vect = ft_vec_en_str
            if topic_vec_es_str is not None:
                topic_vect = topic_vec_es_str
            elif topic_vec_en_str is not None:
                topic_vect = topic_vec_en_str
        else:
            if ft_vec_en_str is not None:
                ft_vect = ft_vec_en_str
            elif ft_vec_es_str is not None:
                ft_vect = ft_vec_es_str
            if topic_vec_en_str is not None:
                topic_vect = topic_vec_en_str
            elif topic_vec_es_str is not None:
                topic_vect = topic_vec_es_str
        docs_db_cur.execute(docs_insert_sql_str, (ft_vect, topic_vect, tw_id))
        cnt += 1
        if cnt % 5000 == 0 and cnt >= 5000:
            docs_db_conn.commit()
            logging.debug('update_ft_vect_topic_vect: %s ft_vect and topic_vect are updated.' % cnt)
    docs_db_conn.commit()
    logging.debug('update_ft_vect_topic_vect: %s ft_vect and topic_vect are updated.' % cnt)


def cp4_ven_doc_sim(txt_set_name, embed_type):
    timer_start = time.time()
    db_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    vect_db_path = db_folder + txt_set_name + '.db'
    vect_db_conn = sqlite3.connect(vect_db_path)
    vect_db_cur = vect_db_conn.cursor()
    vect_sql_str = '''select %s from docs where tw_id = ?''' % embed_type
    sim_db_path = db_folder + txt_set_name + '.db'
    sim_db_conn = sqlite3.connect(sim_db_path)
    sim_db_cur = sim_db_conn.cursor()
    sim_sql_str = '''update docs_sim_8cat set %s_lexvec = ? WHERE doc_pair = ? or doc_pair = ?''' % embed_type

    tw_sample_json_path = db_folder + 'tw_man_nar_8cat_samples.json'
    with open(tw_sample_json_path, 'r') as in_fd:
        d_samples = json.load(in_fd)
        in_fd.close()
    l_sample_tw_ids = []
    for cat in d_samples:
        l_sample_tw_ids += d_samples[cat]

    d_doc_vects = dict()
    for tw_id in l_sample_tw_ids:
        vect_db_cur.execute(vect_sql_str, (tw_id,))
        rec = vect_db_cur.fetchone()[0]
        doc_vect = [float(ele.strip()) for ele in rec.split(',')]
        d_doc_vects[tw_id] = doc_vect
    vect_db_conn.close()

    cnt = 0
    for i in range(0, len(l_sample_tw_ids)-1):
        for j in range(i+1, len(l_sample_tw_ids)):
            doc_id_pair_1 = l_sample_tw_ids[i] + '|' + l_sample_tw_ids[j]
            doc_id_pair_2 = l_sample_tw_ids[j] + '|' + l_sample_tw_ids[i]
            # TEST ONLY STARTS
            # if doc_id_pair_1 == '-6yWDKi1IrLR1kHeKgj8zA|RwbJvq-UnD7VlDwiP7YLnA' \
            #     or doc_id_pair_2 == '-6yWDKi1IrLR1kHeKgj8zA|RwbJvq-UnD7VlDwiP7YLnA':
            #     print()
            # TEST ONLY ENDS
            if embed_type == 'pos_decomp':
                doc_1_nn_vect = np.asarray(d_doc_vects[l_sample_tw_ids[i]][0:300])
                doc_1_vn_vect = np.asarray(d_doc_vects[l_sample_tw_ids[i]][300:600])
                # doc_1_vv_vect = np.asarray(d_doc_vects[l_sample_tw_ids[i]][600:])
                doc_2_nn_vect = np.asarray(d_doc_vects[l_sample_tw_ids[j]][0:300])
                doc_2_vn_vect = np.asarray(d_doc_vects[l_sample_tw_ids[j]][300:600])
                # doc_2_vv_vect = np.asarray(d_doc_vects[l_sample_tw_ids[j]][600:])
                nn_sim = 1.0 - scipyd.cosine(doc_1_nn_vect, doc_2_nn_vect)
                if not np.isfinite(nn_sim):
                    nn_sim = 0.0
                vn_sim = 1.0 - scipyd.cosine(doc_1_vn_vect, doc_2_vn_vect)
                if not np.isfinite(vn_sim):
                    vn_sim = 0.0
                # vv_sim = 1.0 - scipyd.cosine(doc_1_vv_vect, doc_2_vv_vect)
                # if not np.isfinite(vv_sim):
                #     vv_sim = 0.0
                # sim = g_d_pos_pair_weights['NOUN_NOUN'] * nn_sim \
                #       + g_d_pos_pair_weights['VERB_NOUN'] * vn_sim \
                #       + g_d_pos_pair_weights['VERB_VERB'] * vv_sim
                sim = nn_sim + vn_sim
            else:
                sim = 1.0 - scipyd.cosine(d_doc_vects[l_sample_tw_ids[i]], d_doc_vects[l_sample_tw_ids[j]])
                if not np.isfinite(sim):
                    sim = 0.0
            sim_db_cur.execute(sim_sql_str, (sim, doc_id_pair_1, doc_id_pair_2))
            cnt += 1
    sim_db_conn.commit()
    sim_db_conn.close()
    logging.debug('doc_sim: %s is done with %s on %s doc pairs in %s secs.'
                  % (txt_set_name, embed_type, cnt, (time.time() - timer_start)))


def cp4_ven_doc_sim_ph_comp():
    work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    tw_sample_json_path = work_folder + 'tw_man_nar_8cat_samples.json'
    with open(tw_sample_json_path, 'r') as in_fd:
        d_samples = json.load(in_fd)
        in_fd.close()
    l_sample_tw_ids = []
    for cat in d_samples:
        l_sample_tw_ids += d_samples[cat]


def cp4_ven_doc_clustering(sim_col):
    db_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    # db_path = db_folder + 'tw_man_nar_data.db'
    out_fd = open(db_folder + 'tw_man_nar_doc_cluster_%s.txt' % sim_col, 'w+')
    total_doc = 800

    tw_to_id, id_to_tw = cp4_ven_doc_clustering_get_doc_ids()
    org_doc_labels = cp4_ven_doc_clustering_label_org_doc_ids(tw_to_id, g_d_cp4_ven_tw_categories, total_doc)
    aff_matrix = None
    kmeans_matrix = None
    for n_size in [8, 9, 10, 11, 12]:
        if aff_matrix is None:
            l_doc_pair_sims = cp4_ven_doc_clustering_get_doc_sim_from_db(sim_col)
            aff_matrix = cp4_ven_doc_clustering_build_aff_matrix(total_doc, l_doc_pair_sims, tw_to_id)
        if aff_matrix[0][0] != 0:
            raise Exception("Aff matrix [i][i] not equal to 0!!")
        sc_labels = SpectralClustering(n_clusters=n_size,
                                       eigen_solver=None,
                                       random_state=None,
                                       n_init=10,
                                       affinity='precomputed',
                                       assign_labels='kmeans',
                                       n_jobs=-1).fit_predict(aff_matrix)

        # if kmeans_matrix is None and aff_matrix is not None:
        #     kmeans_matrix = copy.deepcopy(aff_matrix)
        #     for i in range(total_doc):
        #         kmeans_matrix[i][i] = 1
        print("\n%s Clustering [n=%s]\n" % ('Spectral', n_size))
        out_fd.write("\n\n%s Clustering [n=%s]\n" % ('Spectral', n_size))
        out_fd.write("\n%s\n" % cp4_ven_doc_clustering_print_docs_by_labels(sc_labels, id_to_tw))
        # cp4_ven_doc_clustering_print_cluster_perform(kmeans_matrix, sc_labels, 'precomputed', out_fd)
        p4_ven_doc_clustering_cluster_perf_evaluation(org_doc_labels, sc_labels, out_fd)
    out_fd.close()


def cp4_ven_doc_clustering_label_org_doc_ids(doc_ids, doc_categories, size):
    work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    tw_sample_json_path = work_folder + 'tw_man_nar_8cat_samples.json'
    with open(tw_sample_json_path, 'r') as in_fd:
        d_samples = json.load(in_fd)
        in_fd.close()
    doc_labels = [-1] * size
    for doc_id, loc in doc_ids.items():
        for cat in d_samples:
            if doc_id in d_samples[cat]:
                doc_labels[loc] = doc_categories[cat]
    if -1 in doc_labels:
        raise Exception("Org doc labels has -1!!")
    return doc_labels


def cp4_ven_doc_clustering_print_cluster_perform(aff_mat, cluster_labels, ss_metric, outfile):
    ss = metrics.silhouette_score(aff_mat, cluster_labels, metric=ss_metric)
    print('[INF]: Sihouette Score = %s' % ss)
    ch = metrics.calinski_harabaz_score(aff_mat, cluster_labels)
    print('[INF]: Calinski-Harabaz Index = %s' % ch)
    db = metrics.davies_bouldin_score(aff_mat, cluster_labels)
    print('[INF]: Davies-Bouldin Index = %s' % db)
    # return ss, ch, db
    outfile.write('\tSihouette Score = %s\n\tCalinski-Harabaz Index = %s\n\tDavies-Bouldin Index = %s' % (ss, ch, db))


def p4_ven_doc_clustering_cluster_perf_evaluation(org_labels, res_labels, outfile):
    ars = metrics.adjusted_rand_score(org_labels, res_labels)
    print('[INF]: Adjusted Rand Score = %s' % ars)
    amis = metrics.normalized_mutual_info_score(org_labels, res_labels)
    print('[INF]: Normalized Mutual Info Score = %s' % amis)
    hs = metrics.homogeneity_score(org_labels, res_labels)
    print('[INF]: Homogeneity Score = %s' % hs)
    cs = metrics.completeness_score(org_labels, res_labels)
    print('[INF]: Completeness Score = %s' % cs)
    vms = metrics.v_measure_score(org_labels, res_labels)
    print('[INF]: V-measure Score = %s' % vms)
    fmi = metrics.fowlkes_mallows_score(org_labels, res_labels)
    print('[INF]: Fowlkes-Mallows Score = %s' % fmi)
    outfile.write("""\n\tAdjusted Rand Score = %s\n\tNormalized Mutual Info Score = %s\n\tHomogeneity Score = %s
                  \n\tCompleteness Score = %s\n\tV-measure Score = %s\n\tFowlkes-Mallows Score = %s\n""" % (ars, amis,hs,cs,vms, fmi))


def cp4_ven_doc_clustering_print_docs_by_labels(labels, id_to_doc):
    label_list = dict()
    for i, k in enumerate(labels):
        if k not in label_list.keys():
            label_list[k] = [id_to_doc[i]]
        else:
            label_list[k].append(id_to_doc[i])
    return label_list


def cp4_ven_doc_clustering_build_aff_matrix(mat_size, l_doc_pair_sims, tw_to_id):
    aff_mat = np.zeros([mat_size, mat_size], dtype=float)
    minv = -1
    maxv = 1
    for doc_pair, sim in l_doc_pair_sims:
        normed_sim = (sim-minv)/(maxv-minv)
        l_tws = [tw.strip() for tw in doc_pair.split('|')]
        xidx = tw_to_id[l_tws[0]]
        yidx = tw_to_id[l_tws[1]]
        aff_mat[xidx][yidx] = normed_sim
        aff_mat[yidx][xidx] = normed_sim
    return aff_mat


def cp4_ven_doc_clustering_get_doc_sim_from_db(sim_col):
    db_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    db_path = db_folder + 'tw_man_nar_data.db'
    db_conn = sqlite3.connect(db_path)
    db_cur = db_conn.cursor()
    sql_str = '''select doc_pair, %s from docs_sim_8cat order by doc_pair''' % sim_col
    db_cur.execute(sql_str)
    l_doc_pair_sims = db_cur.fetchall()
    return l_doc_pair_sims


def cp4_ven_doc_clustering_get_doc_ids():
    work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
    tw_sample_json_path = work_folder + 'tw_man_nar_8cat_samples.json'
    with open(tw_sample_json_path, 'r') as in_fd:
        d_samples = json.load(in_fd)
        in_fd.close()
    l_sample_tw_ids = []
    for cat in d_samples:
        l_sample_tw_ids += d_samples[cat]
    l_sample_tw_ids = sorted(l_sample_tw_ids)

    tw_to_id = dict()
    id_to_tw = dict()
    for idx, tw in enumerate(l_sample_tw_ids):
        tw_to_id[tw] = idx
        id_to_tw[idx] = tw
    return tw_to_id, id_to_tw

g_cp4_ven_tw_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
g_cp4_ven_tw_embed_int_folder = g_cp4_ven_tw_folder + 'cp4_ven_tw_embed_int/'
g_sem_unit_folder = g_cp4_ven_tw_folder + 'sem_units_full/'

def cp4_ven_full_doc_embedding_thread(l_tw_ids, txt_set_name, embedding_type, db_cur, t_id):
    int_out_file = g_cp4_ven_tw_embed_int_folder + embedding_type + '#' + str(t_id) + '.json'
    d_tw_embed = dict()
    cnt = 0
    timer_start = time.time()
    for tw_id in l_tw_ids:
        cls_graph_file_path = g_sem_unit_folder + tw_id + '_cls_graph.gml'
        nps_file_path = g_sem_unit_folder + tw_id + '_nps.txt'
        cls_graph = nx.read_gml(cls_graph_file_path)
        l_nps = []
        with open(nps_file_path, 'r') as in_fd:
            for ln in in_fd:
                l_nps.append(ln.strip())
            in_fd.close()
        doc_vect = doc_embedding(cls_graph, l_nps, embedding_type, tw_id, db_cur)
        d_tw_embed[tw_id] = ','.join([str(ele) for ele in doc_vect])
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('cp4_ven_full_doc_embedding_thread %s: %s is done with %s on %s docs in %s secs.'
                          % (t_id, txt_set_name, embedding_type, cnt, (time.time() - timer_start)))
    with open(int_out_file, 'w+') as out_fd:
        json.dump(d_tw_embed, out_fd)
        out_fd.close()
    logging.debug('cp4_ven_full_doc_embedding_thread %s: %s is done with %s on %s docs in %s secs.'
                  % (t_id, txt_set_name, embedding_type, cnt, (time.time() - timer_start)))


def cp4_ven_full_doc_embedding_multithreads(op_func, l_full_tw_ids, txt_set_name, embedding_type, db_cur):
    batch_size = math.ceil(len(l_full_tw_ids) / multiprocessing.cpu_count())
    l_l_tw_ids = []
    for i in range(0, len(l_full_tw_ids), batch_size):
        if i + batch_size < len(l_full_tw_ids):
            l_l_tw_ids.append(l_full_tw_ids[i:i + batch_size])
        else:
            l_l_tw_ids.append(l_full_tw_ids[i:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_tw_ids:
        t = threading.Thread(target=op_func, args=(l_each_batch, txt_set_name, embedding_type, db_cur, str(t_id)))
        t.setName('cp4_embed_thread_' + str(t_id))
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

    logging.debug('All done!')


def cp4_ven_full_doc_embedding():
    txt_set_name = 'ven_tw_en_v2-1'
    # l_embed_types = ['avg_nps_vect', 'w_avg_edge_vect']
    l_embed_types = ['sp_w_avg_edge_vect']
    tw_db_conn = sqlite3.connect(g_cp4_ven_tw_folder + txt_set_name + '.db')
    tw_db_cur = tw_db_conn.cursor()
    sql_str = '''select tw_id from ven_tw_cls_exist'''
    tw_db_cur.execute(sql_str)
    l_full_tw_ids = [rec[0] for rec in tw_db_cur.fetchall()]

    # for embed_type in l_embed_types:
    #     cp4_ven_full_doc_embedding_multithreads(cp4_ven_full_doc_embedding_thread, l_full_tw_ids, txt_set_name,
    #                                             embed_type, tw_db_cur)

    cp4_ven_full_doc_embedding_write_db(tw_db_cur, tw_db_conn, l_embed_types)

    l_embed_types = ['avg_nps_vect_sp_w_avg_edge_vect']
    for embed_type in l_embed_types:
        cp4_ven_full_doc_embedding_thread(l_full_tw_ids, txt_set_name, embed_type, tw_db_cur, 0)
        # cp4_ven_full_doc_embedding_multithreads(cp4_ven_full_doc_embedding_thread, l_full_tw_ids, txt_set_name,
        #                                         embed_type, tw_db_cur)

    cp4_ven_full_doc_embedding_write_db(tw_db_cur, tw_db_conn, l_embed_types)
    tw_db_conn.close()


def cp4_ven_full_doc_embedding_write_db(tw_db_cur, tw_db_conn, l_embed_types):
    update_sql_str_format = 'update ven_tw_en set {0} = ? where tw_id = ?'
    cnt = 0
    for (dirpath, dirname, filenames) in walk(g_cp4_ven_tw_embed_int_folder):
        for filename in filenames:
            embed_type = filename.split('#')[0].strip()
            if embed_type not in l_embed_types:
                continue
            with open(dirpath + '/' + filename, 'r') as in_fd:
                d_tw_vects = json.load(in_fd)
                in_fd.close()
            for tw_id in d_tw_vects:
                tw_vect_str = d_tw_vects[tw_id]
                tw_db_cur.execute(update_sql_str_format.format(embed_type), (tw_vect_str, tw_id))
                cnt += 1
                if cnt % 50000 == 0 and cnt >= 50000:
                    tw_db_conn.commit()
                    logging.debug('%s tw vects are written to db.' % cnt)
    tw_db_conn.commit()
    logging.debug('%s tw vects are written to db.' % cnt)


def write_vects_to_txt():
    tw_db_conn = sqlite3.connect(g_cp4_ven_tw_folder + 'ven_tw_en_v2-1.db')
    tw_db_cur = tw_db_conn.cursor()
    sql_str = '''select tw_id, avg_nps_vect, avg_nps_vect_w_avg_edge_vect, topic_vect, avg_nps_vect_sp_w_avg_edge_vect from ven_tw_en where avg_nps_vect is not null and avg_nps_vect_w_avg_edge_vect is not null and topic_vect is not null and avg_nps_vect_sp_w_avg_edge_vect is not null'''
    # sql_str = '''select tw_id, topic_vect from ven_tw_en where topic_vect is not null'''
    l_recs = tw_db_cur.execute(sql_str)
    avg_nps_vect_out_fd = open(g_cp4_ven_tw_folder + 'ven_tw_v2-1_avg_nps_vect.txt', 'w+')
    avg_nps_vect_out_fd.write('tw_id|avg_nps_vect')
    avg_nps_vect_out_fd.write('\n')
    w_avg_edge_vect_out_fd = open(g_cp4_ven_tw_folder + 'ven_tw_v2-1_w_avg_edge_vect.txt', 'w+')
    w_avg_edge_vect_out_fd.write('tw_id|w_avg_edge_vect')
    w_avg_edge_vect_out_fd.write('\n')
    topic_vect_out_fd = open(g_cp4_ven_tw_folder + 'ven_tw_v2-1_topic_vect.txt', 'w+')
    topic_vect_out_fd.write('tw_id|topic_vect')
    topic_vect_out_fd.write('\n')
    avg_nps_vect_sp_w_avg_edge_vect_out_fd = open(g_cp4_ven_tw_folder + 'ven_tw_v2-1_avg_nps_vect_sp_w_avg_edge_vect.txt', 'w+')
    avg_nps_vect_sp_w_avg_edge_vect_out_fd.write('\n')
    cnt = 0
    for rec in l_recs:
        tw_id = rec[0]
        avg_nps_vect = rec[1]
        w_avg_edge_vect = rec[2]
        topic_vect = rec[3]
        avg_nps_vect_sp_w_avg_edge_vect = rec[4]
        avg_nps_vect_out_fd.write(tw_id + '|' + avg_nps_vect.strip())
        avg_nps_vect_out_fd.write('\n')
        w_avg_edge_vect_out_fd.write(tw_id + '|' + w_avg_edge_vect.strip())
        w_avg_edge_vect_out_fd.write('\n')
        topic_vect_out_fd.write(tw_id + '|' + topic_vect.strip())
        topic_vect_out_fd.write('\n')
        avg_nps_vect_sp_w_avg_edge_vect_out_fd.write(tw_id + '|' + avg_nps_vect_sp_w_avg_edge_vect.strip())
        avg_nps_vect_sp_w_avg_edge_vect_out_fd.write('\n')
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('%s vectors are written.' % cnt)
    logging.debug('%s vectors are written.' % cnt)
    avg_nps_vect_out_fd.close()
    w_avg_edge_vect_out_fd.close()
    topic_vect_out_fd.close()
    avg_nps_vect_sp_w_avg_edge_vect_out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # fetch_manual_narrative_labeled_tws()

    # l_txt_set_names = ['20news50short10', 'bbc', 'reuters']
    # l_txt_set_names = ['bbc', 'reuters']
    # l_txt_set_names = ['tw_man_nar_data']
    # for txt_set_name in l_txt_set_names:
    #     build_txt_db(txt_set_name)

    # l_cols = [('avg_node_vect', 'text'), ('avg_edge_vect', 'text'), ('s_avg_edge_vect', 'text'),
    #           ('w_avg_edge_vect', 'text'), ('sp_avg_node_vect', 'text'), ('sp_w_avg_edge_vect', 'text'),
    #           ('ft_vect', 'text'), ('topic_vect', 'text')]
    # l_cols = [('avg_nps_vect', 'text')]
    # for txt_set_name in l_txt_set_names:
    #     for col_name, col_type in l_cols:
    #         alter_table_add_col(txt_set_name, col_name, col_type)

    # l_embed_types = ['avg_node_vect', 'avg_edge_vect', 's_avg_edge_vect',
    #                  'w_avg_edge_vect', 'sp_avg_node_vect', 'sp_w_avg_edge_vect', 'avg_nps_vect']
    # l_embed_types = ['ft_vect', 'topic_vect']
    # l_embed_types = ['avg_nps_vect_avg_node_vect', 'avg_nps_vect_avg_edge_vect', 'avg_nps_vect_s_avg_edge_vect',
    #                  'avg_nps_vect_w_avg_edge_vect', 'avg_nps_vect_sp_avg_node_vect', 'avg_nps_vect_sp_w_avg_edge_vect',
    #                  'pos_decomp', 'gp_avg_vect', 'cls_phrases']
    # l_embed_types = ['avg_nps_vect_w_avg_edge_vect']
    # for txt_set_name in l_txt_set_names:
    #     for embed_type in l_embed_types:
    #         update_doc_embeddings(txt_set_name, embed_type)
            # update_doc_embeddings_cp4(txt_set_name, embed_type)
    # update_ft_vect_topic_vect()

    # l_sim_col = ['avg_node_vect_lexvec', 'avg_edge_vect', 's_avg_edge_vect',
    #                  'w_avg_edge_vect', 'sp_avg_node_vect', 'sp_w_avg_edge_vect']
    # for txt_set_name in l_txt_set_names:
    #     for embed_type in l_embed_types:
    #         doc_sim(txt_set_name, embed_type)
            # cp4_ven_doc_sim(txt_set_name, embed_type)

    # cp4_ven_tw_sampling()
    # create_docsim_table()
    # create_cls_graph_exist_table()

    # for txt_set_name in l_txt_set_names:
    #     for embed_type in l_embed_types:
    #         cp4_ven_doc_clustering(embed_type + '_lexvec')

    # cp4_ven_full_doc_embedding()
    write_vects_to_txt()
    # update_topic_vect_to_ven_tw_en_db()