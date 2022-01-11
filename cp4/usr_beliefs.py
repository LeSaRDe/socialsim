import logging
import json
import networkx as nx
import time
import os
from os import walk, path
import sqlite3
import sys
sys.path.insert(1, '/home/mf3jh/workspace/lib/lexvec/lexvec/python/lexvec/')
import model as lexvec
import scipy.spatial.distance as scipyd
import numpy as np


version = 'v2-1'
g_ven_tw_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
g_sel_usr_list = g_ven_tw_folder + 'ven_tw_resp_sel_usrs_50_100.txt'
g_sem_units_folder = g_ven_tw_folder + 'sem_units_full/'
g_usr_beliefs_folder = g_ven_tw_folder + 'usr_beliefs/'
g_resp_db = g_ven_tw_folder + 'ven_tw_resp_' + version + '.db'
g_tw_db = g_ven_tw_folder + 'ven_tw_en_' + version + '.db'
g_word_embedding_model = 'lexvec'
g_node_sim_threshold = 0.6
g_tw_datetime_cutoff = ''


def renama_sem_unit_files():
    for (dirpath, dirname, filenames) in walk(g_sem_units_folder):
        for filename in filenames:
            if filename[-14:] == '_cls_graph.gml':
                tw_id = filename.split('|')[0].strip()
                os.rename(dirpath + '/' + filename, dirpath + '/' + tw_id + '_cls_graph.gml')
            elif filename[-8:] == '_nps.txt':
                tw_id = filename.split('|')[0].strip()
                os.rename(dirpath + '/' + filename, dirpath + '/' + tw_id + '_nps.txt')


def get_usr_list():
    l_usrs = []
    with open(g_sel_usr_list, 'r') as in_fd:
        for ln in in_fd:
            l_usrs.append(ln.strip())
        in_fd.close()
    return l_usrs


def build_usr_belief_graph(usr_id):
    resp_db_conn = sqlite3.connect(g_resp_db)
    res_db_cur = resp_db_conn.cursor()
    resp_sql_str = '''select retweets, replies, quotes, originals from ven_resp_by_usrs where usr_id = ?'''
    res_db_cur.execute(resp_sql_str, (usr_id,))
    rec = res_db_cur.fetchone()
    retweets = rec[0]
    replies = rec[1]
    quotes = rec[2]
    originals = rec[3]
    l_t_pairs = [[tw_id.strip() for tw_id in pair.strip().split('|')] for pair in retweets.split('\n')] if retweets is not None else []
    l_r_pairs = [[tw_id.strip() for tw_id in pair.strip().split('|')] for pair in replies.split('\n')] if replies is not None else []
    l_q_pairs = [[tw_id.strip() for tw_id in pair.strip().split('|')] for pair in quotes.split('\n')] if quotes is not None else []
    l_n_pairs = [[tw_id.strip() for tw_id in pair.strip().split('|')] for pair in originals.split('\n')] if originals is not None else []
    ub_graph = nx.DiGraph()
    for from_tw_id, to_tw_id, tw_datetime in l_t_pairs:
        ub_graph = add_sem_units_into_usr_belief_graph(ub_graph, from_tw_id, to_tw_id)
    for from_tw_id, to_tw_id, tw_datetime in l_r_pairs:
        ub_graph = add_sem_units_into_usr_belief_graph(ub_graph, from_tw_id, to_tw_id)
    for from_tw_id, to_tw_id, tw_datetime in l_q_pairs:
        ub_graph = add_sem_units_into_usr_belief_graph(ub_graph, from_tw_id, to_tw_id)
    for from_tw_id, to_tw_id, tw_datetime in l_n_pairs:
        ub_graph = add_sem_units_into_usr_belief_graph(ub_graph, from_tw_id, to_tw_id)
    return ub_graph


def add_sem_units_into_usr_belief_graph(ub_graph, from_tw_id, to_tw_id):
    if path.exists(g_sem_units_folder + from_tw_id + '_cls_graph.gml') \
            and path.exists(g_sem_units_folder + to_tw_id + '_cls_graph.gml'):
        from_cls_graph = nx.read_gml(g_sem_units_folder + from_tw_id + '_cls_graph.gml')
        to_cls_graph = nx.read_gml(g_sem_units_folder + to_tw_id + '_cls_graph.gml')
        ub_graph = add_cls_graphs_into_usr_belief_graph(ub_graph, from_cls_graph, to_cls_graph)
    if path.exists(g_sem_units_folder + from_tw_id + '_nps.txt') \
            and path.exists(g_sem_units_folder + to_tw_id + '_nps.txt'):
        l_from_nps = []
        l_to_nps = []
        with open(g_sem_units_folder + from_tw_id + '_nps.txt', 'r') as in_fd:
            for ln in in_fd:
                l_from_nps.append(ln.strip())
            in_fd.close()
        with open(g_sem_units_folder + to_tw_id + '_nps.txt', 'r') as in_fd:
            for ln in in_fd:
                l_to_nps.append(ln.strip())
            in_fd.close()
        ub_graph = add_nps_into_usr_belief_graph(ub_graph, l_from_nps, l_to_nps)
    return ub_graph


def add_cls_graphs_into_usr_belief_graph(ub_graph, from_cls_graph, to_cls_graph):
    s_sel_pos_pairs = {'NOUN_NOUN', 'NOUN_VERB', 'VERB_NOUN', 'VERB_VERB'}
    l_ub_from_nodes = []
    for edge in from_cls_graph.edges():
        node_1 = edge[0]
        node_2 = edge[1]
        node_1_pos = from_cls_graph.nodes(data=True)[node_1]['pos']
        node_2_pos = from_cls_graph.nodes(data=True)[node_2]['pos']
        if node_1_pos + '_' + node_2_pos in s_sel_pos_pairs:
            ub_from_node, ub_graph = classify_input_phrase(ub_graph, ' '.join([node_1, node_2]))
            l_ub_from_nodes.append(ub_from_node)
    l_ub_to_nodes = []
    for edge in to_cls_graph.edges():
        node_1 = edge[0]
        node_2 = edge[1]
        node_1_pos = from_cls_graph.nodes(data=True)[node_1]['pos']
        node_2_pos = from_cls_graph.nodes(data=True)[node_2]['pos']
        if node_1_pos + '_' + node_2_pos in s_sel_pos_pairs:
            ub_to_node, ub_graph = classify_input_phrase(ub_graph, ' '.join([node_1, node_2]))
            l_ub_to_nodes.append(ub_to_node)
    if len(l_ub_from_nodes) == 0 or len(l_ub_to_nodes) == 0:
        return ub_graph
    for from_node in l_ub_from_nodes:
        for to_node in l_ub_to_nodes:
            if ub_graph.has_edge(from_node, to_node):
                ub_graph.edges[(from_node, to_node)]['sal'] += 1
            else:
                ub_graph.add_edge(from_node, to_node, sal=1)
    return ub_graph


def add_nps_into_usr_belief_graph(ub_graph, from_nps, to_nps):
    if len(from_nps) == 0 or len(to_nps) == 0:
        return ub_graph
    l_ub_from_nodes = []
    l_ub_to_nodes = []
    for noun_phrase in from_nps:
        ub_from_node, ub_graph = classify_input_phrase(ub_graph, noun_phrase)
        l_ub_from_nodes.append(ub_from_node)
    for noun_phrase in to_nps:
        ub_to_node, ub_graph = classify_input_phrase(ub_graph, noun_phrase)
        l_ub_from_nodes.append(ub_to_node)
    if len(l_ub_from_nodes) == 0 or len(l_ub_to_nodes) == 0:
        return ub_graph
    for from_node in l_ub_from_nodes:
        for to_node in l_ub_to_nodes:
            if ub_graph.has_edge(from_node, to_node):
                ub_graph.edges[(from_node, to_node)]['sal'] += 1
            else:
                ub_graph.add_edge(from_node, to_node, sal=1)
    return ub_graph



def classify_input_phrase(ub_graph, input_phrase):
    input_node_vect = phrase_embedding(input_phrase)
    best_matched_node = None
    best_matched_sim = 0.0
    best_matched_vect = None
    for node in ub_graph.nodes(data=True):
        node_vect = node[1]['avg_vect']
        sim = 1.0 - scipyd.cosine(input_node_vect, node_vect)
        if sim >= g_node_sim_threshold and sim > best_matched_sim:
            best_matched_node = node
            best_matched_sim = 0.0
            best_matched_vect = node_vect
    if best_matched_node is not None:
        best_matched_node[1]['phrases'] += '\n'
        best_matched_node[1]['phrases'] += input_phrase
        best_matched_node[1]['avg_vect'] = (best_matched_vect + input_node_vect) / 2
        return best_matched_node[0], ub_graph
    else:
        ub_graph.add_node(input_phrase, avg_vect=input_node_vect, phrases=input_phrase)
        return input_phrase, ub_graph


def load_lexvec_model():
    global g_lexvec_model
    model_file = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
    g_lexvec_model = lexvec.Model(model_file)


def phrase_embedding(phrase_str):
    if g_word_embedding_model == 'lexvec':
        if g_lexvec_model is None:
            load_lexvec_model()
        phrase_vec = np.zeros(300)
        l_words = [word.strip().lower() for word in phrase_str.split(' ')]
        for word in l_words:
            word_vec = g_lexvec_model.word_rep(word)
            phrase_vec += word_vec
        return phrase_vec



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # renama_sem_unit_files()
    l_usrs = get_usr_list()
    for usr_id in l_usrs:
        ub_graph = build_usr_belief_graph(usr_id)
        print()