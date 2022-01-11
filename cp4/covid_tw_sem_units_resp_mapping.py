import logging
import networkx as nx
import time
from os import walk, path
import os
import sys
sys.path.insert(1, '/home/mf3jh/workspace/lib/lexvec/lexvec/python/lexvec/')
import model as lexvec
import numpy as np
# import tensorflow as tf
import scipy.spatial.distance as scipyd


'''
NOTE:
tensorflow's cosine similarity is much slower than scipyd!!!
'''


# 82571 tw_id in covid_tw_resp_pairs.txt, but only 81776 are sem units.
g_covid_resp_pairs_file = 'covid_tw_resp_pairs.txt'
g_covid_resp_en_sem_unit_folder = '/home/mf3jh/workspace/cp4_narratives/covid19/en_sem_units/'
g_covid_resp_map_graph_file = 'covid_tw_resp_map_graph.gml'

g_sal_pos_pairs = {'VERB_NOUN', 'NOUN_VERB', 'NOUN_NOUN', 'VERB_VERB'}
g_word_embedding_model = 'lexvec'
g_lexvec_model = None
g_sim_threshold = 0.6


def change_sem_units_file_names():
    for (dirpath, dirname, filenames) in walk(g_covid_resp_en_sem_unit_folder):
        for filename in filenames:
            if filename[-14:] != '_cls_graph.gml' and filename[-8:] != '_nps.txt':
                continue
            filename_fields = filename.split('|')
            if filename[-14:] == '_cls_graph.gml':
                new_filename = filename_fields[0] + '_cls_graph.gml'
            if filename[-8:] == '_nps.txt':
                new_filename = filename_fields[0] + '_nps.txt'
            os.rename(dirpath + '/' + filename, dirpath + '/' + new_filename)


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


def extract_sal_sem_units_from_cls_graph_and_nps(cls_graph_path, nps_path):
    sem_units = []
    cls_graph = None
    if path.exists(cls_graph_path):
        resp_cls_graph = nx.read_gml(cls_graph_path)
    phrases = []
    if cls_graph is not None:
        for edge in cls_graph.edges(data=True):
            node_1 = edge[0]
            node_2 = edge[1]
            node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
            node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
            if node_1_pos + '_' + node_2_pos in g_sal_pos_pairs:
                node_1_txt = cls_graph.nodes(data=True)[node_1]['txt'].strip()
                node_2_txt = cls_graph.nodes(data=True)[node_2]['txt'].strip()
                phrases.append(node_1_txt + ' ' + node_2_txt)
    nps = []
    if path.exists(nps_path):
        with open(nps_path, 'r') as in_fd:
            for ln in in_fd:
                nps.append(ln.strip())
            in_fd.close()
    sem_units = phrases + nps
    return sem_units


def classify_new_node(new_node_txt, trg_graph):
    new_node_txt = new_node_txt.strip()
    new_node_vect = phrase_embedding(new_node_txt)
    new_node = new_node_txt
    is_classified = False
    for node in trg_graph.nodes(data=True):
        node_vect = node[1]['vect']
        sim = 1.0 - scipyd.cosine(node_vect, new_node_vect)
        if sim >= g_sim_threshold:
            new_node = node[0]
            l_fulltxt = [phrase.strip().lower() for phrase in node[1]['fulltxt'].split('\n')]
            if new_node_txt.lower() not in l_fulltxt:
                node[1]['fulltxt'] += '\n'
                node[1]['fulltxt'] += new_node_txt
            is_classified = True
            break
    if not is_classified:
        trg_graph.add_node(new_node, fulltxt=new_node_txt, vect=new_node_vect)
    return new_node


def build_resp_mapping():
    resp_map_graph = nx.DiGraph()
    rec_cnt = 0
    timer_start = time.time()
    with open(g_covid_resp_pairs_file, 'r') as in_fd:
        for ln in in_fd:
            l_tw_ids = [tw_id.strip() for tw_id in ln.split('|')]
            resp_tw_id = l_tw_ids[0]
            src_tw_id = l_tw_ids[1]
            if (not path.exists(g_covid_resp_en_sem_unit_folder + resp_tw_id + '_cls_graph.gml')
                and path.exists(g_covid_resp_en_sem_unit_folder + resp_tw_id + '_nps.txt')) \
                or \
                (not path.exists(g_covid_resp_en_sem_unit_folder + src_tw_id + '_cls_graph.gml')
                 and path.exists(g_covid_resp_en_sem_unit_folder + src_tw_id + '_nps.txt')):
                rec_cnt += 1
                continue

            resp_sem_units = extract_sal_sem_units_from_cls_graph_and_nps(g_covid_resp_en_sem_unit_folder + resp_tw_id + '_cls_graph.gml',
                                                                          g_covid_resp_en_sem_unit_folder + resp_tw_id + '_nps.txt')
            src_sem_units = extract_sal_sem_units_from_cls_graph_and_nps(g_covid_resp_en_sem_unit_folder + src_tw_id + '_cls_graph.gml',
                                                                         g_covid_resp_en_sem_unit_folder + src_tw_id + '_nps.txt')
            for resp_su_idx, resp_su in enumerate(resp_sem_units):
                resp_node = classify_new_node(resp_su, resp_map_graph)
                for src_su_idx, src_su in enumerate(src_sem_units):
                    src_node = classify_new_node(src_su, resp_map_graph)
                    if resp_map_graph.has_edge(src_node, resp_node):
                        resp_map_graph.edges()[(src_node, resp_node)]['cnt'] += 1
                    else:
                        resp_map_graph.add_edge(src_node, resp_node, cnt=1)
            rec_cnt += 1
            if rec_cnt % 1000 == 0 and rec_cnt >= 1000:
                logging.debug('%s resp pairs scanned in %s secs.' % (rec_cnt, time.time() - timer_start))
        in_fd.close()
    logging.debug('%s resp pairs scanned in %s secs.' % (rec_cnt, time.time() - timer_start))
    logging.debug('resp_map_graph is done.')

    for node in resp_map_graph.nodes(data=True):
        node[1]['vect'] = ','.join([str(ele) for ele in node[1]['vect']])

    nx.write_gml(resp_map_graph, g_covid_resp_map_graph_file)
    logging.debug('All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # change_sem_units_file_names()
    load_lexvec_model()
    build_resp_mapping()


