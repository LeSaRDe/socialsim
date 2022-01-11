import logging
import os
from os import path, walk
import json
import sys
sys.path.insert(1, '/home/mf3jh/workspace/lib/lexvec/lexvec/python/lexvec/')
import model as lexvec
import numpy as np
import scipy.spatial.distance as scipyd
import time
import networkx as nx


g_dsctp_int_rets_folder = '/home/mf3jh/workspace/data/docsim/dsctp_int_rets/dsctp_int_rets/'
g_pos_map = {'CC': 'CC', 'CD': 'CD', 'DT': 'D', 'EX': 'E', 'FW': 'F', 'IN': 'I', 'JJ': 'J', 'JJR': 'J', 'JJS': 'J',
             'LS': 'L', 'MD': 'M', 'NN': 'N', 'NNS': 'N', 'NNP': 'N', 'NNPS': 'N', 'PDT': 'PD', 'POS': 'PO', 'PRP': 'P',
             'PRP$': 'P', 'RB': 'R', 'RBR': 'R', 'RBS': 'R', 'RP': 'RP', 'SYM': 'S', 'TO': 'T', 'UH': 'U', 'VB': 'V',
             'VBD': 'V', 'VBG': 'V', 'VBN': 'V', 'VBP': 'V', 'VBZ': 'V', 'WDT': 'W', 'WP': 'P', 'WP$': 'P', 'WRB': 'R'}
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


def rename_sem_unit_files():
    l_dataset_names = ['20news50short10', 'reuters', 'bbc']
    sem_unit_folder_format = '/home/mf3jh/workspace/data/docsim/sem_units_{0}'
    for dataset_name in l_dataset_names:
        for (dirpath, dirname, filenames) in walk(sem_unit_folder_format.format(dataset_name)):
            for filename in filenames:
                if filename[-14:] == '_cls_graph.gml':
                    doc_id = filename.split('|')[0].strip()
                    os.rename(dirpath + '/' + filename, dirpath + '/' + doc_id + '_cls_graph.gml')
                if filename[-8:] == '_nps.txt':
                    doc_id = filename.split('|')[0].strip()
                    os.rename(dirpath + '/' + filename, dirpath + '/' + doc_id + '_nps.txt')


def match_dsctp_gp_sem_units():
    stats_str = ''
    l_dataset_names = ['20news50short10', 'reuters', 'bbc']
    sem_unit_folder_format = '/home/mf3jh/workspace/data/docsim/sem_units_{0}/'
    dsctp_int_folder_format = '/home/mf3jh/workspace/data/docsim/dsctp_int_rets/dsctp_int_rets/{0}_nasari_30_rmswcbwexpws_w3-3/'
    for datasetname in l_dataset_names:
        for (dirpath, dirname, filenames) in walk(dsctp_int_folder_format.format(datasetname)):
            for filename in filenames:
                if filename[-5:] == '.json':
                    doc_ids = filename[:-5].split('#')
                    doc_id_1 = doc_ids[0].strip()
                    doc_id_2 = doc_ids[1].strip()
                    with open(dirpath + '/' + filename, 'r') as in_fd:
                        int_ret_json = json.load(in_fd)
                        in_fd.close()
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
                            cls_graph_1 = nx.read_gml()

                            for edge in cls_graph_1.edges():
                                node_1 = edge[0]
                                node_2 = edge[1]
                                l_node_1_tokens = [token.strip().lower() for token in node_1.split('|')]
                                l_node_2_tokens = [token.strip().lower() for token in node_2.split('|')]





if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    rename_sem_unit_files()

    # timer_start = time.time()
    # d_pos_pair_stats = dict()
    # cycle_cnt = 0
    # pos_match_cnt = 0
    # l_sims = []
    # for (dirpath, dirname, filenames) in walk(g_dsctp_int_rets_folder):
    #     for filename in filenames:
    #         if filename[-5:] == '.json':
    #             with open(dirpath + '/' + filename, 'r') as in_fd:
    #                 int_ret_json = json.load(in_fd)
    #                 in_fd.close()
    #             for sent_pair_key in int_ret_json['sentence_pair']:
    #                 for cycle in int_ret_json['sentence_pair'][sent_pair_key]['cycles']:
    #                     ##################################################
    #                     #   GP SIMS STATS STARTS
    #                     ##################################################
    #                     l_s1_words = []
    #                     l_s2_words = []
    #                     for token in cycle:
    #                         if token[3] == 'L':
    #                             token_lemma = token.split('#')[5].split(':')[0].strip()
    #                             if token[:2] == 's1':
    #                                 l_s1_words.append(token_lemma)
    #                             elif token[:2] == 's2':
    #                                 l_s2_words.append(token_lemma)
    #                     s1_embed = phrase_embedding(' '.join(l_s1_words))
    #                     s2_embed = phrase_embedding(' '.join(l_s2_words))
    #                     sim = 1.0 - scipyd.cosine(s1_embed, s2_embed)
    #                     l_sims.append(sim)
    #                     cycle_cnt += 1
    #                     if cycle_cnt % 10000 == 0 and cycle_cnt >= 10000:
    #                         logging.debug('%s have done in % secs.' % (cycle_cnt, time.time() - timer_start))
    #                     ##################################################
    #                     #   GP SIMS STATS ENDS
    #                     ##################################################
    #
    #                     ##################################################
    #                     #   POS TAG PAIRS STATS STARTS
    #                     ##################################################
    #                     # l_s1_pos = []
    #                     # l_s2_pos = []
    #                     # for token in cycle:
    #                     #     if token[3] == 'L':
    #                     #         token_pos = token.split('#')[4]
    #                     #         if token_pos in g_pos_map:
    #                     #             token_pos = g_pos_map[token_pos]
    #                     #         if token[:2] == 's1':
    #                     #             l_s1_pos.append(token_pos)
    #                     #         elif token[:2] == 's2':
    #                     #             l_s2_pos.append(token_pos)
    #                     #
    #                     # if len(l_s1_pos) > 1:
    #                     #     s1_pos_pair = l_s1_pos[0] + '#' + l_s1_pos[1]
    #                     #     s1_pos_pair_rev = l_s1_pos[1] + '#' + l_s1_pos[0]
    #                     # else:
    #                     #     s1_pos_pair = l_s1_pos[0]
    #                     #     s1_pos_pair_rev = s1_pos_pair
    #                     #
    #                     # if s1_pos_pair not in d_pos_pair_stats and s1_pos_pair_rev not in d_pos_pair_stats:
    #                     #     d_pos_pair_stats[s1_pos_pair] = 1
    #                     # elif s1_pos_pair in d_pos_pair_stats:
    #                     #     d_pos_pair_stats[s1_pos_pair] += 1
    #                     # elif s1_pos_pair_rev in d_pos_pair_stats:
    #                     #     d_pos_pair_stats[s1_pos_pair_rev] += 1
    #                     #
    #                     # if len(l_s2_pos) > 1:
    #                     #     s2_pos_pair = l_s2_pos[0] + '#' + l_s2_pos[1]
    #                     #     s2_pos_pair_rev = l_s2_pos[1] + '#' + l_s2_pos[0]
    #                     # else:
    #                     #     s2_pos_pair = l_s2_pos[0]
    #                     #     s2_pos_pair_rev = s2_pos_pair
    #                     #
    #                     # if s2_pos_pair not in d_pos_pair_stats and s2_pos_pair_rev not in d_pos_pair_stats:
    #                     #     d_pos_pair_stats[s2_pos_pair] = 1
    #                     # elif s2_pos_pair in d_pos_pair_stats:
    #                     #     d_pos_pair_stats[s2_pos_pair] += 1
    #                     # elif s2_pos_pair_rev in d_pos_pair_stats:
    #                     #     d_pos_pair_stats[s2_pos_pair_rev] += 1
    #                     #
    #                     # if (s1_pos_pair == s2_pos_pair) or (s1_pos_pair_rev == s2_pos_pair_rev) \
    #                     #         or (s1_pos_pair == s2_pos_pair_rev) or (s1_pos_pair_rev == s2_pos_pair):
    #                     #     pos_match_cnt += 1
    #                     # cycle_cnt += 1
    #                     ##################################################
    #                     #   POS TAG PAIRS STATS ENDS
    #                     ##################################################
    #
    # ##################################################
    # #   GP SIMS STATS STARTS
    # ##################################################
    # logging.debug('%s have done in % secs.' % (cycle_cnt, time.time() - timer_start))
    # with open(g_dsctp_int_rets_folder + 'dsctp_gp_sim_stats.txt', 'w+') as out_fd:
    #     out_str = '\n'.join([str(sim) for sim in l_sims])
    #     out_fd.write(out_str)
    #     out_fd.close()
    # ##################################################
    # #   GP SIMS STATS ENDS
    # ##################################################
    #
    # ##################################################
    # #   POS TAG PAIRS STATS STARTS
    # ##################################################
    # # logging.debug('cycle_cnt = %s, pos_match_cnt = %s.' % (cycle_cnt, pos_match_cnt))
    # #
    # # with open(g_dsctp_int_rets_folder + 'dsctp_pos_pair_stats.txt', 'w+') as out_fd:
    # #     for pos_pair in d_pos_pair_stats:
    # #         out_str = pos_pair + '|' + str(d_pos_pair_stats[pos_pair])
    # #         out_fd.write(out_str)
    # #         out_fd.write('\n')
    # #     out_fd.close()
    # ##################################################
    # #   POS TAG PAIRS STATS ENDS
    # ##################################################