import logging
from datetime import datetime, timedelta
from os import walk
import networkx as nx
import json
import codecs
import numpy as np
import sys
sys.path.insert(1, '/home/mf3jh/workspace/lib/lexvec/lexvec/python/lexvec/')
import model as lexvec
from gensim.models import KeyedVectors
import os
import scipy.spatial.distance as scipyd
import matplotlib.pyplot as plt


g_ven_tw_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
g_ven_tw_cas_folder = g_ven_tw_folder + 'cascades/'
g_ven_tw_size_span_stats = g_ven_tw_folder + 'ven_tw_size_span_stats.json'
g_ven_tw_tint = g_ven_tw_folder + 'ven_tw_tint.json'
g_ven_tw_cas_tint_dist = g_ven_tw_folder + 'ven_tw_cas_tint_dist.json'

g_start_datetime_str = '20151018063220'
g_end_datetime_str = '20190131235959'
g_datetime_format = '%Y%m%d%H%M%S'
g_delta_datetime = timedelta(days=7)
g_holdback_datetime = timedelta(seconds=1)
g_cas_size_threshold = 10
g_selected_cas_ids = ['ki4OfrPWfLm3mVoykrvgZA', 'NAUniYUFrv9ohzs8IeFcVg', 'eyzPCXEkTdh-dSUGkme0Ug',
                      'WcVqnWcvB4a5u45m3wr1Rg', '6RlZEobUb5uwyaSq47byqQ', 'YBdNG1rWfnZfuYEGWbu6Dg']
g_selected_cas_tint_dist = g_ven_tw_folder + 'ven_tw_sel_cas_tint_dist.json'
g_glove_model = None
g_lexvec_model = None
g_lexvec_sim_threshold = 0.6
g_sel_tint_graph_format = g_ven_tw_folder + '6_samples/' + 'ven_sel_tint_graph_{0}.gml'
g_sel_tint_graph_img_format = g_ven_tw_folder + '6_samples/' + 'ven_sel_tint_graph_{0}.png'


def convert_to_binary(embedding_path):
    """
    Here, it takes path to embedding text file provided by glove.
    :param embedding_path: takes path of the embedding which is in text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot less time to load.
    """
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            splitlines = line.split()
            if len(splitlines) != 301:
                print(splitlines[:-300])
            vocab_write.write(' '.join(splitlines[:-300]).strip())
            vocab_write.write("\n")
            wv.append([float(val) for val in splitlines[-300:]])
            count += 1
    np.save(embedding_path + ".npy", np.array(wv))


def load_embeddings_binary(embeddings_path):
    global g_glove_model
    """
    It loads embedding provided by glove which is saved as binary file. Loading of this model is
    about  second faster than that of loading of txt glove file as model.
    :param embeddings_path: path of glove file.
    :return: glove model
    """
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(embeddings_path + '.npy', allow_pickle=True)
    g_glove_model = {}
    for i, w in enumerate(index2word):
        g_glove_model[w] = wv[i]
    return g_glove_model


def glove_w2v(word):
    """
    :param sentence: inputs a single sentences whose word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding of all words    in input sentence.
    """
    return g_glove_model.get(word, np.zeros(100))
    # return np.array([model.get(val, np.zeros(100)) for val in sentence.split()], dtype=np.float64)


def get_lexvec_vec(phrase):
    # word_vec = g_lexvec_model.wv[word]
    phrase_vec = np.zeros(300)
    l_words = [word.strip() for word in phrase.split(' ')]
    for word in l_words:
        word_vec = g_lexvec_model.word_rep(word)
        phrase_vec += word_vec
    return phrase_vec


def load_lexvec_model():
    global g_lexvec_model
    # model_file = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.300d.W+C.pos.vectors'
    # g_lexvec_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    model_file = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
    g_lexvec_model = lexvec.Model(model_file)


def fit_node_into_tint_graph(tint_graph, new_node, tint):
    new_node_vec = get_lexvec_vec(new_node)
    for node in tint_graph.nodes(data=True):
        node_str = node[0]
        l_node_words = [word.strip() for word in node[1]['txt'].split('\n')]
        if new_node in l_node_words:
            if tint in node[1]['sal']:
                node[1]['sal'][tint] += 1
            else:
                node[1]['sal'][tint] = 1
            return True, node_str
        else:
            node_vec = node[1]['vec']
            sim = 1.0 - scipyd.cosine(node_vec, new_node_vec)
            if sim >= g_lexvec_sim_threshold:
                node[1]['txt'] += '\n'
                node[1]['txt'] += new_node
                node[1]['vec'] = (node_vec + new_node_vec) / 2
                if tint in node[1]['sal']:
                    node[1]['sal'][tint] += 1
                else:
                    node[1]['sal'][tint] = 1
                return True, node_str
    tint_graph.add_node(new_node, txt=new_node, vec=new_node_vec, sal={tint: 1})
    return False, new_node


def fit_edge_into_tint_graph(tint_graph, new_edge, tint):
    node_1 = new_edge[0]
    node_2 = new_edge[1]
    if node_1 == node_2:
        return
    if tint_graph.has_edge(node_1, node_2):
        if tint in tint_graph.edges[node_1, node_2]['sal']:
            tint_graph.edges[node_1, node_2]['sal'][tint] += 1
        else:
            tint_graph.edges[node_1, node_2]['sal'][tint] = 1
    else:
        tint_graph.add_edge(node_1, node_2, sal={tint: 1})


def build_one_tint_sem_graph(tint_graph, l_tws, tint):
    # tint_graph = nx.Graph()
    for tw_id in l_tws:
        for (dirpath, dirname, filenames) in walk(g_ven_tw_cas_folder):
            for filename in filenames:
                if filename[-14:] != '_cls_graph.gml' or filename[:-14] != tw_id:
                    continue
                tw_cls_graph = nx.read_gml(dirpath + '/' + filename)
                for edge in tw_cls_graph.edges():
                    node_1_label = edge[0]
                    node_1_txt = node_1_label.split('|')[2]
                    is_fit_1, fit_node_1 = fit_node_into_tint_graph(tint_graph, node_1_txt, tint)
                    node_2_label = edge[1]
                    node_2_txt = node_2_label.split('|')[2]
                    is_fit_2, fit_node_2 = fit_node_into_tint_graph(tint_graph, node_2_txt, tint)
                    new_edge = (fit_node_1, fit_node_2)
                    fit_edge_into_tint_graph(tint_graph, new_edge, tint)
    # nx.write_gml(tint_graph, g_sel_tint_graph_format.format(tint))
    return tint_graph


def build_tint_sem_graphs():
    l_tints, d_tint_tws = get_selected_tint_tws()
    tint_graph = nx.Graph()
    for tint in l_tints:
        l_tws = d_tint_tws[tint]
        tint_graph = build_one_tint_sem_graph(tint_graph, l_tws, tint)
    for node in tint_graph.nodes(data=True):
        node[1]['vec'] = np.array2string(node[1]['vec'])
    nx.write_gml(tint_graph, g_sel_tint_graph_format.format('full'))
    return tint_graph


def draw_tint_graph():
    l_tints, d_tint_tws = get_selected_tint_tws()
    tint_graph = nx.read_gml(g_sel_tint_graph_format.format('full'))
    max_node_sal = 0
    for node in tint_graph.nodes(data=True):
        cur_max_sal = max(list(node[1]['sal'].values()))
        if cur_max_sal > max_node_sal:
            max_node_sal = cur_max_sal
    max_node_sal = float(max_node_sal)
    max_edge_sal = 0
    for edge in tint_graph.edges:
        cur_max_sal = max(list(tint_graph.edges[edge[0], edge[1]]['sal'].values()))
        if cur_max_sal > max_edge_sal:
            max_edge_sal = cur_max_sal
    max_edge_sal = float(max_edge_sal)

    plt.figure(1, figsize=(50, 50), tight_layout={'pad': 1, 'w_pad': 100, 'h_pad': 100, 'rect': None})
    pos = nx.spring_layout(tint_graph, k=0.8)
    # pos = nx.kamada_kawai_layout(tint_graph)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)

    for tint in l_tints:
        l_emp_nodes_data = sorted([node for node in tint_graph.nodes(data=True) if tint in node[1]['sal']],
                                  key=lambda k: k[1]['sal'][tint])
        l_emp_nodes = [node[0] for node in l_emp_nodes_data]
        d_emp_node_labels = {node[0]: node[1]['txt'] for node in l_emp_nodes_data}
        l_emp_node_colors = [float(node[1]['sal'][tint]+10.0) for node in l_emp_nodes_data]
        l_other_nodes = [node[0] for node in tint_graph.nodes(data=True) if tint not in node[1]['sal']]

        l_emp_edges = [(edge[0], edge[1]) for edge in tint_graph.edges(data=True)
                       if tint in tint_graph.edges[edge[0], edge[1]]['sal']]
        l_emp_edge_colors = [float(tint_graph.edges[edge[0], edge[1]]['sal'][tint]+2.0) for edge in l_emp_edges]
        l_other_edges = [(edge[0], edge[1]) for edge in tint_graph.edges(data=True)
                         if tint not in tint_graph.edges[edge[0], edge[1]]['sal']]

        nx.draw_networkx_nodes(tint_graph, pos, nodelist=l_other_nodes, node_color='#D5D8DC', node_size=200)
        nx.draw_networkx_edges(tint_graph, pos, edgelist=l_other_edges, width=2, edge_color='#D5D8DC')

        nx.draw_networkx_edges(tint_graph, pos, edgelist=l_emp_edges, width=6, edge_color=l_emp_edge_colors,
                               edge_cmap=plt.cm.get_cmap('Reds'), edge_vmin=0.0, edge_vmax=max_edge_sal)
        nx.draw_networkx_nodes(tint_graph, pos, nodelist=l_emp_nodes, node_color=l_emp_node_colors,
                               cmap=plt.cm.get_cmap('Blues'), vmin=0.0, vmax=max_node_sal, node_size=600)
        nx.draw_networkx_labels(tint_graph, pos, labels=d_emp_node_labels, font_color='k', font_size=50,
                                font_family="sans-serif")

        plt.savefig(g_sel_tint_graph_img_format.format(tint), format="PNG")
        plt.clf()


def get_selected_tint_tws():
    with open(g_selected_cas_tint_dist, 'r') as in_fd:
        d_sel_cas = json.load(in_fd)
        in_fd.close()
    d_tint_tws = dict()
    for cas_id in d_sel_cas:
        for tint in d_sel_cas[cas_id]:
            if tint in d_tint_tws:
                tint_mod = ''.join(tint.split('_'))
                d_tint_tws[tint_mod] += d_sel_cas[cas_id][tint]
            else:
                tint_mod = ''.join(tint.split('_'))
                d_tint_tws[tint_mod] = d_sel_cas[cas_id][tint]
    l_tints = sorted(list(d_tint_tws.keys()), key=lambda k: int(k[4:]))
    return l_tints, d_tint_tws



def retrieve_selected_cas_tws():
    d_sel_cas = dict()
    with open(g_ven_tw_size_span_stats, 'r') as in_fd:
        d_stat = json.load(in_fd)
        in_fd.close()
    for cas_id in g_selected_cas_ids:
        cas_data = d_stat[cas_id]
        del cas_data['size']
        d_sel_cas[cas_id] = cas_data
    with open(g_selected_cas_tint_dist, 'w+') as out_fd:
        json.dump(d_sel_cas, out_fd)
        out_fd.close()


def gen_time_intervals():
    l_time_ints = []
    abs_start_datetime = datetime.strptime(g_start_datetime_str, g_datetime_format)
    abs_end_datetime = datetime.strptime(g_end_datetime_str, g_datetime_format)
    cur_start_datetime = abs_start_datetime
    cur_end_datetime = abs_start_datetime
    while cur_end_datetime < abs_end_datetime:
        cur_end_datetime += g_delta_datetime
        cur_start_datetime_str = cur_start_datetime.strftime(g_datetime_format)
        cur_end_datetime_str = (cur_end_datetime - g_holdback_datetime).strftime(g_datetime_format)
        l_time_ints.append((cur_start_datetime_str, cur_end_datetime_str))
        cur_start_datetime = cur_end_datetime
    d_tint = dict()
    for idx, tint in enumerate(l_time_ints):
        d_tint['tint_' + str(idx)] = [tint[0], tint[1]]
    with open(g_ven_tw_tint, 'w+') as out_fd:
        json.dump(d_tint, out_fd)
        out_fd.close()


def retrieve_time_intervals():
    with open(g_ven_tw_tint, 'r') as in_fd:
        d_tint = json.load(in_fd)
        in_fd.close()
    return d_tint


def get_tw_time_interval(tw_datetime_str, d_tint):
    for tint_item in d_tint.items():
        start_datetime_str = tint_item[1][0]
        end_datetime_str = tint_item[1][1]
        if start_datetime_str <= tw_datetime_str <= end_datetime_str:
            return tint_item[0]
    raise Exception('%s is not in any time interval.' % tw_datetime_str)


def cas_size_span_stat():
    d_cas_stat = dict()
    d_tint = retrieve_time_intervals()
    for (dirpath, dirname, filenames) in walk(g_ven_tw_cas_folder):
        for filename in filenames:
            if filename[-4:] != '.gml' or filename[-14:] == '_cls_graph.gml':
                continue
            cas_id = filename[:-4]
            cas_graph = nx.read_gml(g_ven_tw_cas_folder + filename)
            if nx.number_of_nodes(cas_graph) < g_cas_size_threshold:
                continue
            d_cas_stat[cas_id] = dict()
            d_cas_stat[cas_id]['size'] = nx.number_of_nodes(cas_graph)
            for tw_data in cas_graph.nodes(data=True):
                tw_id = tw_data[0]
                tw_datetime_str = tw_data[1]['datetime']
                tw_t_int = get_tw_time_interval(tw_datetime_str, d_tint)
                if tw_t_int in d_cas_stat[cas_id]:
                    d_cas_stat[cas_id][tw_t_int].append(tw_id)
                else:
                    d_cas_stat[cas_id][tw_t_int] = [tw_id]
    with open(g_ven_tw_size_span_stats, 'w+') as out_fd:
        json.dump(d_cas_stat, out_fd)
        out_fd.close()


def cas_dist_over_tints():
    total_cas_cnt = 0
    total_tw_cnt = 0
    d_tint_dist = dict()
    with open(g_ven_tw_size_span_stats, 'r') as in_fd:
        d_stat = json.load(in_fd)
        in_fd.close()
    total_cas_cnt = len(d_stat)
    for cas_item in d_stat.items():
        d_cas_data = cas_item[1]
        for tint in d_cas_data.keys():
            if tint == 'size':
                continue
            if tint in d_tint_dist:
                d_tint_dist[tint] += d_cas_data[tint]
            else:
                d_tint_dist[tint] = d_cas_data[tint]
            total_tw_cnt += len(d_cas_data[tint])
    with open(g_ven_tw_cas_tint_dist, 'w+') as out_fd:
        json.dump(d_tint_dist, out_fd)
        out_fd.close()
    logging.debug('%s cas, %s tws.' % (total_cas_cnt, total_tw_cnt))


def cas_span_stat():
    l_t_ints = []
    for (dirpath, dirname, filenames) in walk(g_ven_tw_cas_folder):
        for filename in filenames:
            if filename[-4:] != '.gml' or filename[-14:] == '_cls_graph.gml':
                continue
            cas_graph = nx.read_gml(g_ven_tw_cas_folder + filename)
            for tw_data in cas_graph.nodes(data=True):
                tw_datetime_str = tw_data[1]['datetime']
                l_t_ints.append(tw_datetime_str)
    with open('ven_tw_tint_stat.txt', 'w+') as out_fd:
        out_fd.write('\n'.join(l_t_ints))
        out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # l_time_ints = get_time_intervals()
    # print(l_time_ints)
    # l_tints, d_tint_tws = get_selected_tint_tws()
    # load_lexvec_model()
    # build_tint_sem_graphs()
    # draw_tint_graph()
    l_tints, d_tint_tw = get_selected_tint_tws()
    print()
    # l_tints = [item[:4] + '_' + item[4:] for item in l_tints]
    # with open(g_ven_tw_tint, 'r') as in_fd:
    #     d_tints = json.load(in_fd)
    #     in_fd.close()
    # with open('6samples_tint.txt', 'w+') as out_fd:
    #     for tint in l_tints:
    #         ln = tint + ':' + d_tints[tint][0] + ' - ' + d_tints[tint][1]
    #         out_fd.write(ln)
    #         out_fd.write('\n')
    #     out_fd.close()