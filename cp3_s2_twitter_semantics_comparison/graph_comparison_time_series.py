import json
import networkx as nx
from sklearn import metrics
import sys

sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils
import logging
import math
import threading
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import igraph as ig


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_fsgraph_gml_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph.gml'
g_tsgraph_gml_format = g_time_series_data_path_prefix + '{0}/{1}_tsgraph.gml'
g_subtsgraph_gml_format = g_time_series_data_path_prefix + '{0}/{1}_subtsgraph.gml'
g_fsgraph_inter_path = g_time_series_data_path_prefix + 'fsgraph_simple_comp_inter_rets/'
g_tsgraph_inter_path = g_time_series_data_path_prefix + 'tsgraph_sample_comp_inter_rets/'
g_fsgraph_inter_format = g_fsgraph_inter_path + '{0}.json'
g_tsgraph_inter_format = g_tsgraph_inter_path + '{0}.json'
g_time_ints_idx_map_path = g_time_series_data_path_prefix + 'time_ints_idx_map.txt'
# g_fsgraph_community_cluster_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph_com_cluster.json'
g_fsgraph_community_cluster_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph_fine_com_cluster_50.json'
# g_fsgraph_com_inter_path = g_time_series_data_path_prefix + 'fsgraph_com_comp_inter_rets/'
g_fsgraph_com_inter_path = g_time_series_data_path_prefix + 'fsgraph_fine_com_comp_inter_rets/'
# g_fsgraph_com_inter_path = g_time_series_data_path_prefix + 'fsgraph_simple_comp_inter_rets/'
g_fsgraph_com_inter_format = g_fsgraph_com_inter_path + '{0}.json'
g_topic_embed_comp_inter_rets_path = g_time_series_data_path_prefix + 'topic_embed_comp_inter_rets/'
g_influential_users_path = g_time_series_data_path_prefix + 'influential_users.json'
g_normalize_weights = False
g_spearman = False
g_sig_com_size_threshold = 200
g_tsgraph_sim_threshold = 0.85
g_bad_tsgraphs = ['20180718_20180724', '20181212_20181218', '20190114_20190120', '20190408_20190414',
                  '20190207_20190213', '20181115_20181121', '20181209_20181215', '20181127_20181203',
                  '20181130_20181206']
g_tsgraph_start_time = '20180829_20180904'
g_tsgraph_end_time = '20181031_20181106'


def rig_spearman_for_two_tsgraphs(time_int_str_1, time_int_str_2):
    tsgraph_1 = nx.read_gml(g_tsgraph_gml_format.format(time_int_str_1, time_int_str_1))
    tsgraph_2 = nx.read_gml(g_tsgraph_gml_format.format(time_int_str_2, time_int_str_2))

    if len(tsgraph_1.edges) == 0 or len(tsgraph_2.edges) == 0:
        return 0.0

    l_nodes_common = list(set(list(tsgraph_1.nodes)) & set(list(tsgraph_2.nodes)))
    sig_1 = len(l_nodes_common) / len(tsgraph_1.nodes)
    sig_2 = len(l_nodes_common) / len(tsgraph_2.nodes)
    sub_tsgraph_1 = nx.subgraph(tsgraph_1, l_nodes_common)
    sub_tsgraph_2 = nx.subgraph(tsgraph_2, l_nodes_common)

    for edge in sub_tsgraph_1.edges.data('weight'):
        if str(edge[2]).lower() == 'nan':
            edge[2] = 0.0
    for edge in sub_tsgraph_2.edges.data('weight'):
        if str(edge[2]).lower() == 'nan':
            edge[2] = 0.0

    l_edge_weights_1 = []
    l_edge_weights_2 = []
    removed_graph = nx.DiGraph()
    for edge_1 in sub_tsgraph_1.edges:
        l_edge_weights_1.append(1)
        if edge_1 in sub_tsgraph_2.edges:
            l_edge_weights_2.append(1)
            removed_graph.add_edge(edge_1[0], edge_1[1])
            # sub_fsgraph_2.remove_edge(edge_1[0], edge_1[1])
        else:
            l_edge_weights_2.append(0)
    for edge_2 in sub_tsgraph_2.edges:
        if edge_2 not in removed_graph.edges:
            l_edge_weights_2.append(1)
            l_edge_weights_1.append(0)

    r_edge_w_1 = stats.rankdata(l_edge_weights_1)
    r_edge_w_2 = stats.rankdata(l_edge_weights_2)
    rho, p = stats.spearmanr(r_edge_w_1, r_edge_w_2)
    return rho, sig_1, sig_2


def get_undiedges_from_digraph(digraph):
    d_undiedges = dict()
    for edge_tup in digraph.edges.data():
        if (edge_tup[0], edge_tup[1]) not in d_undiedges:
            d_undiedges[(edge_tup[0], edge_tup[1])] = edge_tup[2]
        else:
            d_undiedges[(edge_tup[0], edge_tup[1])] += edge_tup[2]
    return d_undiedges


def nmi_for_two_graph_communities(time_int_str_1, time_int_str_2):
    with open(g_fsgraph_community_cluster_format.format(time_int_str_1, time_int_str_1), 'r') as in_fd:
        d_com_1 = json.load(in_fd)
        in_fd.close()
    fs_ig_1 = ig.load(g_fsgraph_gml_format.format(time_int_str_1, time_int_str_1))
    if len(d_com_1) != len(fs_ig_1.vs):
        raise Exception('Incorrect clustering %s' % time_int_str_1)
    with open(g_fsgraph_community_cluster_format.format(time_int_str_2, time_int_str_2), 'r') as in_fd:
        d_com_2 = json.load(in_fd)
        in_fd.close()
    fs_ig_2 = ig.load(g_fsgraph_gml_format.format(time_int_str_2, time_int_str_2))
    if len(d_com_2) != len(fs_ig_2.vs):
        raise Exception('Incorrect clustering %s' % time_int_str_2)
    l_cluster_1 = []
    l_cluster_2 = []
    max_label_1 = max(set(d_com_1.values()))
    max_label_2 = max(set(d_com_2.values()))
    for uid_1 in d_com_1:
        l_cluster_1.append(d_com_1[uid_1])
        if uid_1 in d_com_2:
            l_cluster_2.append(d_com_2[uid_1])
            del d_com_2[uid_1]
        else:
            max_label_2 += 1
            l_cluster_2.append(max_label_2)
    for uid_2 in d_com_2:
        l_cluster_2.append(d_com_2[uid_2])
        max_label_1 += 1
        l_cluster_1.append(max_label_1)

    mi = metrics.normalized_mutual_info_score(l_cluster_1, l_cluster_2, average_method='arithmetic')
    return mi


def rig_nmi_for_two_graph_communities(time_int_str_1, time_int_str_2):
    with open(g_fsgraph_community_cluster_format.format(time_int_str_1, time_int_str_1), 'r') as in_fd:
        d_com_1 = json.load(in_fd)
        in_fd.close()
    fs_ig_1 = ig.load(g_fsgraph_gml_format.format(time_int_str_1, time_int_str_1))
    if len(d_com_1) != len(fs_ig_1.vs):
        raise Exception('Incorrect clustering %s' % time_int_str_1)
    with open(g_fsgraph_community_cluster_format.format(time_int_str_2, time_int_str_2), 'r') as in_fd:
        d_com_2 = json.load(in_fd)
        in_fd.close()
    fs_ig_2 = ig.load(g_fsgraph_gml_format.format(time_int_str_2, time_int_str_2))
    if len(d_com_2) != len(fs_ig_2.vs):
        raise Exception('Incorrect clustering %s' % time_int_str_2)

    l_uid_common = list(set(d_com_1.keys()) & set(d_com_2.keys()))
    sig_1 = len(l_uid_common) / len(d_com_1)
    sig_2 = len(l_uid_common) / len(d_com_2)

    l_cluster_1 = []
    l_cluster_2 = []
    for uid in l_uid_common:
        l_cluster_1.append(d_com_1[uid])
        l_cluster_2.append(d_com_2[uid])
    nmi = metrics.normalized_mutual_info_score(l_cluster_1, l_cluster_2, average_method='arithmetic')

    return nmi, sig_1, sig_2


def nmi_for_graph_community_pairs(l_batch, tid):
    d_comp_rets = dict()
    for time_int_pair in l_batch:
        time_int_1_str = data_preprocessing_utils.time_int_to_time_int_str(time_int_pair[0])
        time_int_2_str = data_preprocessing_utils.time_int_to_time_int_str(time_int_pair[1])
        # nmi = nmi_for_two_graph_communities(time_int_1_str, time_int_2_str)
        nmi, sig_1, sig_2 = rig_nmi_for_two_graph_communities(time_int_1_str, time_int_2_str)
        # if time_int_1_str + ':' + time_int_2_str == '20180504_20180510:20180606_20180612':
        #     print()
        d_comp_rets[time_int_1_str + ':' + time_int_2_str] = {'nmi': nmi, 'sig_1': sig_1, 'sig_2':sig_2}
    with open(g_fsgraph_com_inter_format.format(tid), 'w+') as out_fd:
        json.dump(d_comp_rets, out_fd)
        out_fd.close()


# we use the union of edges to compute nmi
def nmi_for_two_graphs(graph_1, graph_2):
    if len(graph_1.edges) == 0 or len(graph_2.edges) == 0:
        return 0.0
    if nx.is_directed(graph_1):
        d_edges_1 = get_undiedges_from_digraph(graph_1)
    else:
        d_edges_1 = graph_1.edges.data()
    if nx.is_directed(graph_2):
        d_edges_2 = get_undiedges_from_digraph(graph_2)
    else:
        d_edges_2 = graph_2.edges.data()

    l_edge_weights_1 = []
    l_edge_weights_2 = []
    for edge_tup in d_edges_1:
        l_edge_weights_1.append(d_edges_1[edge_tup]['weight'])
        if edge_tup in d_edges_2:
            l_edge_weights_2.append(d_edges_2[edge_tup]['weight'])
            del d_edges_2[edge_tup]
        else:
            l_edge_weights_2.append(0.0)
    if len(l_edge_weights_1) != len(l_edge_weights_2):
        raise Exception('Edge alignment for graph_1 is incorrect.')
    for edge_tup in d_edges_2:
        l_edge_weights_2.append(d_edges_2[edge_tup]['weight'])
        l_edge_weights_1.append(0.0)

    mi = metrics.normalized_mutual_info_score(l_edge_weights_1, l_edge_weights_2, average_method='arithmetic')
    return mi


def spearman_for_two_fsgraphs(fsgraph_1, fsgraph_2):
    if len(fsgraph_1.edges) == 0 or len(fsgraph_2.edges) == 0:
        return 0.0
    l_edge_weights_1 = []
    l_edge_weights_2 = []
    for edge_1 in fsgraph_1.edges.data():
        l_edge_weights_1.append(fsgraph_1.edges[edge_1[0], edge_1[1]]['weight'])
        if edge_1 in fsgraph_2.edges.data():
            l_edge_weights_2.append(fsgraph_2.edges[edge_1[0], edge_1[1]]['weight'])
            fsgraph_2.remove_edge(edge_1[0], edge_1[1])
        else:
            l_edge_weights_2.append(0.0)
    for edge_2 in fsgraph_2.edges.data():
        l_edge_weights_2.append(fsgraph_2.edges[edge_2[0], edge_2[1]]['weight'])
        l_edge_weights_1.append(0.0)

    rho, p = stats.spearmanr(l_edge_weights_1, l_edge_weights_2)
    return rho, p


def rig_nmi_for_two_fsgraphs(time_int_str_1, time_int_str_2):
    fsgraph_1 = nx.read_gml(g_fsgraph_gml_format.format(time_int_str_1, time_int_str_1))
    fsgraph_2 = nx.read_gml(g_fsgraph_gml_format.format(time_int_str_2, time_int_str_2))

    if len(fsgraph_1.edges) == 0 or len(fsgraph_2.edges) == 0:
        return 0.0

    l_nodes_common = list(set(list(fsgraph_1.nodes)) & set(list(fsgraph_2.nodes)))
    sig_1 = len(l_nodes_common) / len(fsgraph_1.nodes)
    sig_2 = len(l_nodes_common) / len(fsgraph_2.nodes)
    sub_fsgraph_1 = nx.subgraph(fsgraph_1, l_nodes_common)
    sub_fsgraph_2 = nx.subgraph(fsgraph_2, l_nodes_common)

    l_edge_weights_1 = []
    l_edge_weights_2 = []
    removed_graph = nx.DiGraph()
    for edge_1 in sub_fsgraph_1.edges:
        l_edge_weights_1.append(1)
        if edge_1 in sub_fsgraph_2.edges:
            l_edge_weights_2.append(1)
            removed_graph.add_edge(edge_1[0], edge_1[1])
            # sub_fsgraph_2.remove_edge(edge_1[0], edge_1[1])
        else:
            l_edge_weights_2.append(0)
    for edge_2 in sub_fsgraph_2.edges:
        if edge_2 not in removed_graph.edges:
            l_edge_weights_2.append(1)
            l_edge_weights_1.append(0)

    nmi = metrics.normalized_mutual_info_score(l_edge_weights_1, l_edge_weights_2, average_method='arithmetic')
    return nmi, sig_1, sig_2


def nmi_for_graph_pairs(l_batch, inter_ret_format, t_id, mode):
    d_inter_rets = dict()
    for time_int_pair in l_batch:
        time_int_1_str = data_preprocessing_utils.time_int_to_time_int_str(time_int_pair[0])
        time_int_2_str = data_preprocessing_utils.time_int_to_time_int_str(time_int_pair[1])
        logging.debug('Compare %s and %s.' % (time_int_1_str, time_int_2_str))
        if mode == 'fs:fs':
            nmi, sig_1, sig_2 = rig_nmi_for_two_fsgraphs(time_int_1_str, time_int_2_str)
            d_inter_rets[time_int_1_str + ':' + time_int_2_str] = {'nmi': nmi, 'sig_1': sig_1, 'sig_2': sig_2}
        elif mode == 'ts:ts':
            rho, sig_1, sig_2 = rig_spearman_for_two_tsgraphs(time_int_1_str, time_int_2_str)
            d_inter_rets[time_int_1_str + ':' + time_int_2_str] = {'rho': rho, 'sig_1': sig_1, 'sig_2': sig_2}
        # elif mode == 'fs:ts':
        #     nmi = nmi_for_two_graphs(graph_1, graph_2)
    with open(inter_ret_format.format(t_id), 'w+') as out_fd:
        json.dump(d_inter_rets, out_fd)
        out_fd.close()
    logging.debug('%s is done.' % t_id)


def spearman_for_graph_pairs(l_batch, gml_path_1_format, gml_path_2_format, inter_ret_format, t_id, mode):
    d_inter_rets = dict()
    for time_int_pair in l_batch:
        time_int_1_str = data_preprocessing_utils.time_int_to_time_int_str(time_int_pair[0])
        time_int_2_str = data_preprocessing_utils.time_int_to_time_int_str(time_int_pair[1])
        logging.debug('Compare %s and %s.' % (time_int_1_str, time_int_2_str))
        graph_1 = nx.read_gml(gml_path_1_format.format(time_int_1_str, time_int_1_str))
        graph_2 = nx.read_gml(gml_path_2_format.format(time_int_2_str, time_int_2_str))
        if mode == 'fs:fs':
            rho, p = spearman_for_two_fsgraphs(graph_1, graph_2)
        elif mode == 'ts:ts':
            pass
        elif mode == 'fs:ts':
            pass
        d_inter_rets[time_int_1_str + ':' + time_int_2_str] = {'rho': rho, 'p': p}
    with open(inter_ret_format.format(t_id), 'w+') as out_fd:
        json.dump(d_inter_rets, out_fd)
        out_fd.close()
    logging.debug('%s is done.' % t_id)


def nmi_multithreads(l_time_ints):
    l_time_int_pairs = []
    for i in range(0, len(l_time_ints) - 1):
        for j in range(i + 1, len(l_time_ints)):
            l_time_int_pairs.append((l_time_ints[i], l_time_ints[j]))
    logging.debug('%s time interval pairs in total' % len(l_time_int_pairs))
    batch_size = math.ceil(len(l_time_int_pairs) / multiprocessing.cpu_count())
    l_batches = []
    for k in range(0, len(l_time_int_pairs), batch_size):
        if k + batch_size < len(l_time_int_pairs):
            l_batches.append(l_time_int_pairs[k:k + batch_size])
        else:
            l_batches.append(l_time_int_pairs[k:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_batches:
        # t = threading.Thread(target=spearman_for_graph_pairs,
        #                      args=(l_each_batch,
        #                            g_fsgraph_gml_format,
        #                            g_fsgraph_gml_format,
        #                            g_fsgraph_inter_format,
        #                            t_id,
        #                            'fs:fs'))

        # t = threading.Thread(target=nmi_for_graph_pairs,
        #                      args=(l_each_batch,
        #                            g_tsgraph_inter_format,
        #                            t_id,
        #                            'ts:ts'))

        t = threading.Thread(target=nmi_for_graph_community_pairs, args=(l_each_batch, t_id))
        t.setName('tsgraph_sample_comp_t_' + str(t_id))
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

    logging.debug('All graph comparisons are done.')


def get_item_idx_map(l_time_ints):
    d_item_idx_map = dict()
    idx = 0
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        d_item_idx_map[time_int_str] = idx
        idx += 1
    return d_item_idx_map


def get_sig_coms(time_int_str):
    # d_coms = dict()
    with open(g_fsgraph_community_cluster_format.format(time_int_str, time_int_str), 'r') as in_fd:
        d_members = json.load(in_fd)
        in_fd.close()
    d_coms = {com: [] for com in set(d_members.values())}
    for uid in d_members:
        d_coms[d_members[uid]].append(uid)
    d_sig_coms = dict()
    for com in d_coms:
        if len(d_coms[com]) >= g_sig_com_size_threshold:
            d_sig_coms[com] = d_coms[com]
    return d_sig_coms


def get_av_label(l_nan_labels):
    if len(l_nan_labels) == 0:
        return 0
    else:
        l_sorted_nan_labels = sorted(l_nan_labels)
        cand = l_sorted_nan_labels[0] + 1
        while cand in l_sorted_nan_labels:
            cand += 1
        return cand


def get_fsgraphs_tsgraphs_coms_for_one_sig_com(this_time_int_str, l_sig_com_uids):
    l_time_ints = data_preprocessing_utils.read_time_ints()
    d_sig_com_series = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        if time_int_str <= this_time_int_str:
            continue
        if not os.path.exists(g_fsgraph_gml_format.format(time_int_str, time_int_str)) \
                or not os.path.exists(g_tsgraph_gml_format.format(time_int_str, time_int_str)) \
                or not os.path.exists(g_fsgraph_community_cluster_format.format(time_int_str, time_int_str)):
            continue
        d_sig_com_series[time_int_str] = dict()
        # fsgraph = nx.read_gml(g_fsgraph_gml_format.format(time_int_str, time_int_str))
        # tsgraph = nx.read_gml(g_tsgraph_gml_format.format(time_int_str, time_int_str))
        # sub_fsgraph = nx.subgraph(fsgraph, l_sig_com_uids)
        # sub_tsgraph = nx.subgraph(tsgraph, l_sig_com_uids)
        # d_sig_com_series[time_int_str]['fs'] = sub_fsgraph
        # d_sig_com_series[time_int_str]['ts'] = sub_tsgraph
        with open(g_fsgraph_community_cluster_format.format(time_int_str, time_int_str), 'r') as in_fd:
            member_json = json.load(in_fd)
            in_fd.close()
        l_sig_com_members = []
        l_gap_idx = []
        for idx, uid in enumerate(l_sig_com_uids):
            if uid in member_json:
                l_sig_com_members.append(member_json[uid])
            else:
                l_gap_idx.append(idx)
        l_nav_mem_labels = list(set(l_sig_com_members))
        for idx in l_gap_idx:
            new_mem = get_av_label(l_nav_mem_labels)
            l_sig_com_members.insert(idx, new_mem)
            l_nav_mem_labels.append(new_mem)
        d_sig_com_series[time_int_str]['com'] = l_sig_com_members
    return d_sig_com_series


def draw_colormap_comp_rets(g_fsgraph_inter_path, d_item_idx_map):
    l_inter_files = os.listdir(g_fsgraph_inter_path)
    mat_size = len(d_item_idx_map)
    ret_mat = np.zeros((mat_size, mat_size))
    sig_mat = np.zeros((mat_size, mat_size))
    d_inter_rets = dict()
    for inter_file in l_inter_files:
        with open(g_fsgraph_inter_path + inter_file, 'r') as in_fd:
            inter_ret_json = json.load(in_fd)
            if not g_spearman:
                d_inter_rets = dict(list(d_inter_rets.items()) + list(inter_ret_json.items()))
            else:
                for key in inter_ret_json:
                    d_inter_rets[key] = inter_ret_json[key]['rho']
            in_fd.close()
    for time_int_pair in d_inter_rets:
        time_int_strs = time_int_pair.split(':')
        time_int_1_idx = d_item_idx_map[time_int_strs[0]]
        time_int_2_idx = d_item_idx_map[time_int_strs[1]]
        if time_int_1_idx == time_int_2_idx:
            ret_mat[time_int_1_idx][time_int_2_idx] = 1.0
            sig_mat[time_int_1_idx][time_int_2_idx] = 1.0
        if not g_spearman:
            ret_mat[time_int_1_idx][time_int_2_idx] = float(d_inter_rets[time_int_pair]['nmi'])
            ret_mat[time_int_2_idx][time_int_1_idx] = float(d_inter_rets[time_int_pair]['nmi'])
            sig_mat[time_int_1_idx][time_int_2_idx] = float(d_inter_rets[time_int_pair]['sig_2'])
            sig_mat[time_int_2_idx][time_int_1_idx] = float(d_inter_rets[time_int_pair]['sig_1'])
        else:
            ret_mat[time_int_1_idx][time_int_2_idx] = float(d_inter_rets[time_int_pair])
            ret_mat[time_int_2_idx][time_int_1_idx] = float(d_inter_rets[time_int_pair])
    for i in range(0, ret_mat.shape[0]):
        ret_mat[i][i] = 1.0
        sig_mat[i][i] = 1.0
    min = 0.0
    max = 1.0

    if g_normalize_weights:
        min = 1.0
        max = -1.0
        for i in range(0, ret_mat.shape[0]):
            cur_min = ret_mat[i].min()
            if cur_min < min:
                min = cur_min
            cur_max = ret_mat[i].max()
            if cur_max > max:
                max = cur_max

        for i in range(0, ret_mat.shape[0]):
            for j in range(i, ret_mat.shape[1]):
                if i == j:
                    ret_mat[i][j] = 1.0
                ret_mat[i][j] = (ret_mat[i][j] - min) / (max - min)
                ret_mat[j][i] = (ret_mat[j][i] - min) / (max - min)

    l_ticks = [i for i in range(0, 131, 10)]
    # plt.imshow(ret_mat, cmap='gray', vmin=min, vmax=max)
    plt.imshow(ret_mat)
    plt.colorbar()
    plt.xticks(l_ticks)
    plt.yticks(l_ticks)
    plt.show()

    l_ticks = [i for i in range(0, 131, 10)]
    # plt.imshow(sig_mat, cmap='gray', vmin=min, vmax=max)
    plt.imshow(sig_mat)
    plt.colorbar()
    plt.xticks(l_ticks)
    plt.yticks(l_ticks)
    plt.show()
    # mat_size = int(math.sqrt(len(d_inter_rets) * 8 + 1) / 2)


def one_sub_tsgraph(time_int_str):
    # with open(g_tsgraph_gml_format.format(time_int_str, time_int_str), 'r') as in_fd:
    logging.debug('%s subtsgraph starts...' % time_int_str)
    if not os.path.exists(g_tsgraph_gml_format.format(time_int_str, time_int_str)):
        logging.debug('No %s tsgraph.' % time_int_str)
        return
    l_del_edges = []
    ts_ig = ig.load(g_tsgraph_gml_format.format(time_int_str, time_int_str))
    for edge in ts_ig.es:
        if type(edge['weight']) != float or edge['weight'] < g_tsgraph_sim_threshold:
            l_del_edges.append(edge)
    ts_ig.delete_edges(l_del_edges)
    ig.write(ts_ig, g_subtsgraph_gml_format.format(time_int_str, time_int_str))
    logging.debug('%s subgraph is done.' % time_int_str)


def multiple_sub_tsgraphs(l_time_ints, tid):
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        one_sub_tsgraph(time_int_str)
    logging.debug('%s subgraphs are done.' % tid)


def sub_tsgraph_multithread(l_time_ints):
    batch_size = math.ceil(len(l_time_ints) / multiprocessing.cpu_count())
    l_l_time_ints = []
    for i in range(0, len(l_time_ints), batch_size):
        if i + batch_size < len(l_time_ints):
            l_l_time_ints.append(l_time_ints[i:i + batch_size])
        else:
            l_l_time_ints.append(l_time_ints[i:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_time_ints:
        t = threading.Thread(target=multiple_sub_tsgraphs, args=(l_each_batch, t_id))
        t.setName('subtsgraph_t_' + str(t_id))
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

    logging.debug('All sugtsgraphs have been written.')


def spearman_fsgraph_vs_topic_embed():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    l_sorted_time_int_strs = sorted([data_preprocessing_utils.time_int_to_time_int_str(time_int)
                                     for time_int in l_time_ints])

    ret_size = len(l_sorted_time_int_strs)

    fs_mat = np.zeros((ret_size, ret_size))
    temb_mat = np.zeros((ret_size, ret_size))
    l_fs = []
    l_temb = []

    l_inter_files = os.listdir(g_fsgraph_inter_path)
    d_fsgraph_rets = dict()
    for inter_file in l_inter_files:
        with open(g_fsgraph_inter_path + inter_file, 'r') as in_fd:
            inter_ret_json = json.load(in_fd)
            d_fsgraph_rets = dict(list(d_fsgraph_rets.items()) + list(inter_ret_json.items()))
            in_fd.close()
    for time_int_pair in d_fsgraph_rets:
        l_time_int_pairs = time_int_pair.split(':')
        time_int_1 = l_time_int_pairs[0]
        time_int_2 = l_time_int_pairs[1]
        idx_1 = l_sorted_time_int_strs.index(time_int_1)
        idx_2 = l_sorted_time_int_strs.index(time_int_2)
        fs_mat[idx_1][idx_2] = d_fsgraph_rets[time_int_pair]['nmi']
        fs_mat[idx_2][idx_1] = d_fsgraph_rets[time_int_pair]['nmi']
    for i in range(0, fs_mat.shape[0]):
        fs_mat[i][i] = 1.0

    d_topic_embed_comps = dict()
    l_inter_files = os.listdir(g_topic_embed_comp_inter_rets_path)
    for inter_file in l_inter_files:
        with open(g_topic_embed_comp_inter_rets_path + inter_file, 'r') as in_fd:
            t_emb_comp_json = json.load(in_fd)
            d_topic_embed_comps = dict(list(d_topic_embed_comps.items()) + list(t_emb_comp_json.items()))
            in_fd.close()
    for time_int_pair in d_topic_embed_comps:
        l_time_int_pairs = time_int_pair.split(':')
        time_int_1 = l_time_int_pairs[0]
        time_int_2 = l_time_int_pairs[1]
        idx_1 = l_sorted_time_int_strs.index(time_int_1)
        idx_2 = l_sorted_time_int_strs.index(time_int_2)
        temb_mat[idx_1][idx_2] = d_topic_embed_comps[time_int_pair]
        temb_mat[idx_2][idx_1] = d_topic_embed_comps[time_int_pair]
    for i in range(0, temb_mat.shape[0]):
        temb_mat[i][i] = 1.0

    for i in range(0, fs_mat.shape[0]):
        l_fs += list(fs_mat[i])
    l_fs = stats.rankdata(l_fs)

    for i in range(0, temb_mat.shape[0]):
        l_temb += list(temb_mat[i])
    l_temb = stats.rankdata(l_temb)

    rho, p = stats.spearmanr(l_fs, l_temb)
    print(rho, p)



def main():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    #
    # l_nan_time_ints = []
    # for time_int in l_time_ints:
    #     time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
    #     if time_int_str > g_tsgraph_end_time or time_int_str < g_tsgraph_start_time:
    #         l_nan_time_ints.append(time_int)
    #     if not os.path.exists(g_tsgraph_gml_format.format(time_int_str, time_int_str)):
    #         l_nan_time_ints.append(time_int)
    #     if time_int_str in g_bad_tsgraphs:
    #         l_nan_time_ints.append(time_int)
    # l_time_ints = [t for t in l_time_ints if t not in l_nan_time_ints]

    # nmi_multithreads(l_time_ints)

    d_item_idx_map = get_item_idx_map(l_time_ints)
    draw_colormap_comp_rets(g_fsgraph_com_inter_path, d_item_idx_map)

    # sub_tsgraph_multithread(l_time_ints)

    # spearman_fsgraph_vs_topic_embed()


def main_sig_com(time_int_str):
    d_sig_coms = get_sig_coms(time_int_str)
    for sig_com in d_sig_coms:
        d_sig_com_series = get_fsgraphs_tsgraphs_coms_for_one_sig_com(time_int_str, d_sig_coms[sig_com])
    print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    # main_sig_com('20180404_20180410')
    # nmi_for_two_graph_communities('20180913_20180919', '20181118_20181124')
    # nmi_for_two_graph_communities('20180404_20180410', '20180407_20180413')
