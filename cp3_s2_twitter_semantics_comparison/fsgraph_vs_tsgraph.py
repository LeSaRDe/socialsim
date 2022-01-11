import json
import networkx as nx
from sklearn import metrics
import numpy as np

g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
             '20180527_20180603', '20180610_20180617', '20180624_20180701',
             '20180708_20180715', '20180722_20180729', '20180805_20180812',
             '20180819_20180826']
g_path_prefix_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/'
g_fsgraph_file_path_fromat = g_path_prefix_format + 'tw_wh_normfsgraph_{1}_full.json'
g_tsgraph_file_path_fromat = g_path_prefix_format + 'tw_wh_normtsgraph_{1}_full.json'


def compute_mutual_info(normfsgraph, normtsgraph):
    d_edges = dict()
    for edge in normfsgraph.edges.data('weight'):
        d_edges[(edge[0], edge[1])] = [edge[2], normtsgraph.edges[edge[0], edge[1]]['weight']]
    l_fs_weights = []
    l_ts_weights = []
    for item in d_edges.values():
        l_fs_weights.append(item[0])
        l_ts_weights.append(item[1])
    # l_fs_weights = [item[0] for item in list(d_edges.values())]
    # l_ts_weights = [item[1] for item in list(d_edges.values())]
    mi = metrics.normalized_mutual_info_score(l_fs_weights, l_ts_weights, average_method='arithmetic')
    return mi


def main():
    for week in g_l_weeks:
        print('%s:' % week)
        with open(g_fsgraph_file_path_fromat.format(week, week), 'r') as in_fs_fd:
            normfsgraph_data = json.load(in_fs_fd)
            normfsgraph = nx.adjacency_graph(normfsgraph_data)
        in_fs_fd.close()
        with open(g_tsgraph_file_path_fromat.format(week, week), 'r') as in_ts_fd:
            normtsgraph_data = json.load(in_ts_fd)
            normtsgraph = nx.adjacency_graph(normtsgraph_data)
        in_ts_fd.close()
        mi = compute_mutual_info(normfsgraph, normtsgraph)
        print('Mutual information = %s' % mi)
        print()


def compute_mutual_info_tsgraph(tsgraph_1, tsgraph_2):
    d_edges = dict()
    for edge in tsgraph_1.edges.data('weight'):
        if tsgraph_2.has_edge(edge[0], edge[1]):
            d_edges[(edge[0], edge[1])] = [edge[2], tsgraph_2.edges[edge[0], edge[1]]['weight']]
        else:
            d_edges[(edge[0], edge[1])] = [edge[2], 0.0]

    for edge in tsgraph_2.edges.data('weight'):
        if (edge[0], edge[1]) in d_edges or (edge[1], edge[0]) in d_edges:
            continue
        else:
            d_edges[(edge[0], edge[1])] = [0.0, edge[2]]

    l_ts_1_weights = [item[0] for item in list(d_edges.values())]
    l_ts_2_weights = [item[1] for item in list(d_edges.values())]
    mi = metrics.normalized_mutual_info_score(l_ts_1_weights, l_ts_2_weights, average_method='arithmetic')
    return mi


def tsgraph_consecutive_main():
    for i in range(0, len(g_l_weeks)-1):
        with open(g_tsgraph_file_path_fromat.format(g_l_weeks[i], g_l_weeks[i]), 'r') as in_ts_1_fd:
            tsgraph_1_data = json.load(in_ts_1_fd)
            tsgraph_1 = nx.adjacency_graph(tsgraph_1_data)
        in_ts_1_fd.close()
        # for j in range(0, len(g_l_weeks)):
        print('%s vs %s' % (g_l_weeks[i], g_l_weeks[i+1]))
        with open(g_tsgraph_file_path_fromat.format(g_l_weeks[i+1], g_l_weeks[i+1]), 'r') as in_ts_2_fd:
            tsgraph_2_data = json.load(in_ts_2_fd)
            tsgraph_2 = nx.adjacency_graph(tsgraph_2_data)
        in_ts_1_fd.close()
        mi = compute_mutual_info_tsgraph(tsgraph_1, tsgraph_2)
        print('mi = %s' % mi)
        print()


def fsgraph_consecutive_main():
    for i in range(0, len(g_l_weeks)-1):
        with open(g_fsgraph_file_path_fromat.format(g_l_weeks[i], g_l_weeks[i]), 'r') as in_ts_1_fd:
            tsgraph_1_data = json.load(in_ts_1_fd)
            tsgraph_1 = nx.adjacency_graph(tsgraph_1_data)
        in_ts_1_fd.close()
        # for j in range(0, len(g_l_weeks)):
        print('%s vs %s' % (g_l_weeks[i], g_l_weeks[i+1]))
        with open(g_fsgraph_file_path_fromat.format(g_l_weeks[i+1], g_l_weeks[i+1]), 'r') as in_ts_2_fd:
            tsgraph_2_data = json.load(in_ts_2_fd)
            tsgraph_2 = nx.adjacency_graph(tsgraph_2_data)
        in_ts_1_fd.close()
        mi = compute_mutual_info_tsgraph(tsgraph_1, tsgraph_2)
        print('mi = %s' % mi)
        print()


def tsgraph_pairwise_main():
    for i in range(0, len(g_l_weeks) - 1):
        with open(g_tsgraph_file_path_fromat.format(g_l_weeks[i], g_l_weeks[i]), 'r') as in_ts_1_fd:
            tsgraph_1_data = json.load(in_ts_1_fd)
            tsgraph_1 = nx.adjacency_graph(tsgraph_1_data)
        in_ts_1_fd.close()
        for j in range(0, len(g_l_weeks)):
            print('%s vs %s' % (g_l_weeks[i], g_l_weeks[j]))
            with open(g_tsgraph_file_path_fromat.format(g_l_weeks[j], g_l_weeks[j]), 'r') as in_ts_2_fd:
                tsgraph_2_data = json.load(in_ts_2_fd)
                tsgraph_2 = nx.adjacency_graph(tsgraph_2_data)
            in_ts_1_fd.close()
            mi = compute_mutual_info_tsgraph(tsgraph_1, tsgraph_2)
            print('mi = %s' % mi)
            print()

def fsgraph_pairwise_main():
    for i in range(0, len(g_l_weeks) - 1):
        with open(g_fsgraph_file_path_fromat.format(g_l_weeks[i], g_l_weeks[i]), 'r') as in_ts_1_fd:
            tsgraph_1_data = json.load(in_ts_1_fd)
            tsgraph_1 = nx.adjacency_graph(tsgraph_1_data)
        in_ts_1_fd.close()
        for j in range(0, len(g_l_weeks)):
            print('%s vs %s' % (g_l_weeks[i], g_l_weeks[j]))
            with open(g_fsgraph_file_path_fromat.format(g_l_weeks[j], g_l_weeks[j]), 'r') as in_ts_2_fd:
                tsgraph_2_data = json.load(in_ts_2_fd)
                tsgraph_2 = nx.adjacency_graph(tsgraph_2_data)
            in_ts_1_fd.close()
            mi = compute_mutual_info_tsgraph(tsgraph_1, tsgraph_2)
            print('mi = %s' % mi)
            print()


if __name__ == '__main__':
    main()
    # tsgraph_pairwise_main()
    # tsgraph_consecutive_main()
    # fsgraph_consecutive_main()
    # fsgraph_pairwise_main()