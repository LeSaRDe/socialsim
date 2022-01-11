import json
import networkx as nx
import logging



g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
            '20180527_20180603', '20180610_20180617', '20180624_20180701',
            '20180708_20180715', '20180722_20180729', '20180805_20180812',
            '20180819_20180826']
g_path_prefix_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/'
g_fsgraph_file_path_format = g_path_prefix_format + 'tw_wh_fsgraph_{1}_full.json'
g_normfsgraph_file_path_format = g_path_prefix_format + 'tw_wh_normfsgraph_{1}_full.json'
g_fsgraph_full_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_friendship_graph.json'
g_normfsgraph_full_file_path ='/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_friendship_graph_norm.json'
g_fsgraph_full_gml_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_friendship_graph_norm.gml'


def to_norm_undigraph(digraph):
    undigraph = nx.Graph()
    # d_undi_edges = dict()
    max_weight = 0.0
    for edge in digraph.edges.data():
        r = digraph.edges[edge[0], edge[1]]['r']
        q = digraph.edges[edge[0], edge[1]]['q']
        t = digraph.edges[edge[0], edge[1]]['t']
        rqt_sum = r + q + t
        if not undigraph.has_edge(edge[0], edge[1]):
            undigraph.add_edge(edge[0], edge[1], weight=rqt_sum)
        else:
            undigraph.edges[edge[0], edge[1]]['weight'] += rqt_sum
        if undigraph.edges[edge[0], edge[1]]['weight'] > max_weight:
            max_weight = undigraph.edges[edge[0], edge[1]]['weight']
    for edge in undigraph.edges.data('weight'):
        undigraph.edges[edge[0], edge[1]]['weight'] = edge[2] / max_weight
    print(nx.info(undigraph))
    return undigraph


def main():
    for week in g_l_weeks:
        print('%s:' % week)
        print('norm_fsgraph:')
        with open(g_fsgraph_file_path_format.format(week, week), 'r') as in_fd:
            fsgraph_data = json.load(in_fd)
            fsgraph = nx.adjacency_graph(fsgraph_data)
            normfsgraph = to_norm_undigraph(fsgraph)
            with open(g_normfsgraph_file_path_format.format(week, week), 'w+') as out_fd:
                normfsgraph_data = nx.adjacency_data(normfsgraph)
                json.dump(normfsgraph_data, out_fd, indent=4)
            out_fd.close()
        in_fd.close()
        print()


def main_full():
    with open(g_fsgraph_full_file_path, 'r') as in_fd:
        fsgraph_data = json.load(in_fd)
        fsgraph = nx.adjacency_graph(fsgraph_data)
        normfsgraph = to_norm_undigraph(fsgraph)
        in_fd.close()
    with open(g_normfsgraph_full_file_path, 'w+') as out_fd:
        normfsgraph_data = nx.adjacency_data(normfsgraph)
        json.dump(normfsgraph_data, out_fd, indent=4)
        out_fd.close()
        nx.write_gml(normfsgraph, g_fsgraph_full_gml_file_path)



if __name__ == '__main__':
    # main()
    main_full()