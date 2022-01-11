import json
import networkx as nx
import logging


g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
            '20180527_20180603', '20180610_20180617', '20180624_20180701',
            '20180708_20180715', '20180722_20180729', '20180805_20180812',
            '20180819_20180826']
g_path_prefix_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/'
g_tsgraph_file_path_format = g_path_prefix_format + 'tw_wh_tsgraph_{1}_full_fs.json'
g_normtsgraph_file_path_format = g_path_prefix_format + 'tw_wh_normtsgraph_{1}_full.json'


def to_norm_tsgraph(tsgraph):
    normtsgraph = nx.Graph()
    for edge in tsgraph.edges.data('weight'):
        if not normtsgraph.has_edge(edge[0], edge[1]):
            normtsgraph.add_edge(edge[0], edge[1], weight=(edge[2]+1.0)/2.0)
        else:
            logging.error('(%s, %s) appears more than once.' % (edge[0], edge[1]))
    print(nx.info(normtsgraph))
    return normtsgraph


def main():
    for week in g_l_weeks:
        print('%s:' % week)
        print('norm_tsgraph:')
        with open(g_tsgraph_file_path_format.format(week, week), 'r') as in_fd:
            tsgraph_data = json.load(in_fd)
            tsgraph = nx.adjacency_graph(tsgraph_data)
            normtsgraph = to_norm_tsgraph(tsgraph)
            with open(g_normtsgraph_file_path_format.format(week, week), 'w+') as out_fd:
                normtsgraph_data = nx.adjacency_data(normtsgraph)
                json.dump(normtsgraph_data, out_fd, indent=4)
            out_fd.close()
        in_fd.close()
        print()



if __name__ == '__main__':
    main()