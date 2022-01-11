import json
import networkx as nx
import matplotlib.pyplot as plt

g_task_name = '20180415_20180422'

g_fs_graph_data_file_path_fortmat = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/tw_wh_fsgraph_{1}_10_sample.json'


def draw_graph(fs_graph):
    nx.draw_networkx(fs_graph)


def main():
    with open(g_fs_graph_data_file_path_fortmat.format(g_task_name, g_task_name), 'r') as in_fd:
        graph_data = json.load(in_fd)
        fs_graph = nx.adjacency_graph(graph_data)
        print(nx.info(fs_graph))
        sum_edge_cnt = sum([fs_graph.get_edge_data(*edge)['r'] + fs_graph.get_edge_data(*edge)['t'] + fs_graph.get_edge_data(*edge)['q'] for edge in fs_graph.edges])
        print('Sum count on edge: %s' % str(sum_edge_cnt))
        reply_cnt = sum([fs_graph.get_edge_data(*edge)['r'] for edge in fs_graph.edges])
        retweet_cnt = sum([fs_graph.get_edge_data(*edge)['t'] for edge in fs_graph.edges])
        quote_cnt = sum([fs_graph.get_edge_data(*edge)['q'] for edge in fs_graph.edges])
        print('Reply count: %s; Retweet count: %s; Quote count: %s.' % (reply_cnt, retweet_cnt, quote_cnt))
        print('Average count: %f' % float(sum_edge_cnt/len(fs_graph.edges)))
        # draw_graph(fs_graph)
    in_fd.close()




if __name__ == '__main__':
    main()