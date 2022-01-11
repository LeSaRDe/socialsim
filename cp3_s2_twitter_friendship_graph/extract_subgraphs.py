import json
import networkx as nx
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils

g_sampled_tweet_data_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_20180401_20180408_by_time_sorted.json'
g_full_tweet_date_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_dict.json'
g_source_graph_data_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_friendship_graph.json'
g_sampled_graph_file_path = 'Tng_an_WH_Twitter_v2_friendship_graph_sampled_20180401_20180408.json'


def main():
    with open(g_source_graph_data_file_path, 'r') as in_fd:
        source_graph_data = json.load(in_fd)
    in_fd.close()
    source_graph = nx.adjacency_graph(source_graph_data)
    print('Source Graph Info:')
    print(nx.info(source_graph))

    with open(g_sampled_tweet_data_file_path, 'r') as in_fd:
        l_sampled_users = []
        line = in_fd.readline()
        while line:
            json_data = json.loads(line)
            user_id = json_data['user']['id_str_h']
            l_sampled_users.append(user_id)
            line = in_fd.readline()
        # sampled_data = json.load(in_fd)
        # d_sampled_users = dict()
        # for tid in sampled_data:
        #     user_id = sampled_data[tid]['user']['id_str_h']
        #     if user_id not in d_sampled_users:
        #         d_sampled_users[user_id] = None
        # del sampled_data
    in_fd.close()

    sampled_subgraph = source_graph.subgraph(l_sampled_users)
    print('Sampled Subgraph Info:')
    print(nx.info(sampled_subgraph))

    with open(g_sampled_graph_file_path, 'w+') as out_fd:
        sampled_subgraph_data = nx.adjacency_data(sampled_subgraph)
        json.dump(sampled_subgraph_data, out_fd, indent=4)
    out_fd.close()


if __name__ == '__main__':
    main()