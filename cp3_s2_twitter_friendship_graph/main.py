import logging
import json
import networkx as nx
import time



g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
            '20180527_20180603', '20180610_20180617', '20180624_20180701',
            '20180708_20180715', '20180722_20180729', '20180805_20180812',
            '20180819_20180826']
# g_retweet_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_Retweet_Chain_WH_50.json'
# g_tweet_data_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2.json'
g_path_prefix_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/'
g_output_file_path_format = g_path_prefix_format + 'tw_wh_fsgraph_{1}_full.json'
g_friendship_data_file_path_format = g_path_prefix_format + 'tw_wh_fsgraph_data_{1}_full.json'
g_users_data_file_path_format = g_path_prefix_format + 'users_{1}_full.txt'


# def parse_graph_from_retweet(fs_graph, tweet_json):
#     with open(g_retweet_file_path, 'r') as in_fd:
#         l_lines = in_fd.readlines()
#         for i in range(1, len(l_lines)):
#             json_data = json.loads(l_lines[i])
#             source_tweet = json_data['retweeted_from_tweet_id_h']
#             target_tweet = json_data['']
#
#     in_fd.close()


# def extract_quote_retweet_relations(fs_graph):
#     try:
#         with open(g_tweet_data_file_path, 'r') as in_fd:
#             l_lines = in_fd.readlines()
#             for i in range(0, len(l_lines)):
#                 tweet = json.loads(l_lines[i])
#                 if 'in_reply_to_status_id_h' in tweet and tweet['in_reply_to_status_id_h'] != '':
#                     target = tweet['user']['id_str_h']
#                     source = tweet['in_reply_to_status_id_h']
#                     if fs_graph.has_edge(source, target):
#                         fs_graph.edges[source, target]['r'] += 1
#                     else:
#                         fs_graph.add_edge(source, target, r=1, t=0, q=0)
#                 elif 'retweeted_status' in tweet:
#                     target = tweet['user']['id_str_h']
#                     source = tweet['retweeted_status']['user']['id_str_h']
#                     if fs_graph.has_edge(source, target):
#                         fs_graph.edges[source, target]['t'] += 1
#                     else:
#                         fs_graph.add_edge(source, target, r=0, t=1, q=0)
#                 elif 'quoted_status' in tweet:
#                     target = tweet['user']['id_str_h']
#                     source = tweet['quoted_status']['user']['id_str_h']
#                     if fs_graph.has_edge(source, target):
#                         fs_graph.edges[source, target]['q'] += 1
#                     else:
#                         fs_graph.add_edge(source, target, r=0, t=0, q=1)
#                 if i % 5000 == 0 and i >= 5000:
#                     logging.debug('%s tweets have been scanned.' % i)
#         in_fd.close()
#         logging.debug('All tweets have been scanned.')
#         nx.info(fs_graph)
#     except Exception as e:
#         logging.error(e)


def extract_extract_quote_retweet_relations_by_time(fs_graph, task_name, start=None, end=None):
    with open(g_friendship_data_file_path_format.format(task_name, task_name), 'r') as in_fd:
        json_data = json.load(in_fd)
        for relation in json_data.values():
            source = relation[0]
            target = relation[1]
            type_t = relation[2]
            time_t = relation[3]
            if start is not None and time_t < start:
                continue
            if end is not None and time_t > end:
                continue
            if type_t == 'r':
                if fs_graph.has_edge(source, target):
                    fs_graph.edges[source, target]['r'] += 1
                else:
                    fs_graph.add_edge(source, target, r=1, t=0, q=0)
            elif type_t == 't':
                if fs_graph.has_edge(source, target):
                    fs_graph.edges[source, target]['t'] += 1
                else:
                    fs_graph.add_edge(source, target, r=0, t=1, q=0)
            elif type_t == 'q':
                if fs_graph.has_edge(source, target):
                    fs_graph.edges[source, target]['q'] += 1
                else:
                    fs_graph.add_edge(source, target, r=0, t=0, q=1)
    in_fd.close()
    return fs_graph


def add_vertices(fs_graph, task_name):
    with open(g_users_data_file_path_format.format(task_name, task_name), 'r') as in_fd:
        l_users = [user.strip() for user in in_fd.readlines()]
    in_fd.close()
    fs_graph.add_nodes_from(l_users)
    return fs_graph


def output_graph(fs_graph, task_name, suffix=None):
    out_file_path = g_output_file_path_format.format(task_name, task_name)
    if suffix != None:
        l_file_path_fields = out_file_path.split('.')
        out_file_path = l_file_path_fields[0] + '_' + suffix + '.' + l_file_path_fields[1]
    with open(out_file_path, 'w+') as out_fd:
        fs_graph_data = nx.adjacency_data(fs_graph)
        json.dump(fs_graph_data, out_fd, indent=4)
    out_fd.close()


# def main():
#     fs_graph = nx.DiGraph()
#     extract_quote_retweet_relations(fs_graph)
#     output_graph(fs_graph)


def main_by_month(month_str):
    fs_graph = nx.DiGraph()
    extract_extract_quote_retweet_relations_by_time(fs_graph, month_str + '01000000', month_str + '31235959')
    output_graph(fs_graph, month_str)


def main(start_time=None, end_time=None):
    for week in g_l_weeks:
        start = time.time()
        fs_graph = nx.DiGraph()
        # add vertices is optional
        # fs_graph = add_vertices(fs_graph, week)
        fs_graph = extract_extract_quote_retweet_relations_by_time(fs_graph, week, start_time, end_time)
        print('%s Graph info:' % week)
        print(nx.info(fs_graph))
        output_graph(fs_graph, week, None)
        logging.debug('%s fsgraph is done in %s seconds.' % (week, str(time.time() - start)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    # main_by_month('201904')
