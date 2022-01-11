import logging
import json
import networkx as nx
# import igraph as ig
import time
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_time_series_data_format = g_time_series_data_path_prefix + '{0}/{1}_nec_data'
g_time_format = '%Y%m%d'
g_fsgraph_output_gml_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph.gml'


# edge: (src, trg, {r: int, t: int, q:int, a:int, weight:real(normalized)})
# 'weight' will not be handled in this function.
def gen_one_fsgraph(time_int_str):
    fsgraph = nx.DiGraph()
    with open(g_time_series_data_format.format(time_int_str, time_int_str), 'r') as in_fd:
        t_nec_str = in_fd.readline()
        while t_nec_str:
            t_nec_json = json.loads(t_nec_str)
            src = t_nec_json['src']
            trg = t_nec_json['uid']
            t_type = t_nec_json['type']
            if t_type != 'n':
                if trg is None or trg == '' or src is None or src == '':
                    logging.error('The following tweet necessary data has trouble with uid or src: %s' % t_nec_str)
                else:
                    if not fsgraph.has_edge(src, trg):
                        if t_type == 'r':
                            fsgraph.add_edge(src, trg, r=1, t=0, q=0, a=1, weight=0.0)
                        elif t_type == 't':
                            fsgraph.add_edge(src, trg, r=0, t=1, q=0, a=1, weight=0.0)
                        elif t_type == 'q':
                            fsgraph.add_edge(src, trg, r=0, t=0, q=1, a=1, weight=0.0)
                        else:
                            logging.error('Undefined tweet type occurs: %s.' % t_nec_str)
                    else:
                        if t_type == 'r':
                            fsgraph[src][trg]['r'] += 1
                            fsgraph[src][trg]['a'] += 1
                        elif t_type == 't':
                            fsgraph[src][trg]['t'] += 1
                            fsgraph[src][trg]['a'] += 1
                        elif t_type == 'q':
                            fsgraph[src][trg]['q'] += 1
                            fsgraph[src][trg]['a'] += 1
                        else:
                            logging.error('Undefined tweet type occurs: %s.' % t_nec_str)
            t_nec_str = in_fd.readline()
        in_fd.close()
    return fsgraph


def compute_weights(fsgraph, time_int_str):
    l_counts = [edge[2] for edge in fsgraph.edges(data='a')]
    max_count = max(l_counts)
    min_count = min(l_counts)
    if max_count - min_count == 0:
        logging.debug('fsgraph for %s has all equal counts, which may be incorrect.' % time_int_str)
        return fsgraph
    for src, trg in fsgraph.edges:
        fsgraph[src][trg]['weight'] = float(fsgraph[src][trg]['a'] - min_count) / (max_count - min_count)
    return fsgraph


def dump_fsgraphs(d_fsgraphs):
    for time_int_str in d_fsgraphs:
        nx.write_gml(d_fsgraphs[time_int_str], g_fsgraph_output_gml_format.format(time_int_str, time_int_str))
    logging.debug('fsgraphs dump is done.')


def gen_fsgraphs(l_time_ints):
    d_fsgraphs = dict()
    timer_start = time.time()
    for time_int in l_time_ints:
        time_int_str = time_int[0].strftime(g_time_format) + '_' + time_int[1].strftime(g_time_format)
        fsgraph = gen_one_fsgraph(time_int_str)
        d_fsgraphs[time_int_str] = compute_weights(fsgraph, time_int_str)
        logging.debug('The fsgraph for %s is done in %s seconds.' % (time_int_str, str(time.time()-timer_start)))
    return d_fsgraphs


def extract_influential_users_from_one_fsgraph(time_int_str, top_num):
    fsgraph = nx.read_gml(g_fsgraph_output_gml_format.format(time_int_str, time_int_str))
    l_degrees = list(fsgraph.out_degree(fsgraph.nodes, 'a'))
    l_degrees = sorted(l_degrees, key=lambda d: d[1], reverse=True)
    l_top_users = [d[0] for d in l_degrees][:top_num]
    top_sub_fsgraph = nx.subgraph(fsgraph, l_top_users)
    return l_top_users, top_sub_fsgraph
    # print(l_top_users)


def main():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    d_fsgraphs = gen_fsgraphs(l_time_ints)
    dump_fsgraphs(d_fsgraphs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    # extract_influential_users_from_one_fsgraph('20180718_20180724', 50)