import igraph as ig
import louvain
import logging
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils
import math
import multiprocessing
import threading
import json
from os import path

g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_fsgraph_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph.gml'
g_fsgraph_community_img_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph_com.pdf'
g_fsgraph_community_cluster_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph_com_cluster.json'
g_fsgraph_fine_community_cluster_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph_fine_com_cluster_50.json'
g_fine_cluster_size = 200
g_modularity_threshold = 0.5


def recursive_communities_for_one_fsgraph(fs_ig):
    if len(fs_ig.vs) <= g_fine_cluster_size:
        return [[v['label'] for v in fs_ig.vs]]
    l_clusters = []
    coms = louvain.find_partition(fs_ig, louvain.ModularityVertexPartition)
    if coms.modularity < g_modularity_threshold:
        return [[v['label'] for v in fs_ig.vs]]
    for member in coms:
        if len(member) > g_fine_cluster_size:
            fs_sub = fs_ig.subgraph(member)
            l_sub_clusters = recursive_communities_for_one_fsgraph(fs_sub)
            l_clusters += l_sub_clusters
        else:
            l_clusters.append([fs_ig.vs[i]['label'] for i in member])
    return l_clusters


def finer_communities_for_one_fsgraph(time_int_str):
    fs_ig = ig.load(g_fsgraph_format.format(time_int_str, time_int_str))
    with open(g_fsgraph_community_cluster_format.format(time_int_str, time_int_str), 'r') as in_fd:
        fs_coarse_cluster_json = json.load(in_fd)
        in_fd.close()
    l_fs_coarse_cluster_member = []
    for v in fs_ig.vs:
        l_fs_coarse_cluster_member.append(fs_coarse_cluster_json[v['label']])
    fs_coarse_cluster = ig.VertexClustering(fs_ig, l_fs_coarse_cluster_member)
    d_uid_to_member = fs_coarse_cluster_json
    # l_av_cluster_label = []
    if fs_coarse_cluster.modularity >= g_modularity_threshold and len(fs_ig.vs) > g_fine_cluster_size:
        l_clusters = []
        for idx, member in enumerate(fs_coarse_cluster):
            l_clusters += recursive_communities_for_one_fsgraph(fs_ig.subgraph(member))
        d_uid_to_member = dict()
        for idx, cluster in enumerate(l_clusters):
            for uid in cluster:
                d_uid_to_member[uid] = idx
    with open(g_fsgraph_fine_community_cluster_format.format(time_int_str, time_int_str), 'w+') as out_fd:
        json.dump(d_uid_to_member, out_fd)
        out_fd.close()


def finer_communities_for_fsgraphs(l_time_ints, t_id):
    # d_communities = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        communities = finer_communities_for_one_fsgraph(time_int_str)
        # d_communities[time_int_str] = communities
    logging.debug('%s: fine communities are done.' % t_id)


def communities_for_one_fsgraph(time_int_str):
    logging.debug('Working on %s communities.' % time_int_str)
    # time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
    fs_ig = ig.load(g_fsgraph_format.format(time_int_str, time_int_str))
    communities = louvain.find_partition(fs_ig, louvain.ModularityVertexPartition)
    d_cluster = dict()
    for vid, member in enumerate(communities.membership):
        uid = fs_ig.vs[vid]['label']
        d_cluster[uid] = member
    if len(d_cluster) != len(fs_ig.vs):
        raise Exception('Incorrect clustering occurs!')
    with open(g_fsgraph_community_cluster_format.format(time_int_str, time_int_str), 'w+') as out_fd:
        json.dump(d_cluster, out_fd)
        out_fd.close()
    logging.debug('%s communities have been written.' % time_int_str)
    return communities


def communities_for_fsgraphs(l_time_ints, t_id):
    d_communities = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        communities = communities_for_one_fsgraph(time_int_str)
        d_communities[time_int_str] = communities
    logging.debug('%s communities are done.' % t_id)
    # for time_int_str in d_communities:
    #     ig.plot(d_communities[time_int_str], g_fsgraph_community_img_format.format(time_int_str, time_int_str),
    #             bbox=(7680, 7680), vertex_label_size=0)
    # logging.debug('%s community images are done.' % t_id)


def communities_multithreads(l_time_ints):
    l_time_ints_av = []
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        if not path.exists(g_fsgraph_fine_community_cluster_format.format(time_int_str, time_int_str)):
            l_time_ints_av.append(time_int)

    batch_size = math.ceil(len(l_time_ints_av) / multiprocessing.cpu_count())
    l_l_time_ints = []
    for i in range(0, len(l_time_ints_av), batch_size):
        if i + batch_size < len(l_time_ints_av):
            l_l_time_ints.append(l_time_ints_av[i:i + batch_size])
        else:
            l_l_time_ints.append(l_time_ints_av[i:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_time_ints:
        t = threading.Thread(target=communities_for_fsgraphs, args=(l_each_batch, t_id))
        t.setName('com_det_' + str(t_id))
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

    logging.debug('All communities are done.')


def fine_communities_multithreads(l_time_ints):
    l_time_ints_av = []
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        # if not path.exists(g_fsgraph_community_img_format.format(time_int_str, time_int_str)) or \
        #         not path.exists(g_fsgraph_community_cluster_format.format(time_int_str, time_int_str)):
        l_time_ints_av.append(time_int)

    batch_size = math.ceil(len(l_time_ints_av) / multiprocessing.cpu_count())
    l_l_time_ints = []
    for i in range(0, len(l_time_ints_av), batch_size):
        if i + batch_size < len(l_time_ints_av):
            l_l_time_ints.append(l_time_ints_av[i:i + batch_size])
        else:
            l_l_time_ints.append(l_time_ints_av[i:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_time_ints:
        t = threading.Thread(target=finer_communities_for_fsgraphs, args=(l_each_batch, t_id))
        t.setName('fine_com_det_' + str(t_id))
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

    logging.debug('All communities are done.')


def main():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    # communities_multithreads(l_time_ints)
    fine_communities_multithreads(l_time_ints)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    # communities_for_one_fsgraph('20180413_20180419')
    # finer_communities_for_one_fsgraph('20180404_20180410')
