import json
import logging
import sqlite3
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils
import numpy as np
import scipy.spatial.distance as scipyd
import os
from gensim.models import KeyedVectors
import multiprocessing
import threading
import math
import sys
sys.path.insert(1, '/home/mf3jh/workspace/lib/lexvec/' + 'lexvec/python/lexvec/')
import model as lexvec
from sklearn.cluster import SpectralClustering
import time
import networkx as nx
import matplotlib.pyplot as plt


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
# g_lexvec_model_path = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.300d.W+C.pos.vectors'
g_lexvec_model_path = '/home/mf3jh/workspace/lib/lexvec/' + 'lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
g_time_int_idx_map_path = g_time_series_data_path_prefix + 'time_int_idx_map.json'
g_topic_sum_format = g_time_series_data_path_prefix + '{0}/{1}_topic_sum'
g_topic_vecs_format = g_time_series_data_path_prefix + '{0}/{1}_topic_vecs.json'
g_topic_pairwise_sim_path = g_time_series_data_path_prefix + 'topic_sims.json'
g_topic_clustering_path = g_time_series_data_path_prefix + 'topic_clusters.json'
g_topic_clustering_updated_path = g_time_series_data_path_prefix + 'topic_clusters_updated.json'
g_time_to_tc_path = g_time_series_data_path_prefix + 'time_to_tc.txt'
g_tc_sorted_path = g_time_series_data_path_prefix + 'tc_sorted.txt'
g_topic_comp_inter_folder_path = g_time_series_data_path_prefix + 'topic_comp_level2_inter_rets/'
g_topic_comp_inter_format = g_topic_comp_inter_folder_path + '{0}.json'
g_topic_cluster_n = 200
g_topic_sim_threshold = 0.65
g_topic_relax_sim_threshold = 0.65
g_topic_clustering_max_diameter = 3



def topic_to_vec(d_topic_words, word_vec_model):
    w_sum = np.zeros(300)
    for word in d_topic_words:
        try:
            # word_vec = word_vec_model.wv[word.lower()]
            word_vec = word_vec_model.word_rep(word.lower())
        except:
            continue
        w_sum += word_vec * d_topic_words[word]
    w_sum = w_sum / len(d_topic_words)
    return w_sum


def topics_to_vecs_for_one_time_int(time_int_str, word_vec_model):
    d_topics = dict()
    with open(g_topic_sum_format.format(time_int_str, time_int_str), 'r') as in_fd:
        one_topic_line = in_fd.readline()
        idx = 0
        while one_topic_line:
            topic_vec_str = ','.join(list([str(ele) for ele in topic_to_vec(json.loads(one_topic_line), word_vec_model)]))
            d_topics[time_int_str + '_' + str(idx)] = topic_vec_str
            idx += 1
            one_topic_line = in_fd.readline()
        in_fd.close()
    with open(g_topic_vecs_format.format(time_int_str, time_int_str), 'w+') as out_fd:
        json.dump(d_topics, out_fd)
        out_fd.close()


def topics_to_vecs_for_time_ints(l_time_ints, word_vec_model):
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        topics_to_vecs_for_one_time_int(time_int_str, word_vec_model)


def load_lexvec_model():
    # lexvec_model = KeyedVectors.load_word2vec_format(g_lexvec_model_path, binary=True)
    lexvec_model = lexvec.Model(g_lexvec_model_path)
    return lexvec_model


def topics_to_vecs_multithreads(l_time_ints, word_vec_model):
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
        t = threading.Thread(target=topics_to_vecs_for_time_ints, args=(l_each_batch, word_vec_model))
        t.setName('topic_to_vec_t' + str(t_id))
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

    logging.debug('All topic vectors have been written.')


def load_topic_sims():
    d_topic_sims = dict()
    l_inter_files = os.listdir(g_topic_comp_inter_folder_path)
    for inter_file in l_inter_files:
        with open(g_topic_comp_inter_folder_path + inter_file, 'r') as in_fd:
            d_topic_sims_batch = json.load(in_fd)
            d_topic_sims = dict(list(d_topic_sims.items()) + list(d_topic_sims_batch.items()))
            in_fd.close()
    return d_topic_sims


def build_aff_mat(l_topicids):
    # d_topic_sims = dict()
    # l_inter_files = os.listdir(g_topic_comp_inter_folder_path)
    # for inter_file in l_inter_files:
    #     with open(g_topic_comp_inter_folder_path + inter_file, 'r') as in_fd:
    #         d_topic_sims_batch = json.load(in_fd)
    #         d_topic_sims = dict(list(d_topic_sims.items()) + list(d_topic_sims_batch.items()))
    #         in_fd.close()
    d_topic_sims = load_topic_sims()

    # l_topicids = []
    # for topic_pair_str in d_topic_sims:
    #     topic_pair = topic_pair_str.split(':')
    #     topic_1 = topic_pair[0].strip()
    #     topic_2 = topic_pair[1].strip()
    #     if topic_1 not in l_topicids:
    #         l_topicids.append(topic_pair[0])
    #     if topic_2 not in l_topicids:
    #         l_topicids.append(topic_pair[1])

    aff_mat_dim = len(l_topicids)
    aff_mat = np.zeros([aff_mat_dim, aff_mat_dim], dtype=float)

    for topic_pair_str in d_topic_sims:
        topic_pair = topic_pair_str.split(':')
        topic_1 = topic_pair[0].strip()
        topic_2 = topic_pair[1].strip()
        topic_1_idx = l_topicids.index(topic_1)
        topic_2_idx = l_topicids.index(topic_2)
        if topic_1_idx == topic_2_idx:
            aff_mat[topic_1_idx][topic_2_idx] = 1.0
        else:
            if str(d_topic_sims[topic_pair_str]).lower() == 'nan':
                sim = 0.0
            else:
                sim = float(d_topic_sims[topic_pair_str])
            aff_mat[topic_1_idx][topic_2_idx] = sim
            aff_mat[topic_2_idx][topic_1_idx] = sim

    return aff_mat, l_topicids


def topic_comparisons_one_batch(l_batch, d_topic_vecs, tid):
    d_topic_comp = dict()
    logging.debug('%s comparisons for %s.' % (len(l_batch), tid))
    timer_start = time.time()
    count = 0
    for topic_pair in l_batch:
        if topic_pair[0] == topic_pair[1]:
            sim = 1.0
        else:
            try:
                topic_vec_1 = [float(ele.strip()) for ele in d_topic_vecs[topic_pair[0]].split(',')]
                topic_vec_2 = [float(ele.strip()) for ele in d_topic_vecs[topic_pair[1]].split(',')]
                sim = 1.0 - scipyd.cosine(topic_vec_1, topic_vec_2)
            except:
                sim = 0.0
            if str(sim).lower() == 'nan':
                sim = 0.0
        d_topic_comp[topic_pair[0] + ':' + topic_pair[1]] = sim
        count += 1
        if count % 1000 and count >= 1000:
            logging.debug('%s task done for %s in %s seconds.' % (count, tid, str(time.time()-timer_start)))
    logging.debug('%s task done for %s in %s seconds.' % (count, tid, str(time.time() - timer_start)))
    with open(g_topic_comp_inter_format.format(tid), 'w+') as out_fd:
        json.dump(d_topic_comp, out_fd)
        out_fd.close()


def get_topic_list_to_dim():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    d_topic_vecs = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        with open(g_topic_vecs_format.format(time_int_str, time_int_str), 'r') as in_fd:
            d_topic_vec_time_int = json.load(in_fd)
            d_topic_vecs = dict(list(d_topic_vecs.items()) + list(d_topic_vec_time_int.items()))
    l_topicid_to_dim = list(d_topic_vecs.keys())
    return l_topicid_to_dim


def get_topic_vecs():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    d_topic_vecs = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        with open(g_topic_vecs_format.format(time_int_str, time_int_str), 'r') as in_fd:
            d_topic_vec_time_int = json.load(in_fd)
            d_topic_vecs = dict(list(d_topic_vecs.items()) + list(d_topic_vec_time_int.items()))
    l_topicid_to_dim = list(d_topic_vecs.keys())
    return d_topic_vecs, l_topicid_to_dim


def topic_comparison_multithreads(d_topic_vecs, l_topicid_to_dim):
    # l_time_ints = data_preprocessing_utils.read_time_ints()
    # d_topic_vecs = dict()
    # for time_int in l_time_ints:
    #     time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
    #     with open(g_topic_vecs_format.format(time_int_str, time_int_str), 'r') as in_fd:
    #         d_topic_vec_time_int = json.load(in_fd)
    #         d_topic_vecs = dict(list(d_topic_vecs.items()) + list(d_topic_vec_time_int.items()))
    # l_topicid_to_dim = list(d_topic_vecs.keys())
    # l_topicid_to_dim = get_topic_list_to_dim()
    # d_topic_vecs, l_topicid_to_dim = get_topic_vecs()

    l_tasks = []
    for i in range(0, len(l_topicid_to_dim)-1):
        for j in range(i, len(l_topicid_to_dim)):
            l_tasks.append((l_topicid_to_dim[i], l_topicid_to_dim[j]))
    logging.debug('%s tasks in total.' % len(l_tasks))

    batch_size = math.ceil(len(l_tasks) / multiprocessing.cpu_count())
    l_batches = []
    for k in range(0, len(l_tasks), batch_size):
        if k + batch_size < len(l_tasks):
            l_batches.append(l_tasks[k:k + batch_size])
        else:
            l_batches.append(l_tasks[k:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_batches:
        t = threading.Thread(target=topic_comparisons_one_batch, args=(l_each_batch, d_topic_vecs, t_id))
        t.setName('topic_comp_t_' + str(t_id))
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

    logging.debug('All topic comparisons are done.')


# def topic_clustering(aff_mat, n_clusters, l_topicid_to_dim):
#     sc_labels = SpectralClustering(n_clusters=n_clusters,
#                             eigen_solver=None,
#                             random_state=None,
#                             n_init=10,
#                             affinity='precomputed',
#                             assign_labels='kmeans',
#                             n_jobs=-1).fit_predict(aff_mat)
#     d_clusters = dict()
#     if len(sc_labels) != len(l_topicid_to_dim):
#         raise Exception('Labels and topicids are not consistent.')
#     for idx, label in enumerate(sc_labels):
#         if label not in d_clusters:
#             d_clusters[label] = [l_topicid_to_dim[idx]]
#         else:
#             d_clusters[label].append(l_topicid_to_dim[idx])
#     with open(g_topic_clustering_path, 'w+') as out_fd:
#         json.dump(d_clusters, out_fd)
#         out_fd.close()
#     return sc_labels


def pred_cluster_size(p_graph, max_diameter):
    comp_graph_avg_deg = math.floor(sum(dict(p_graph.degree()).values()) / p_graph.number_of_nodes())
    pred_c_size = 1
    for i in range(max_diameter):
        pred_c_size += math.pow(comp_graph_avg_deg, i + 1)
    pred_c_size = 1 + comp_graph_avg_deg + math.pow(comp_graph_avg_deg, 2) + math.pow(comp_graph_avg_deg, 3)
    pred_n_clusters = int(math.ceil(p_graph.number_of_nodes() / float(pred_c_size)))
    if pred_n_clusters < 2:
        pred_n_clusters = 2
    #TODO
    # for bi-clustering
    # pred_n_clusters = 2
    return pred_n_clusters


def verify_phrase_clusters(p_clusters, max_diameter):
    good_mark = True
    for c in p_clusters.keys():
        if not isinstance(p_clusters[c], nx.Graph):
            raise Exception('[ERR]: Wrong cluster emerges!')
        sub_graph_c = p_clusters[c]
        if nx.diameter(sub_graph_c) > max_diameter:
            logging.debug('Cluster violates max_diameter condition!')
            logging.debug(p_clusters[c])
            good_mark = False
    if good_mark:
        logging.debug('Topic clusters are fine!')
    return good_mark


def topic_clustering(p_graph, aff_mat, init_n_clusters, max_diameter):
    logging.debug('Enter phrase_clustering: p_graph:')
    logging.debug('%s nodes.' % len(p_graph.nodes))
    #c_comps, comp_labels = connected_components(aff_mat)
    if not nx.is_connected(p_graph):
        raise Exception('[ERR]: p_graph is not connected!')
    d_clusters = dict()
    if nx.diameter(p_graph) <= max_diameter:
        if list(p_graph.nodes)[0] in d_clusters.keys():
            raise Exception('[ERR]: Overlapping clusters emerge!')
        d_clusters[list(p_graph.nodes)[0]] = p_graph
        logging.debug('A good cluster is done!')
        logging.debug(p_graph.nodes)
        return d_clusters

    start = time.time()
    p_labels = SpectralClustering(n_clusters=init_n_clusters, affinity='precomputed', n_jobs=-1).fit_predict(aff_mat)
    cand_clusters = dict()
    nodes = list(p_graph.nodes)
    for id, label in enumerate(p_labels):
        if label not in cand_clusters.keys():
            cand_clusters[label] = [nodes[id]]
        else:
            cand_clusters[label].append(nodes[id])
    logging.debug('SpectralClustering cost %s secs for n = %s' % (str(time.time()-start), init_n_clusters))

    # keep doing clustering for each cluster if this cluster's diameter is greater than max_diameter
    for c in cand_clusters.keys():
        sub_graph_c = p_graph.subgraph(cand_clusters[c])
        if not nx.is_connected(sub_graph_c):
            logging.debug('Disconnected resulting clusters emerge!')
            #print nx.info(sub_graph_c)
            #raise Exception('[ERR]: sub_graph_c is not connected!')
        l_sub_graph_comp = [sub_graph_c.subgraph(comp) for comp in nx.connected_components(sub_graph_c)]
        for sub_graph_comp_id, sub_graph_comp in enumerate(l_sub_graph_comp):
            if len(sub_graph_comp.nodes) <= 0:
                continue
            sub_graph_aff_mat = nx.adjacency_matrix(sub_graph_comp)
            n_clusters = pred_cluster_size(sub_graph_comp, g_topic_clustering_max_diameter)
            logging.debug('recursive phrase_clustering with n = %s' % n_clusters)
            d_c_clusters = topic_clustering(sub_graph_comp, sub_graph_aff_mat, n_clusters, max_diameter)
            d_clusters = d_clusters.copy()
            d_clusters.update(d_c_clusters)

    return d_clusters


def recursive_bi_spectral_clustering(d_topic_sims):
    topic_graph = nx.Graph()
    for topic_pair_str in d_topic_sims:
        if str(d_topic_sims[topic_pair_str]).lower() == 'nan':
            sim = 0.0
        else:
            sim = float(d_topic_sims[topic_pair_str])
        if sim < g_topic_sim_threshold:
            continue
        l_topics = topic_pair_str.split(':')
        topic_1 = l_topics[0].strip()
        topic_2 = l_topics[1].strip()
        topic_graph.add_edge(topic_1, topic_2, weight=sim)

    l_comp_graphs = [topic_graph.subgraph(comp) for comp in nx.connected_components(topic_graph)]
    total_comp = len(l_comp_graphs)
    logging.debug('components size = %s' % total_comp)
    logging.debug([len(c) for c in sorted(l_comp_graphs, key=len, reverse=True)])

    d_final_clusters = dict()

    start = time.time()
    for comp_id, comp_graph in enumerate(l_comp_graphs):
        if len(comp_graph.nodes) <= 0:
            continue
        aff_mat = nx.adjacency_matrix(comp_graph)
        # we use the first phrase as the key of a cluster
        # aff_mat = build_aff_mat(comp_graph)
        # comp_graph_avg_deg = math.floor(sum(dict(comp_graph.degree()).values())/comp_graph.number_of_nodes())
        # pred_c_size = 1
        # for i in range(MAX_DIAMETER):
        #     pred_c_size += math.pow(comp_graph_avg_deg, i+1)
        # pred_c_size = 1 + comp_graph_avg_deg + math.pow(comp_graph_avg_deg, 2) + math.pow(comp_graph_avg_deg, 3)
        # n_init_clusters = int(math.ceil(comp_graph.number_of_nodes()/float(pred_c_size)))
        # n_init_clusters = 2
        n_init_clusters  = pred_cluster_size(comp_graph, g_topic_clustering_max_diameter)
        logging.debug('Initial phrase_clustering with n = %s' % n_init_clusters)
        d_ret_clusters = topic_clustering(comp_graph, aff_mat, n_init_clusters, g_topic_clustering_max_diameter)
        d_final_clusters = d_final_clusters.copy()
        d_final_clusters.update(d_ret_clusters)
        logging.debug('%s rbsc is done in %s secs.' % (float(comp_id+1)/total_comp, str(time.time()-start)))
    if not verify_phrase_clusters(d_final_clusters, g_topic_clustering_max_diameter):
        logging.debug('Resulting clusters are not valid!')
    return d_final_clusters


def load_topic_clusters():
    with open(g_topic_clustering_updated_path, 'r') as in_fd:
        d_topic_clusters = json.load(in_fd)
        in_fd.close()
    return d_topic_clusters


def get_time_to_topic_clusters():
    d_topic_clusters = load_topic_clusters()
    l_sorted_tcs = sorted(d_topic_clusters.keys(), key=lambda k: len(d_topic_clusters[k]), reverse=True)
    d_time_to_tcs = dict()
    for tc in d_topic_clusters:
        # if len(d_topic_clusters[tc]) <= 5:
        #     continue
        for topic in d_topic_clusters[tc]:
            time_int = topic[:17]
            if time_int not in d_time_to_tcs:
                d_time_to_tcs[time_int] = [tc]
            else:
                d_time_to_tcs[time_int].append(tc)
    with open(g_time_to_tc_path, 'w+') as out_fd:
        for time_int in d_time_to_tcs:
            for tc in d_time_to_tcs[time_int]:
                out_fd.write(time_int + ':' + tc)
                out_fd.write('\n')
        out_fd.close()
    with open(g_tc_sorted_path, 'w+') as out_fd:
        for tc in l_sorted_tcs:
            out_fd.write(tc)
            out_fd.write('\n')
        out_fd.close()


def draw_time_to_topic_clusters():
    d_time_to_tc = dict()
    l_tcs = []
    with open(g_tc_sorted_path, 'r') as in_fd:
        tc = in_fd.readline()
        while tc:
            l_tcs.append(tc.strip())
            tc = in_fd.readline()
        in_fd.close()

    with open(g_time_to_tc_path, 'r') as in_fd:
        line = in_fd.readline()
        while line:
            l_fields = line.split(':')
            time_int = l_fields[0]
            tc = l_fields[1]
            if time_int not in d_time_to_tc:
                d_time_to_tc[time_int] = [tc.strip()]
            else:
                d_time_to_tc[time_int].append(tc.strip())
            # if tc not in l_tcs:
            #     l_tcs.append(tc.strip())
            line = in_fd.readline()
        in_fd.close()
    # l_tcs = sorted(list(set(l_tcs)))
    l_time_ints = sorted(list(d_time_to_tc.keys()))
    mat = np.zeros((len(l_tcs), len(l_time_ints)))
    for time_int in d_time_to_tc:
        idx_time = l_time_ints.index(time_int)
        for tc in d_time_to_tc[time_int]:
            idx_tc = l_tcs.index(tc)
            mat[idx_tc][idx_time] = 1.0

    l_xticks = [i for i in range(0, len(l_time_ints), 10)]
    l_time_ints_samples = [l_time_ints[i] for i in range(0, len(l_time_ints), 10)]
    l_yticks = [i for i in range(0, len(l_tcs), 100)]
    l_tcs_samples = [l_tcs[i] for i in range(0, len(l_tcs), 100)]

    # plt.imshow(ret_mat, cmap='gray', vmin=min, vmax=max)
    # plt.figure(figsize=(500, 500))
    plt.imshow(mat)
    plt.colorbar()
    plt.xticks(l_xticks)
    plt.yticks(l_xticks, l_tcs_samples)
    plt.show()



def load_topic_vecs():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    d_topic_vecs = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        with open(g_topic_vecs_format.format(time_int_str, time_int_str), 'r') as in_fd:
            topic_vecs_json = json.load(in_fd)
            in_fd.close()
        for topic in topic_vecs_json:
            d_topic_vecs[topic] = [float(ele.strip()) for ele in topic_vecs_json[topic].split(',')]
    return d_topic_vecs


def relax_topic_clustering(level_str):
    d_topic_clusters = load_topic_clusters()
    # d_topic_sims = load_topic_sims()
    d_topic_vecs = load_topic_vecs()

    d_node_vecs = dict()
    for cid in d_topic_clusters:
        c_vec = np.zeros(300)
        for topicid in d_topic_clusters[cid]:
            c_vec += np.asarray(d_topic_vecs[topicid])
        c_vec = c_vec / len(d_topic_clusters[cid])
        d_node_vecs[level_str + str(cid)] = ','.join([str(ele) for ele in list(c_vec)])

    l_node_to_dim = list(d_node_vecs.keys())
    return d_node_vecs, l_node_to_dim



def main():
    # word_vec_model = load_lexvec_model()
    # l_time_ints = data_preprocessing_utils.read_time_ints()
    # topics_to_vecs_multithreads(l_time_ints, word_vec_model)
    #d_topic_vecs, l_topicid_to_dim = get_topic_vecs()
    # d_topic_vecs, l_topicid_to_dim = relax_topic_clustering('level2_')
    # topic_comparison_multithreads(d_topic_vecs, l_topicid_to_dim)

    # d_topic_sims = load_topic_sims()
    # d_final_clusters = recursive_bi_spectral_clustering(d_topic_sims)
    # d_cluster_members = dict()
    # idx = 0
    # for key in d_final_clusters:
    #     d_cluster_members[idx] = list(d_final_clusters[key].nodes)
    #     idx += 1
    # with open(g_topic_clustering_path, 'w+') as out_fd:
    #     json.dump(d_cluster_members, out_fd)
    #     out_fd.close()

    load_topic_clusters()
    get_time_to_topic_clusters()
    draw_time_to_topic_clusters()

    # load_topic_sims()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()

