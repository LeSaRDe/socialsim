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
import scipy.spatial.distance as scipyd



g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_topic_weights_by_uid_format = g_time_series_data_path_prefix + '{0}/{1}_topic_weights_by_uid.json'
g_topic_clustering_path = g_time_series_data_path_prefix + 'topic_clusters.json'
g_topic_clustering_updated_path = g_time_series_data_path_prefix + 'topic_clusters_updated.json'
g_topic_embed_inter_rets_path = g_time_series_data_path_prefix + 'topic_embed_inter_rets/'
g_topic_embed_inter_rets_format = g_topic_embed_inter_rets_path + '{0}.json'
g_topic_embed_comp_inter_rets_path = g_time_series_data_path_prefix + 'topic_embed_comp_inter_rets/'
g_topic_embed_comp_inter_rets_format = g_topic_embed_comp_inter_rets_path + '{0}.json'
g_topic_vecs_format = g_time_series_data_path_prefix + '{0}/{1}_topic_vecs.json'
g_influential_users_path = g_time_series_data_path_prefix + 'influential_users.json'
g_time_series_data_db_format = g_time_series_data_path_prefix + '{0}/{1}_nec_data.db'
g_inf_vs_usr_top_vec_inter_rets_path = g_time_series_data_path_prefix + 'inf_vs_usr_top_vec_inter_rets/'
g_inf_vs_usr_top_vec_inter_rets_inf_format = g_inf_vs_usr_top_vec_inter_rets_path + 'inf_{0}.json'
g_inf_vs_usr_top_vec_inter_rets_usr_format = g_inf_vs_usr_top_vec_inter_rets_path + 'usr_{0}.json'
g_topic_classification_threshold = 0.65


def load_topic_embeddings():
    d_time_topic_embeds = dict()
    l_inter_files = os.listdir(g_topic_embed_inter_rets_path)
    for inter_file in l_inter_files:
        with open(g_topic_embed_inter_rets_path + inter_file, 'r') as in_fd:
            d_json = json.load(in_fd)
            d_time_topic_embeds = dict(list(d_time_topic_embeds.items()) + list(d_json.items()))
            in_fd.close()
    return d_time_topic_embeds


def topic_embed_comp_for_two_time_ints(time_int_str_1, time_int_str_2, d_time_topic_embeds):
    l_tc_dims = list(d_time_topic_embeds[time_int_str_1].keys())
    l_t_emb_1 = []
    l_t_emb_2 = []
    for tc_dim in l_tc_dims:
        l_t_emb_1.append(d_time_topic_embeds[time_int_str_1][tc_dim])
        l_t_emb_2.append(d_time_topic_embeds[time_int_str_2][tc_dim])
    try:
        sim = 1.0 - scipyd.cosine(l_t_emb_1, l_t_emb_2)
    except:
        sim = 0.0
    if str(sim).lower() == 'nan':
        sim = 0.0
    return sim


def topic_embed_comps_for_time_int_batch(l_batch, d_time_topic_embeds, tid):
    d_topic_embed_rets = dict()
    count = 0
    for time_int_pair in l_batch:
        time_int_str_1 = data_preprocessing_utils.time_int_to_time_int_str(time_int_pair[0])
        time_int_str_2 = data_preprocessing_utils.time_int_to_time_int_str(time_int_pair[1])
        sim = topic_embed_comp_for_two_time_ints(time_int_str_1, time_int_str_2, d_time_topic_embeds)
        d_topic_embed_rets[time_int_str_1 + ':' + time_int_str_2] = sim
        count += 1
        if count % 5000 and count >= 5000:
            logging.debug('%s compairsons have done at %s.' % (count, tid))
    logging.debug('%s compairsons have done at %s.' % (count, tid))
    with open(g_topic_embed_comp_inter_rets_format.format(tid), 'w+') as out_fd:
        json.dump(d_topic_embed_rets, out_fd)
        out_fd.close()
    logging.debug('%s topic compairsons are done.' % tid)


def topic_embed_comps_multithreads(l_time_ints, d_time_topic_embeds):
    l_tasks = []
    for i in range(0, len(l_time_ints) - 1):
        for j in range(i, len(l_time_ints)):
            l_tasks.append((l_time_ints[i], l_time_ints[j]))
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
        t = threading.Thread(target=topic_embed_comps_for_time_int_batch,
                             args=(l_each_batch, d_time_topic_embeds, t_id))
        t.setName('topic_embed_comp_t_' + str(t_id))
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

    logging.debug('All topic embedding comparisons are done.')



def topic_classification(topic_id, d_topic_vecs, d_topic_cluster_vecs):
    topic_vec = d_topic_vecs[topic_id]
    l_rets = []
    for tc in d_topic_cluster_vecs:
        sim = 1.0 - scipyd.cosine(topic_vec, d_topic_cluster_vecs[tc])
        l_rets.append((tc, sim))
    l_sorted_rets = sorted(l_rets, key=lambda k: k[1], reverse=True)
    if l_sorted_rets[0][1] >= g_topic_classification_threshold:
        return l_sorted_rets[0][0]
    else:
        return 'tc_' + str(len(d_topic_cluster_vecs))


def topic_cluster_vecs(d_topic_vecs, d_topic_clusters):
    d_topic_cluster_vecs = dict()
    for tc in d_topic_clusters:
        topic_cluster_vec = np.zeros(300)
        for topic_id in d_topic_clusters[tc]:
            topic_cluster_vec += d_topic_vecs[topic_id]
        d_topic_cluster_vecs[tc] = topic_cluster_vec / len(d_topic_clusters[tc])
    return d_topic_cluster_vecs


def load_topic_vecs():
    l_time_ints = data_preprocessing_utils.read_time_ints()
    d_topic_vecs = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        with open(g_topic_vecs_format.format(time_int_str, time_int_str), 'r') as in_fd:
            topic_vecs_json = json.load(in_fd)
            in_fd.close()
        for topic in topic_vecs_json:
            d_topic_vecs[topic] = np.asarray([float(ele.strip()) for ele in topic_vecs_json[topic].split(',')])
    return d_topic_vecs


def load_topic_clusters():
    with open(g_topic_clustering_updated_path, 'r') as in_fd:
        d_topic_clusters = json.load(in_fd)
        in_fd.close()
    return d_topic_clusters


def get_topic_cluster_memberships(d_topic_clusters):
    d_topic_members = dict()
    l_tcs = []
    for tc in d_topic_clusters:
        for topic in d_topic_clusters[tc]:
            tc_id = tc
            d_topic_members[topic] = tc_id
            if tc_id not in l_tcs:
                l_tcs.append(tc_id)
    l_tcs = sorted(l_tcs)
    return d_topic_members, l_tcs


def topic_embedding_for_one_uid(time_int_str, d_topic_weights, d_topic_members, l_tcs):
    d_topic_embed_uid = {tc_id : 0.0 for tc_id in l_tcs}
    for topic in d_topic_weights:
        topic_id = time_int_str + '_' + str(topic)
        tc_member = d_topic_members[topic_id]
        d_topic_embed_uid[tc_member] += d_topic_weights[topic]
    return d_topic_embed_uid


def topic_embedding_for_one_time_interval(time_int_str, d_topic_members, l_tcs):
    d_topic_embed_time_int = {tc_id : 0.0 for tc_id in l_tcs}
    with open(g_topic_weights_by_uid_format.format(time_int_str, time_int_str), 'r') as in_fd:
        d_uid_topic_weights = json.load(in_fd)
        in_fd.close()
    # d_uid_topic_embeds = dict()
    for uid in d_uid_topic_weights:
        d_topic_embed = topic_embedding_for_one_uid(time_int_str, d_uid_topic_weights[uid], d_topic_members, l_tcs)
        for tc_id in d_topic_embed:
            d_topic_embed_time_int[tc_id] += d_topic_embed[tc_id]
    logging.debug('%s topic embedding is done.' % time_int_str)
    return d_topic_embed_time_int


def topic_embeddings_for_multiple_time_intervals(l_time_ints, d_topic_members, l_tcs, tid):
    d_time_ints_topic_embeds = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        d_topic_embed = topic_embedding_for_one_time_interval(time_int_str, d_topic_members, l_tcs)
        d_time_ints_topic_embeds[time_int_str] = d_topic_embed
    with open(g_topic_embed_inter_rets_format.format(tid), 'w+') as out_fd:
        json.dump(d_time_ints_topic_embeds, out_fd)
        out_fd.close()
    logging.debug('%s topic embeddings are done.' % tid)


def topic_embeddings_multithreads(l_time_ints, d_topic_members, l_tcs):
    batch_size = math.ceil(len(l_time_ints) / multiprocessing.cpu_count())
    l_batches = []
    for k in range(0, len(l_time_ints), batch_size):
        if k + batch_size < len(l_time_ints):
            l_batches.append(l_time_ints[k:k + batch_size])
        else:
            l_batches.append(l_time_ints[k:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_batches:
        t = threading.Thread(target=topic_embeddings_for_multiple_time_intervals,
                             args=(l_each_batch, d_topic_members, l_tcs, t_id))
        t.setName('topic_embed_t_' + str(t_id))
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

    logging.debug('All topic embeddings are done.')


def topic_clusters_update():
    l_tbd_topic_ids = []
    l_del_tcs = []
    d_topic_clusters = load_topic_clusters()
    for tc in d_topic_clusters:
        if len(d_topic_clusters[tc]) < 5:
            l_tbd_topic_ids += d_topic_clusters[tc]
            l_del_tcs.append(tc)
    for tc in l_del_tcs:
        del d_topic_clusters[tc]
    d_new_topic_clusters = dict()
    tc_idx = 0
    for tc in d_topic_clusters:
        d_new_topic_clusters['tc_' + str(tc_idx)] = d_topic_clusters[tc]
        tc_idx += 1
    logging.debug('%s clusters are left.' % len(d_new_topic_clusters))
    d_topic_vecs = load_topic_vecs()
    d_topic_cluster_vecs = topic_cluster_vecs(d_topic_vecs, d_new_topic_clusters)
    for topic_id in l_tbd_topic_ids:
        topic_c = topic_classification(topic_id, d_topic_vecs, d_topic_cluster_vecs)
        topic_vec = d_topic_vecs[topic_id]
        if topic_c in d_new_topic_clusters:
            d_new_topic_clusters[topic_c].append(topic_id)
            d_topic_cluster_vecs[topic_c] = (d_topic_cluster_vecs[topic_c] + topic_vec) / 2
        else:
            d_new_topic_clusters[topic_c] = [topic_id]
            d_topic_cluster_vecs[topic_c] = topic_vec
    with open(g_topic_clustering_updated_path, 'w+') as out_fd:
        json.dump(d_new_topic_clusters, out_fd)
        out_fd.close()


def draw_topic_embed_comp_rets(l_sorted_time_int_strs):
    d_topic_embed_comps = dict()
    ret_size = len(l_sorted_time_int_strs)
    ret_mat = np.zeros((ret_size, ret_size))
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
        ret_mat[idx_1][idx_2] = d_topic_embed_comps[time_int_pair]
        ret_mat[idx_2][idx_1] = d_topic_embed_comps[time_int_pair]
    for i in range(0, ret_mat.shape[0]):
        ret_mat[i][i] = 1.0

    l_ticks = [i for i in range(0, ret_size, 10)]
    # plt.imshow(ret_mat, cmap='gray', vmin=min, vmax=max)
    plt.imshow(ret_mat)
    plt.colorbar()
    plt.xticks(l_ticks)
    plt.yticks(l_ticks)
    plt.show()


def load_influentials():
    with open(g_influential_users_path, 'r') as in_fd:
        d_influentials = json.load(in_fd)
        in_fd.close()
    return d_influentials


def one_uid_topic_vec(uid, time_int_str, d_time_int_topic_vecs):
    with open(g_topic_weights_by_uid_format.format(time_int_str, time_int_str), 'r') as in_fd:
        d_topic_weights = json.load(in_fd)
        in_fd.close()
    wavg_topic_vec = np.zeros(300)
    for topic_id in d_time_int_topic_vecs:
        topic_vec = np.asarray([float(ele.strip()) for ele in d_time_int_topic_vecs[topic_id].split(',')])
        if uid in d_topic_weights:
            topic_weight = d_topic_weights[uid][topic_id[18:]]
        else:
            continue
        topic_vec = topic_vec * topic_weight
        wavg_topic_vec += topic_vec
    return wavg_topic_vec


def influentials_and_users_topic_vecs(l_time_ints, d_influentials, tid):
    d_time_int_inf_top_vecs = dict()
    d_time_int_usr_top_vecs = dict()
    logging.debug('%s starts...' % tid)
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        l_time_int_influentials = d_influentials[time_int_str]
        with open(g_topic_vecs_format.format(time_int_str, time_int_str), 'r') as in_fd:
            d_time_int_topic_vecs = json.load(in_fd)
            in_fd.close()

        db_conn = sqlite3.connect(g_time_series_data_db_format.format(time_int_str, time_int_str))
        db_cur = db_conn.cursor()
        sql_str = '''SELECT uid from wh_nec_data'''
        db_cur.execute(sql_str)
        l_users = [row[0] for row in db_cur.fetchall()]
        db_conn.close()

        avg_inf_top_vec = np.zeros(300)
        for uid in l_time_int_influentials:
            inf_top_vec = np.asarray(one_uid_topic_vec(uid, time_int_str, d_time_int_topic_vecs))
            avg_inf_top_vec += inf_top_vec
        avg_inf_top_vec = avg_inf_top_vec / len(l_time_int_influentials)
        d_time_int_inf_top_vecs[time_int_str] = ','.join([str(ele) for ele in list(avg_inf_top_vec)])

        avg_usr_top_vec = np.zeros(300)
        for uid in l_users:
            usr_top_vec = np.asarray(one_uid_topic_vec(uid, time_int_str, d_time_int_topic_vecs))
            avg_usr_top_vec += usr_top_vec
        avg_usr_top_vec = avg_usr_top_vec / len(l_users)
        d_time_int_usr_top_vecs[time_int_str] = ','.join([str(ele) for ele in list(avg_usr_top_vec)])
        logging.debug('%s is done.' % time_int_str)

    with open(g_inf_vs_usr_top_vec_inter_rets_inf_format.format(tid), 'w+') as out_fd:
        json.dump(d_time_int_inf_top_vecs, out_fd)
        out_fd.close()
    with open(g_inf_vs_usr_top_vec_inter_rets_usr_format.format(tid), 'w+') as out_fd:
        json.dump(d_time_int_usr_top_vecs, out_fd)
        out_fd.close()

    logging.debug('%s completes.' % tid)
    return d_time_int_inf_top_vecs, d_time_int_usr_top_vecs


def influentials_and_users_topic_vecs_multithreads(l_time_ints, d_influentials):
    batch_size = math.ceil(len(l_time_ints) / multiprocessing.cpu_count())
    l_batches = []
    for k in range(0, len(l_time_ints), batch_size):
        if k + batch_size < len(l_time_ints):
            l_batches.append(l_time_ints[k:k + batch_size])
        else:
            l_batches.append(l_time_ints[k:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_batches:
        t = threading.Thread(target=influentials_and_users_topic_vecs,
                             args=(l_each_batch, d_influentials, t_id))
        t.setName('inf_vs_usr_t_' + str(t_id))
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

    logging.debug('All inf and usr top vecs are done.')



def time_int_inf_vs_usr_top_vec_cos(d_time_int_inf_top_vecs, d_time_int_usr_top_vecs):
    l_time_ints = data_preprocessing_utils.read_time_ints()
    l_sorted_time_int_strs = sorted([data_preprocessing_utils.time_int_to_time_int_str(time_int)
                                     for time_int in l_time_ints])
    l_sorted_time_int_strs = [time_int for time_int in l_sorted_time_int_strs if time_int in d_time_int_inf_top_vecs.keys()]
    l_rets = np.zeros(len(l_sorted_time_int_strs))
    for time_int in d_time_int_inf_top_vecs:
        sim = 1.0 - scipyd.cosine(d_time_int_inf_top_vecs[time_int], d_time_int_usr_top_vecs[time_int])
        l_rets[l_sorted_time_int_strs.index(time_int)] = sim

    l_ticks = [i for i in range(0, len(d_time_int_inf_top_vecs), 5)]
    # plt.imshow(ret_mat, cmap='gray', vmin=min, vmax=max)
    # plt.imshow(l_rets)
    # plt.colorbar()
    plt.plot(l_rets)
    plt.xticks(l_ticks, l_sorted_time_int_strs, rotation='vertical')
    # plt.yticks(l_ticks)
    plt.show()


def load_inf_and_usr_top_vecs():
    d_time_int_inf_top_vecs = dict()
    d_time_int_usr_top_vecs = dict()

    l_inter_files = os.listdir(g_inf_vs_usr_top_vec_inter_rets_path)
    for inter_file in l_inter_files:
        if inter_file[:3] == 'inf':
            with open(g_inf_vs_usr_top_vec_inter_rets_path + inter_file, 'r') as in_fd:
                inf_top_vecs = json.load(in_fd)
                d_time_int_inf_top_vecs = dict(list(d_time_int_inf_top_vecs.items()) + list(inf_top_vecs.items()))
                in_fd.close()
        elif inter_file[:3] == 'usr':
            with open(g_inf_vs_usr_top_vec_inter_rets_path + inter_file, 'r') as in_fd:
                inf_top_vecs = json.load(in_fd)
                d_time_int_usr_top_vecs = dict(list(d_time_int_usr_top_vecs.items()) + list(inf_top_vecs.items()))
                in_fd.close()

    for time_int in d_time_int_inf_top_vecs:
        d_time_int_inf_top_vecs[time_int] = np.asarray(
            [float(ele.strip()) for ele in d_time_int_inf_top_vecs[time_int].split(',')])
    for time_int in d_time_int_usr_top_vecs:
        d_time_int_usr_top_vecs[time_int] = np.asarray(
            [float(ele.strip()) for ele in d_time_int_usr_top_vecs[time_int].split(',')])

    return d_time_int_inf_top_vecs, d_time_int_usr_top_vecs



def main():
    # l_time_ints = data_preprocessing_utils.read_time_ints()
    # d_topic_clusters = load_topic_clusters()
    # d_topic_members, l_tcs = get_topic_cluster_memberships(d_topic_clusters)
    # topic_embeddings_multithreads(l_time_ints, d_topic_members, l_tcs)

    # d_time_topic_embeds = load_topic_embeddings()
    # topic_embed_comps_multithreads(l_time_ints, d_time_topic_embeds)

    # l_sorted_time_int_strs = sorted([data_preprocessing_utils.time_int_to_time_int_str(time_int)
    #                                  for time_int in l_time_ints])
    # draw_topic_embed_comp_rets(l_sorted_time_int_strs)

    # d_influentials = load_influentials()
    # d_time_int_inf_top_vecs, d_time_int_usr_top_vecs = influentials_and_users_topic_vecs(l_time_ints, d_influentials)
    # influentials_and_users_topic_vecs_multithreads(l_time_ints, d_influentials)
    # time_int_inf_vs_usr_top_vec_cos(d_time_int_inf_top_vecs, d_time_int_usr_top_vecs)

    d_time_int_inf_top_vecs, d_time_int_usr_top_vecs = load_inf_and_usr_top_vecs()
    time_int_inf_vs_usr_top_vec_cos(d_time_int_inf_top_vecs, d_time_int_usr_top_vecs)
    print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()