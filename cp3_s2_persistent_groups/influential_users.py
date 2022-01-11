import logging
import networkx as nx
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils
import time
import json
import math
import multiprocessing
import threading
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_fsgraph_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph.gml'
g_influential_users_inter_rets_path = g_time_series_data_path_prefix + 'influential_user_inter_rets/'
g_influential_users_inter_rets_format = g_influential_users_inter_rets_path + '{0}.json'
g_influential_users_path = g_time_series_data_path_prefix + 'influential_users.json'
g_init_cut_point_threshold = 0.5
# g_out_degree_bottom = 10
g_other_dont_move_tag = 999999
g_max_iterations = 10000


def find_init_cut(fsgraph, l_sorted_nodes, time_int_str):
    covered_edge_count = 0
    total_edge_count = len(fsgraph.edges)
    for idx, node in enumerate(l_sorted_nodes):
        covered_edge_count += node[1]
        if float(covered_edge_count) / total_edge_count >= g_init_cut_point_threshold:
            l_init_influentials = l_sorted_nodes[:idx+1]
            l_init_others = l_sorted_nodes[idx+1:]
            return l_init_influentials, l_init_others
    raise Exception('%s cannot find init cut.' % time_int_str)


def find_cand_movable_nodes(fsgraph, l_init_influentials, l_init_others):
    influential_bottom = 0.1 * sum([node[1] for node in l_init_influentials]) / len(l_init_influentials)
    l_influential_moves = []
    for node_i in l_init_influentials:
        in_deg = fsgraph.in_degree(node_i[0])
        out_deg = node_i[1]
        if out_deg == 0:
            l_init_others.append(node_i)
            continue
        # the higher the more likely to be moved
        move_p = in_deg / out_deg
        l_influential_moves.append((node_i[0], move_p))
    l_influential_moves.append(('DUMMY_I', -1))
    l_influential_moves = sorted(l_influential_moves, key=lambda k: k[1], reverse=True)

    l_other_moves = []
    for node_o in l_init_others:
        in_deg = fsgraph.in_degree(node_o[0])
        out_deg = node_o[1]
        if out_deg == 0 or out_deg < influential_bottom:
            move_p = g_other_dont_move_tag
        else:
            # the lower the more likely to be moved
            move_p = in_deg / out_deg
        l_other_moves.append((node_o[0], move_p))
    l_other_moves.append(('DUMMY_O', g_other_dont_move_tag-1))
    l_other_moves = sorted(l_other_moves, key=lambda k: k[1])

    return l_influential_moves, l_other_moves


def pre_partition_for_one_fsgraph(time_int_str):
    fsgraph = nx.read_gml(g_fsgraph_format.format(time_int_str, time_int_str))
    l_sorted_nodes = sorted(list(fsgraph.out_degree(fsgraph.nodes)), key=lambda k: k[1], reverse=True)
    l_init_influentials, l_init_others = find_init_cut(fsgraph, l_sorted_nodes, time_int_str)
    return fsgraph, l_init_influentials, l_init_others


def compute_obj(fsgraph, l_influential_moves, l_other_moves, time_int_str):
    l_influentials = [node[0] for node in l_influential_moves]
    l_others = [node[0] for node in l_other_moves]
    e_AA = 0
    e_AB = 0
    e_BA = 0
    e_BB = 0
    for edge in fsgraph.edges:
        src = edge[0]
        trg = edge[1]
        if src in l_influentials and trg in l_influentials:
            e_AA += 1
        elif src in l_influentials and trg in l_others:
            e_AB += 1
        elif src in l_others and trg in l_influentials:
            e_BA += 1
        elif src in l_others and trg in l_others:
            e_BB += 1
        else:
            raise Exception('Incorrect nodes %s and %s at %s' % (src, trg, time_int_str))
    a_A = e_AA + e_AB
    a_B = e_BA + e_BB
    b_A = e_AA + e_BA
    b_B = e_AB + e_BB

    if (1 - a_A*b_B) != 0:
        obj = (e_AB - a_A*b_B) / (1 - a_A*b_B)
    else:
        obj = 1.0
    return obj


def greedy_optimizer(fsgraph, l_influential_moves, l_other_moves, time_int_str, iter):
    if iter >= g_max_iterations:
        return compute_obj(fsgraph, l_influential_moves, l_other_moves,
                           time_int_str), l_influential_moves, l_other_moves

    top_influential_move = l_influential_moves[0]
    top_other_move = l_other_moves[0]
    best_move_mark = None
    # if top_other_move[1] == 0 and top_influential_move[1] == 0:
    #     return True, l_influential_moves, l_other_moves
    if top_other_move[1] == 0:
        best_move = top_other_move
        # if best_move[0] == 'DUMMY_I':
        #     return True, l_influential_moves, l_other_moves
        best_move_mark = 'o'
    elif top_other_move[1] != 0 and top_influential_move[1] == 0:
        best_move = top_other_move
        if best_move[0] == 'DUMMY_O':
            return compute_obj(fsgraph, l_influential_moves, l_other_moves, time_int_str), l_influential_moves, l_other_moves
        best_move_mark = 'o'
    else:
        if top_other_move[0] == 'DUMMY_O' and top_influential_move[0] == 'DUMMY_I':
            return compute_obj(fsgraph, l_influential_moves, l_other_moves, time_int_str), l_influential_moves, l_other_moves
        elif top_other_move[0] == 'DUMMY_O' and top_influential_move[0] != 'DUMMY_I':
            best_move = top_influential_move
            best_move_mark = 'i'
        elif top_other_move[0] != 'DUMMY_O' and top_influential_move[0] == 'DUMMY_I':
            best_move = top_other_move
            best_move_mark = 'o'
        else:
            if top_other_move[1] < (1/top_influential_move[1]):
                best_move = top_other_move
                best_move_mark = 'o'
            else:
                best_move = top_influential_move
                best_move_mark = 'i'

    cur_obj = compute_obj(fsgraph, l_influential_moves, l_other_moves, time_int_str)
    if best_move_mark == 'i':
        imp_obj = compute_obj(fsgraph, [node for node in l_influential_moves if node != best_move],
                              l_other_moves + [best_move], time_int_str)
    elif best_move_mark == 'o':
        imp_obj = compute_obj(fsgraph, l_influential_moves + [best_move],
                              [node for node in l_other_moves if node != best_move], time_int_str)
    else:
        raise Exception('%s best_move is not selected.' % time_int_str)
    if imp_obj > cur_obj:
        if best_move_mark == 'i':
            l_other_moves.append(best_move)
            l_influential_moves.remove(best_move)
        elif best_move_mark == 'o':
            l_influential_moves.append(best_move)
            l_other_moves.remove(best_move)
    else:
        if best_move_mark == 'i':
            l_influential_moves.remove(best_move)
            l_influential_moves.append(best_move)
        elif best_move_mark == 'o':
            l_other_moves.remove(best_move)
            l_other_moves.append(best_move)

    iter += 1

    return greedy_optimizer(fsgraph, l_influential_moves, l_other_moves, time_int_str, iter)


def find_influentials_for_one_time_interval(time_int_str):
    logging.debug('Start %s influential user finding...' % time_int_str)
    timer_start = time.time()
    fsgraph, l_init_influentials, l_init_others = pre_partition_for_one_fsgraph(time_int_str)
    l_influential_moves, l_other_moves = find_cand_movable_nodes(fsgraph, l_init_influentials, l_init_others)
    score, l_influentials, l_others = greedy_optimizer(fsgraph, l_influential_moves, l_other_moves, time_int_str, 0)
    logging.debug('%s influential users are done in %s seconds.' % (time_int_str, str(time.time()-timer_start)))
    return score, l_influentials, l_others


def find_influentials_for_time_intervals(l_time_ints, tid):
    d_influential_users = dict()
    for time_int in l_time_ints:
        time_int_str = data_preprocessing_utils.time_int_to_time_int_str(time_int)
        score, l_influentials, l_others = find_influentials_for_one_time_interval(time_int_str)
        d_influential_users[time_int_str] = dict()
        d_influential_users[time_int_str]['score'] = score
        d_influential_users[time_int_str]['influentials'] = l_influentials
    logging.debug('%s influential users are done.' % len(l_time_ints))
    with open(g_influential_users_inter_rets_format.format(tid), 'w+') as out_fd:
        json.dump(d_influential_users, out_fd)
        out_fd.close()
    logging.debug('%s batch is done.' % tid)


def find_influentials_multithreads(l_time_ints):
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
        t = threading.Thread(target=find_influentials_for_time_intervals, args=(l_each_batch, t_id))
        t.setName('Influential_t_' + str(t_id))
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

    logging.debug('All influential users have been written.')


def build_influentials():
    l_inter_files = os.listdir(g_influential_users_inter_rets_path)
    d_influential_users = dict()
    for inter_file in l_inter_files:
        with open(g_influential_users_inter_rets_path + inter_file, 'r') as in_fd:
            d_inf = json.load(in_fd)
            in_fd.close()
            for time_int in d_inf:
                l_influentials = [item[0] for item in d_inf[time_int]['influentials']]
                l_influentials.remove('DUMMY_I')
                d_influential_users[time_int] = l_influentials
    with open(g_influential_users_path, 'w+') as out_fd:
        json.dump(d_influential_users, out_fd)
        out_fd.close()
    # l_existing_time_ints = list(d_influential_users.keys())
    return d_influential_users


def load_influentials():
    with open(g_influential_users_path, 'r') as in_fd:
        d_influentials = json.load(in_fd)
        in_fd.close()
    return d_influentials


def influentials_on_time_ints(d_influentials):
    l_time_ints = data_preprocessing_utils.read_time_ints()
    l_sorted_time_int_strs = sorted([data_preprocessing_utils.time_int_to_time_int_str(time_int)
                                     for time_int in l_time_ints])
    l_influentials = []
    for time_int in d_influentials:
        for uid in d_influentials[time_int]:
            if uid not in l_influentials:
                l_influentials.append(uid)

    influential_mat = np.zeros((len(l_influentials), len(l_sorted_time_int_strs)))
    for time_int in d_influentials:
        idx_2 = l_sorted_time_int_strs.index(time_int)
        for idx_1, uid in enumerate(l_influentials):
            if uid in d_influentials[time_int]:
                influential_mat[idx_1][idx_2] = 1
            else:
                influential_mat[idx_1][idx_2] = 0

    l_xticks = [i for i in range(0, len(l_sorted_time_int_strs), 10)]
    l_yticks = [i for i in range(0, len(l_sorted_time_int_strs), 5)]
    # plt.imshow(ret_mat, cmap='gray', vmin=min, vmax=max)
    plt.imshow(influential_mat, interpolation=None)
    # plt.colorbar()
    plt.xticks(l_xticks)
    plt.yticks(l_yticks, l_influentials)
    plt.show()


g_time_format = '%Y%m%d'
def main():
    # l_time_ints = data_preprocessing_utils.read_time_ints()
    # l_existing_time_ints = load_existing_files()
    # l_time_ints = [time_int for time_int in l_time_ints if
    #                data_preprocessing_utils.time_int_to_time_int_str(time_int) not in l_existing_time_ints]
    # find_influentials_multithreads(l_time_ints)

    # start_day_str = '20180907'
    # end_day_str = '20180913'
    # find_influentials_for_time_intervals([(datetime.strptime(start_day_str.strip(), g_time_format),
    #                                        datetime.strptime(end_day_str.strip(), g_time_format))], 0)

    # build_influentials()

    d_influentials = load_influentials()
    influentials_on_time_ints(d_influentials)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()