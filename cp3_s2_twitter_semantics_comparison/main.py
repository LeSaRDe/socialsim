import json
import time
import logging
import sent2vec
import scipy.spatial.distance as scipyd
import multiprocessing as mp
import math
import threading
import networkx as nx
from os import listdir
import os

# '20180513_20180520', '20180527_20180603', '20180624_20180701', '20180708_20180715',
# '20180415_20180422', '20180429_20180506', '20180513_20180520',
#              '20180527_20180603', '20180610_20180617', '20180624_20180701',
#              '20180708_20180715', '20180722_20180729', '20180805_20180812',
#              '20180819_20180826'
g_sent2vec_model_file_path = 'sent2vec/twitter_unigrams.bin'
g_l_weeks = ['20180819_20180826',
              '20180610_20180617',
             '20180805_20180812',  '20180415_20180422', '20180429_20180506', '20180722_20180729']
# g_task_name = '20180513_20180520'
# g_task_fields = g_task_name.split('_')
g_path_prefix_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/'
g_clean_text_file_path_format = g_path_prefix_format + 'tw_wh_data_{1}_full_user_text_clean_by_moment.json'
g_fsgraph_file_path_fromat = g_path_prefix_format + 'tw_wh_fsgraph_{1}_full.json'
# g_start_time = g_task_fields[0] + '000000'
# g_end_time = g_task_fields[1] + '000000'
g_output_tsgraph_file_path_format = g_path_prefix_format + 'tw_wh_tsgraph_{1}_full_fs.json'
g_text_sim_folder_path_format = g_path_prefix_format + 'text_sim_{1}/'
g_users_file_path_format = g_path_prefix_format + 'users_{1}_full.txt'
g_l_top_users= ['I_Wry8ROzibHG44UBImpiQ', 'a0fYqjn3qCvgH6MYNqZRew', 'fDk3wnVbzNUk9l-47tt8UQ', 'b0Z0FKyh_ciPQJx0Onzbqg']
# g_output_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/'+g_task_name+'/'
g_sim_threshold = 0.5
g_en_pure_tsgraph = False


def fetch_text(task_name, l_users, start=None, end=None):
    d_user_text = dict()
    timer_s = time.time()
    i = 0
    with open(g_clean_text_file_path_format.format(task_name, task_name), 'r') as in_fd:
        tw_data = json.load(in_fd)
        logging.debug('%s users to be scanned.' % len(l_users))
        for idx, user_id in enumerate(l_users):
            if user_id in tw_data:
                l_user_text = []
                for tw_time in tw_data[user_id]:
                    if start is not None and tw_time < start:
                        continue
                    if end is not None and tw_time > end:
                        continue
                    l_user_text += tw_data[user_id][tw_time]
                if len(l_user_text) > 0:
                    d_user_text[user_id] = l_user_text
                else:
                    logging.error('%s does not have meaningful text.' % user_id)
            else:
                logging.error('%s is not in the clean text user list.' % user_id)
            if idx % 100 == 0 and idx >= 100:
                logging.debug('%s users have been scanned in %s seconds.' % (idx, time.time()-timer_s))
        logging.debug('All users have been scanned in %s seconds.' % str(time.time()-timer_s))
    in_fd.close()
    # del tw_data
    return d_user_text


def get_one_doc_vect(sent2vec_model, l_sents):
    l_sents = [sent.lower() for sent in l_sents]
    embs = sent2vec_model.embed_sentences(l_sents, mp.cpu_count())
    doc_vect = [sum(x)/len(l_sents) for x in zip(*embs)]
    return doc_vect


def generate_job_list(l_users):
    len_l_users = len(l_users)
    total = len_l_users * (len_l_users-1) / 2
    total_per_job = math.ceil(total / mp.cpu_count())
    l_l_jobs = []
    l_jobs = []
    for i in range(0, len_l_users-1):
        for j in range(i+1, len_l_users):
            if len(l_jobs) >= total_per_job:
                l_l_jobs.append(l_jobs)
                l_jobs = []
            l_jobs.append([l_users[i], l_users[j]])
    if len(l_jobs) > 0:
        l_l_jobs.append(l_jobs)
    return l_l_jobs


def generate_job_list_by_fsgraph(fsgraph):
    d_undi_edges = dict()
    for edge in fsgraph.edges:
        if edge[0] in d_undi_edges:
            if edge[1] in d_undi_edges[edge[0]]:
                continue
            else:
                if edge[1] not in d_undi_edges:
                    d_undi_edges[edge[1]] = [edge[0]]
                else:
                    if edge[0] in d_undi_edges[edge[1]]:
                        continue
                    else:
                        if len(d_undi_edges[edge[1]]) < len(d_undi_edges[edge[0]]):
                            d_undi_edges[edge[1]].append(edge[0])
                        else:
                            d_undi_edges[edge[0]].append(edge[1])
        else:
            if edge[1] not in d_undi_edges:
                d_undi_edges[edge[1]] = [edge[0]]
            else:
                if edge[0] in d_undi_edges[edge[1]]:
                    continue
                else:
                    d_undi_edges[edge[0]] = [edge[1]]

    count_jobs = sum([len(item) for item in d_undi_edges.values()])
    logging.debug('%s jobs in total.' % count_jobs)
    total_per_job = math.ceil(count_jobs / mp.cpu_count())
    l_l_jobs = []
    l_jobs = []
    for node_0 in d_undi_edges:
        for node_1 in d_undi_edges[node_0]:
            if len(l_jobs) >= total_per_job:
                l_l_jobs.append(l_jobs)
                l_jobs = []
            l_jobs.append([node_0, node_1])
    if len(l_jobs) > 0:
        l_l_jobs.append(l_jobs)
    return l_l_jobs


def text_sim_thread_func(l_jobs, task_name, d_user_text_vect, t_name):
    timer_s = time.time()
    logging.debug('Thread %s: %s jobs to go.' % (t_name, len(l_jobs)))
    l_rets = []
    count = 0
    sim = 0.0
    for job in l_jobs:
        if not job[0] in d_user_text_vect or not job[1] in d_user_text_vect:
            ret_str = job[0] + ' ' + job[1] + ' ' + str(0.0)
            l_rets.append(ret_str)
            logging.error('%s or %s is not in d_user_text.' % (job[0], job[1]))
            # count += 1
            # if count % 5000 == 0:
            #     logging.debug('Thread %s: %s jobs have done in %s.' % (t_name, count, str(time.time() - timer_s)))
            # continue
        else:
            sim = 1.0 - scipyd.cosine(d_user_text_vect[job[0]], d_user_text_vect[job[1]])
            if sim == float('nan'):
                sim = 0.0
            if str(sim) != 'nan':
                ret_str = job[0] + ' ' + job[1] + ' ' + str(sim)
                l_rets.append(ret_str)
            else:
                ret_str = job[0] + ' ' + job[1] + ' ' + str(0.0)
                l_rets.append(ret_str)
                logging.error('%s, %s lead to NaN.' % (job[0], job[1]))
        count += 1
        if count % 5000 == 0:
            logging.debug('Thread %s: %s jobs have done in %s.' % (t_name, count, str(time.time()-timer_s)))
    logging.debug('Thread %s: All jobs have done in %s seconds.' % (t_name, str(time.time()-timer_s)))
    out_str = '\n'.join(l_rets)

    output_file_path = g_text_sim_folder_path_format.format(task_name, task_name) + t_name
    with open(output_file_path, 'w+') as out_fd:
        out_fd.write(out_str)
    out_fd.close()
    logging.debug('Thread %s: Written to file. All done in %s seconds!' % (t_name, str(time.time() - timer_s)))


def construct_text_sim_graph(task_name):
    ts_graph = nx.Graph()
    file_count = 0
    t_start = time.time()
    for each_file in [g_text_sim_folder_path_format.format(task_name, task_name) + file for file in listdir(g_text_sim_folder_path_format.format(task_name, task_name))]:
        file_count += 1
        line_count = 0
        with open(each_file, 'r') as in_fd:
            line = in_fd.readline()
            while line:
                line_count += 1
                fields = [field.strip() for field in line.split(' ')]
                node_1 = fields[0]
                node_2 = fields[1]
                if fields[2] == 'nan':
                    line = in_fd.readline()
                    if line_count % 5000 == 0 and line_count > 5000:
                        logging.debug('%s lines read from file %s.' % (line_count, file_count))
                    logging.error('%s, %s have NaN in the record.' % (node_1, node_2))
                    continue
                sim = float(fields[2])
                if not ts_graph.has_edge(node_1, node_2):
                    if g_en_pure_tsgraph:
                        if sim >= g_sim_threshold:
                            ts_graph.add_edge(node_1, node_2, weight=sim)
                    else:
                        ts_graph.add_edge(node_1, node_2, weight=sim)
                else:
                    logging.error('Edge: (%s, %s) already exists.' % (node_1, node_2))
                line = in_fd.readline()
                if line_count % 5000 == 0 and line_count > 5000:
                    logging.debug('%s lines read from file %s.' % (line_count, file_count))
        in_fd.close()
        logging.debug('%s file is done in %s seconds.' % (file_count, str(time.time()-t_start)))
    logging.debug('Text similarity graph is done in %s seconds.' % str(time.time()-t_start))

    with open(g_output_tsgraph_file_path_format.format(task_name, task_name), 'w+') as out_fd:
        ts_graph_data = nx.adjacency_data(ts_graph)
        json.dump(ts_graph_data, out_fd, indent=4)
    out_fd.close()
    logging.debug('Text similarity graph write-to-file is done in %s seconds.' % str(time.time() - t_start))
    print(nx.info(ts_graph))


def read_ts_graph():
    with open(g_output_graph_file_path, 'r') as in_fd:
        ts_graph_data = json.load(in_fd)
        ts_graph = nx.adjacency_graph(ts_graph_data)
        print(nx.info(ts_graph))
        sum = 0
        i = 0
        for item in ts_graph.edges.data('weight'):
            sum += item[2]
            i += 1
            print('%s : %s' % (i, sum))
            if i % 500 == 0 and i > 500:
                input('Wait...')
        # avg_weight = sum([item[2] for item in ts_graph.edges.data('weight')]) / len(ts_graph.edges)
        # print('Avg. weight = %s' % avg_weight)
    in_fd.close()


def main():
    for week in g_l_weeks:
        print('%s starts...' % week)
        with open(g_fsgraph_file_path_fromat.format(week, week), 'r') as in_fsgraph_fd:
            fsgraph_data = json.load(in_fsgraph_fd)
            fsgraph = nx.adjacency_graph(fsgraph_data)
        in_fsgraph_fd.close()
        # del fsgraph_data
        l_fsgraph_users = list(fsgraph.nodes)
        d_user_text = fetch_text(week, l_fsgraph_users)
        logging.debug('%s users have texts.' % len(d_user_text))
        n_sent = 0
        for user_id in d_user_text:
            n_sent += len(d_user_text[user_id])
        logging.debug('Each user of fsgraph in average has %s sentences.' % str(n_sent / len(d_user_text)))

        timer_s = time.time()
        d_user_text_vect = dict()
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(g_sent2vec_model_file_path)
        for user_id in d_user_text:
            user_text_vect = get_one_doc_vect(sent2vec_model, d_user_text[user_id])
            d_user_text_vect[user_id] = user_text_vect
        logging.debug('Got all user text vectors in %s seconds.' % str(time.time() - timer_s))
        # del d_user_text

        l_l_jobs = generate_job_list_by_fsgraph(fsgraph)
        logging.debug('Job list is done in % seconds.' % str(time.time()-timer_s))

        for each_file in [g_text_sim_folder_path_format.format(week, week) + file for file in
                          listdir(g_text_sim_folder_path_format.format(week, week))]:
            os.remove(each_file)

        l_threads = []
        t_count = 0
        timer_s = time.time()
        for l_jobs in l_l_jobs:
            job_t_name = 'UT_SIM_'+str(t_count)
            job_t = threading.Thread(target=text_sim_thread_func, args=(l_jobs, week, d_user_text_vect, job_t_name))
            job_t.setName(job_t_name)
            job_t.start()
            l_threads.append(job_t)
            t_count += 1
        while len(l_threads) > 0:
            for t in l_threads:
                t.join(2)
                if not t.is_alive():
                    logging.debug('%s is done.' % t.getName())
                    l_threads.remove(t)
        logging.debug('All user text sim threads are done in %s seconds.' % str(time.time() - timer_s))

        construct_text_sim_graph(week)
        print()

    # timer_s = time.time()
    # l_users = list(d_user_text_vect.keys())
    # user_text_graph = nx.Graph()
    # for i in range(0, len(l_users)-1):
    #     for j in range(i+1, len(l_users)):
    #         if not user_text_graph.has_edge(l_users[i], l_users[j]):
    #             sim = 1.0 - scipyd.cosine(d_user_text_vect[l_users[i]], d_user_text_vect[l_users[j]])
    #             user_text_graph.add_edge(l_users[i], l_users[j], weight=sim)
    # logging.debug('User text graph is done in %s seconds.' % str(time.time() - timer_s))
    #
    # with open(g_output_graph_file_path, 'w+') as out_fd:
    #     user_text_graph_data = nx.adjacency_data(user_text_graph)
    #     json.dump(user_text_graph_data, out_fd, indent=4)
    # out_fd.close()


def pure_tsgraph_main():
    for week in g_l_weeks:
        print('%s starts...' % week)
        with open(g_users_file_path_format.format(week, week), 'r') as in_users_fd:
            l_users = [user.strip() for user in in_users_fd.readlines()]
        d_user_text = fetch_text(week, l_users)
        logging.debug('%s users have texts.' % len(d_user_text))
        n_sent = 0
        for user_id in d_user_text:
            n_sent += len(d_user_text[user_id])
        logging.debug('Each user in average has %s sentences.' % str(n_sent / len(d_user_text)))

        timer_s = time.time()
        d_user_text_vect = dict()
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(g_sent2vec_model_file_path)
        for user_id in d_user_text:
            user_text_vect = get_one_doc_vect(sent2vec_model, d_user_text[user_id])
            d_user_text_vect[user_id] = user_text_vect
        logging.debug('Got all user text vectors in %s seconds.' % str(time.time() - timer_s))
        # del d_user_text

        l_l_jobs = generate_job_list(l_users)
        logging.debug('Job list is done in % seconds.' % str(time.time()-timer_s))

        for each_file in [g_text_sim_folder_path_format.format(week, week) + file for file in
                          listdir(g_text_sim_folder_path_format.format(week, week))]:
            os.remove(each_file)

        l_threads = []
        t_count = 0
        timer_s = time.time()
        for l_jobs in l_l_jobs:
            job_t_name = 'UT_SIM_'+str(t_count)
            job_t = threading.Thread(target=text_sim_thread_func, args=(l_jobs, week, d_user_text_vect, job_t_name))
            job_t.setName(job_t_name)
            job_t.start()
            l_threads.append(job_t)
            t_count += 1
        while len(l_threads) > 0:
            for t in l_threads:
                t.join(2)
                if not t.is_alive():
                    logging.debug('%s is done.' % t.getName())
                    l_threads.remove(t)
        logging.debug('All user text sim threads are done in %s seconds.' % str(time.time() - timer_s))

        construct_text_sim_graph(week)
        print()


def sample_main():
    d_user_text = fetch_text(g_start_time, g_end_time)
    l_overall_text = []
    d_top_user_text = dict()
    for user_id in d_user_text:
        l_overall_text += d_user_text[user_id]
        if user_id in g_l_top_users:
            d_top_user_text[user_id] = d_user_text[user_id]

    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(g_sent2vec_model_file_path)
    embs = sent2vec_model.embed_sentences(l_overall_text, mp.cpu_count())
    overall_text_vect = [sum(x) / len(l_overall_text) for x in zip(*embs)]
    for user in d_top_user_text:
        embs = sent2vec_model.embed_sentences(d_top_user_text[user], mp.cpu_count())
        user_text_vect = [sum(x) / len(d_top_user_text[user]) for x in zip(*embs)]
        d_top_user_text[user] = user_text_vect
    with open(g_output_prefix + 'overall_text_vect.txt', 'w+') as out_fd:
        d_output = dict()
        d_output[g_task_name] = overall_text_vect
        json.dump(d_output, out_fd, indent=4)
    out_fd.close()
    with open(g_output_prefix + 'top_user_text_vect.txt', 'w+') as out_fd:
        json.dump(d_top_user_text, out_fd, indent=4)
    out_fd.close()

    d_top_vs_overall = {user_id : None for user_id in g_l_top_users}
    for user_id in d_top_vs_overall:
        sim = 1.0 - scipyd.cosine(overall_text_vect, d_top_user_text[user_id])
        d_top_vs_overall[user_id] = sim
    with open(g_output_prefix + 'top_user_vs_overall.txt', 'w+') as out_fd:
        json.dump(d_top_vs_overall, out_fd, indent=4)
    out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # read_ts_graph()
    # construct_text_sim_graph()
    main()
    # sample_main()
    # pure_tsgraph_main()