import json
import logging
import random
import math
import operator
import data_preprocessing_utils

g_raw_data_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2.json'
g_raw_data_dict_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_dict.json'
g_sample_rate = 0.01
g_sampled_raw_data_dict_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_s' + str(int(g_sample_rate*100)) + '_by_time_dict.json'
g_sampled_raw_data_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_s' + str(int(g_sample_rate*100)) + '.json'


def sample_on_sorted_by_time():
    try:
        d_tid_time = dict()
        with open(g_raw_data_dict_file_path, 'r') as in_fd:
            json_data = json.load(in_fd)
            d_tid_time = dict.fromkeys(json_data.keys(), None)
        in_fd.close()
        for tid in d_tid_time:
            t_time = data_preprocessing_utils.get_user_time(json_data[tid]['created_at'].split(' '))
            d_tid_time[tid] = t_time
        l_sorted_tid_time = sorted(d_tid_time.items(), key = operator.itemgetter(1))
        l_sorted_tid_time = [item[0] for item in l_sorted_tid_time]
        n_samples = math.ceil(len(l_sorted_tid_time) * g_sample_rate)
        l_sampled_indices = random.sample(l_sorted_tid_time, n_samples)
        logging.debug('Sampling is done. Total = %s, Sampled = %s.' % (len(d_tid_time), len(l_sampled_indices)))

        d_sampled_data = dict()
        for s_tid in l_sampled_indices:
            d_sampled_data[s_tid] = json_data[s_tid]
        del json_data
    except:
        print()

    return d_sampled_data


def random_sample():
    with open(g_raw_data_file_path, 'r') as in_fd:
        l_lines = in_fd.readlines()
        l_sampled_indices = generate_sample_indices(len(l_lines))
    in_fd.close()

    l_sampled_data = []
    for i in l_sampled_indices:
        l_sampled_data.append(l_lines[i])
    with open(g_sampled_raw_data_file_path, 'w+') as out_fd:
        out_fd.writelines(l_sampled_data)
    out_fd.close()

    d_sampled_data = dict()
    for i in l_sampled_indices:
        line_json_data = json.loads(l_lines[i])
        d_sampled_data[line_json_data['id_str_h']] = line_json_data

    return d_sampled_data


def generate_sample_indices(max_index):
    l_full_indices = []
    l_sample_indices = None
    n_samples = 0
    for i in range(0, max_index):
        l_full_indices.append(i)
    n_samples = math.ceil(len(l_full_indices) * g_sample_rate)
    l_sample_indices = random.sample(l_full_indices, n_samples)
    logging.debug('Sampling is done. Total = %s, Sampled = %s.' % (len(l_full_indices), len(l_sample_indices)))
    return l_sample_indices


# this function is not correct
def verify_subgraph():
    with open(g_sampled_raw_data_dict_file_path, 'r') as in_fd:
        l_user_apr_sampled = []
        sampled_data = json.load(in_fd)
        for user_id in sampled_data:
            if '20180401000000' <= data_preprocessing_utils.get_user_time(sampled_data[user_id]['created_at'].split(' ')) <= '20180431235959':
                if 'retweeted_status' in sampled_data[user_id] \
                        or 'quoted_status' in sampled_data[user_id] \
                        or 'in_reply_to_status_id_h' in sampled_data[user_id] \
                        and sampled_data[user_id]['in_reply_to_status_id_h'] != '':
                    if user_id not in l_user_apr_sampled:
                        l_user_apr_sampled.append(user_id)
        del sampled_data
    in_fd.close()
    logging.debug('201804: Sampled Users: %s' % len(l_user_apr_sampled))

    l_user_apr_full = []
    with open(g_raw_data_dict_file_path, 'r') as in_fd:
        full_tweet_data = json.load(in_fd)
        for user_id in full_tweet_data:
            if '20180401000000' <= data_preprocessing_utils.get_user_time(full_tweet_data[user_id]['created_at'].split(' ')) <= '20180431235959':
                if 'retweeted_status' in full_tweet_data[user_id] \
                    or 'quoted_status' in full_tweet_data[user_id] \
                    or 'in_reply_to_status_id_h' in full_tweet_data[user_id] \
                        and full_tweet_data[user_id]['in_reply_to_status_id_h'] != '':
                    if user_id not in l_user_apr_full:
                        l_user_apr_full.append(user_id)
        del full_tweet_data
    in_fd.close()
    logging.debug('201804: Total Users: %s' % len(l_user_apr_full))

    overlap_count = 0
    for user_id in l_user_apr_sampled:
        if user_id in l_user_apr_full:
            overlap_count += 1
    logging.debug('201804: Overlap Users: %s' % overlap_count)


def main():
    d_sampled_data = sample_on_sorted_by_time()

    with open(g_sampled_raw_data_dict_file_path, 'w+') as out_fd:
        json.dump(d_sampled_data, out_fd)
    out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    # verify_subgraph()