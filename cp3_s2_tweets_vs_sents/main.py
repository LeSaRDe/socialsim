import json
import logging
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils


g_task_name = '20180415_20180422'
g_task_fields = g_task_name.split('_')
g_twitter_data_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_dict.json'
g_twitter_data_sorted_by_time_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_by_time_sorted.json'
g_output_data_file_path = 'Tng_an_WH_Twitter_v2_stat_top_user.json'
g_time_start = datetime.strptime(g_task_fields[0]+'000000', '%Y%m%d%H%M%S')
g_time_end = datetime.strptime(g_task_fields[1]+'000000', '%Y%m%d%H%M%S')
g_time_period = timedelta(days=7)
g_time_step = timedelta(days=7)
g_l_top_users= ['I_Wry8ROzibHG44UBImpiQ', 'a0fYqjn3qCvgH6MYNqZRew', 'fDk3wnVbzNUk9l-47tt8UQ', 'b0Z0FKyh_ciPQJx0Onzbqg']
g_en_top_user_stat = True

# {t_id: {t_time('20180412162436'), t_type('n'|'r'|'t'|'q'), t_sent([-0.56(Pos), 0.07(Neu), -1.15(Neg)])}}
# def extract_sent_data():
#     d_sent_data = dict()
#     with open(g_twitter_data_file_path, 'r') as in_fd:
#         full_json_data = json.load(in_fd)
#         for t_id in full_json_data:
#             d_sent_data[t_id] =


# '20180412162436'
def time_str_to_datetime(time_str):
    return datetime.strptime(time_str, '%Y%m%d%H%M%S')


# input: datetime objects
def gen_time_slot_seq(start, end, period, step):
    l_time_slot_seq = []
    current_slot = (start, start + period)
    while current_slot[0] <= end:
        l_time_slot_seq.append(current_slot)
        current_slot = (current_slot[0] + step, current_slot[1] + step)
    logging.debug('%s time slots are generated.' % len(l_time_slot_seq))
    return l_time_slot_seq


def sort_t_data_by_time():
    with open(g_twitter_data_file_path, 'r') as in_fd:
        t_data = json.load(in_fd)
        logging.debug('%s tweets in total.' % len(t_data))
        l_sorted_t_data = sorted(t_data.items(), key=lambda k: data_preprocessing_utils.get_user_time(k[1]['created_at'].split(' ')))
    in_fd.close()

    with open(g_twitter_data_sorted_by_time_file_path, 'a+') as out_fd:
        for t_data in l_sorted_t_data:
            json.dump(t_data[1], out_fd)
            out_fd.write('\n')
    out_fd.close()
    logging.debug('All done!')
    return l_sorted_t_data


def count_by_time_seq(l_sorted_t_data, l_time_slot_seq, curr_user_id=None):
    d_count_by_time_seq = dict()
    current_time_slot_idx = 0
    current_time_slot = l_time_slot_seq[current_time_slot_idx]
    current_time_slot_t_count = 0
    # for time_idx, time_slot in enumerate(l_time_slot_seq):
    # retweet(int), quote(int), reply(int), new post(int), pos(real), neu(real), neg(real)
    d_count_by_time_seq[current_time_slot_idx] = [0, 0, 0, 0, 0, 0, 0]
    for data_idx, t_data in enumerate(l_sorted_t_data):
        if g_en_top_user_stat:
            user_id = t_data[1]['user']['id_str_h']
            if user_id != curr_user_id:
                continue
        t_time = time_str_to_datetime(data_preprocessing_utils.get_user_time(t_data[1]['created_at'].split(' ')))
        if t_time > current_time_slot[1]:
            d_count_by_time_seq[current_time_slot_idx][4] = d_count_by_time_seq[current_time_slot_idx][4] / current_time_slot_t_count
            d_count_by_time_seq[current_time_slot_idx][5] = d_count_by_time_seq[current_time_slot_idx][5] / current_time_slot_t_count
            d_count_by_time_seq[current_time_slot_idx][6] = d_count_by_time_seq[current_time_slot_idx][6] / current_time_slot_t_count
            if current_time_slot_idx >= len(l_time_slot_seq):
                logging.error('Tweet beyond the time sequence occurs: %s' % t_data[0])
                return d_count_by_time_seq
            else:
                current_time_slot_t_count = 0
                current_time_slot_idx += 1
                current_time_slot = l_time_slot_seq[current_time_slot_idx]
                d_count_by_time_seq[current_time_slot_idx] = [0, 0, 0, 0, 0, 0, 0]
        if data_preprocessing_utils.get_tweet_type(t_data[1]) == 't':
            d_count_by_time_seq[current_time_slot_idx][0] += 1
        elif data_preprocessing_utils.get_tweet_type(t_data[1]) == 'q':
            d_count_by_time_seq[current_time_slot_idx][1] += 1
        elif data_preprocessing_utils.get_tweet_type(t_data[1]) == 'r':
            d_count_by_time_seq[current_time_slot_idx][2] += 1
        elif data_preprocessing_utils.get_tweet_type(t_data[1]) == 'n':
            d_count_by_time_seq[current_time_slot_idx][3] += 1
        else:
            raise Exception('Wrong type of tweet occurs!')
        if 'extension' in t_data[1] and 'sentiment_scores' in t_data[1]['extension']:
            l_sents = json.loads(t_data[1]['extension']['sentiment_scores'])
        else:
            logging.debug('This tweet does not have extension or sentiment_scores: %s' % t_data[0])
            l_sents = {'positive': 0, 'neutral': 0, 'negative': 0}
        if l_sents is None or len(l_sents) != 3:
            raise Exception('No sentiment available to this tweet: %s' % t_data[0])
        d_count_by_time_seq[current_time_slot_idx][4] += l_sents['positive']
        d_count_by_time_seq[current_time_slot_idx][5] += l_sents['neutral']
        d_count_by_time_seq[current_time_slot_idx][6] += l_sents['negative']
        current_time_slot_t_count += 1
        if data_idx % 10000 == 0 and data_idx >= 10000:
            logging.debug('%s tweets have been scanned. %s time slots have been filled.' % (data_idx+1, current_time_slot_idx+1))
    d_count_by_time_seq[current_time_slot_idx][4] = d_count_by_time_seq[current_time_slot_idx][4] / current_time_slot_t_count
    d_count_by_time_seq[current_time_slot_idx][5] = d_count_by_time_seq[current_time_slot_idx][5] / current_time_slot_t_count
    d_count_by_time_seq[current_time_slot_idx][6] = d_count_by_time_seq[current_time_slot_idx][6] / current_time_slot_t_count
    logging.debug('All tweets have been scanned. %s time slots have been filled.' % str(current_time_slot_idx + 1))
    return d_count_by_time_seq



def main():
    l_sorted_t_data = sort_t_data_by_time()
    l_time_slot_seq = gen_time_slot_seq(g_time_start, g_time_end, g_time_period, g_time_step)
    d_count_by_time_seq = count_by_time_seq(l_sorted_t_data, l_time_slot_seq)
    with open(g_output_data_file_path, 'w+') as out_fd:
        json.dump(d_count_by_time_seq, out_fd, indent=4)
    out_fd.close()
    # print()


def sample_main():
    l_sorted_t_data = sort_t_data_by_time()
    l_time_slot_seq = gen_time_slot_seq(g_time_start, g_time_end, g_time_period, g_time_step)
    d_top_user_stat = dict()
    for user_id in g_l_top_users:
        d_count_by_time_seq = count_by_time_seq(l_sorted_t_data, l_time_slot_seq, user_id)
        d_top_user_stat[user_id] = d_count_by_time_seq
    with open(g_output_data_file_path, 'w+') as out_fd:
        json.dump(d_count_by_time_seq, out_fd, indent=4)
    out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # main()
    sample_main()