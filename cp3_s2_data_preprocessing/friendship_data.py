import json
import logging
import time



g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
             '20180527_20180603', '20180610_20180617', '20180624_20180701',
             '20180708_20180715', '20180722_20180729', '20180805_20180812',
             '20180819_20180826']
# g_task_name = '20180513_20180520'
g_path_prefix_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/'
# g_raw_data_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_dict.json'
g_raw_data_file_path_format = g_path_prefix_format + 'tw_wh_data_{1}_timesorted_dict.json'
g_output_file_path_format = g_path_prefix_format + '/tw_wh_fsgraph_data_{1}_full.json'
g_retweet_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_Retweet_Chain_WH.json'
g_users_file_path_format = g_path_prefix_format + 'users_{1}_full.txt'


def translate_time(time_str):
    l_fields = [field.strip() for field in time_str.split()]
    month = l_fields[1]
    if month == 'Jan':
        month = '01'
    elif month == 'Feb':
        month = '02'
    elif month == 'Mar':
        month = '03'
    elif month == 'Apr':
        month = '04'
    elif month == 'May':
        month = '05'
    elif month == 'Jun':
        month = '06'
    elif month == 'Jul':
        month = '07'
    elif month == 'Aug':
        month = '08'
    elif month == 'Sep':
        month = '09'
    elif month == 'Oct':
        month = '10'
    elif month == 'Nov':
        month = '11'
    elif month == 'Dec':
        month = '12'
    day = l_fields[2]
    year = l_fields[5]
    time_t = l_fields[3]
    time_t = ''.join([time_field for time_field in time_t.split(':')])
    ret_time = year + month + day + time_t
    return ret_time


def load_users(task_name):
    with open(g_users_file_path_format.format(task_name, task_name), 'r') as in_fd:
        l_users = in_fd.readlines()
    in_fd.close()
    return {user.strip() : None for user in l_users}


def construct_fsgraph(task_name, d_users):
    d_friendship_data = dict()
    start = time.time()
    i = 0
    with open(g_raw_data_file_path_format.format(task_name, task_name), 'r') as raw_in_fd:
        raw_tweet_dict = json.load(raw_in_fd)
        for tweet_id in raw_tweet_dict:
            tweet = raw_tweet_dict[tweet_id]
            time_t = translate_time(tweet['created_at'])
            source = None
            target = None
            if 'in_reply_to_status_id_h' in tweet and tweet['in_reply_to_status_id_h'] != '':
                source_tw_id = tweet['in_reply_to_status_id_h']
                if source_tw_id in raw_tweet_dict:
                    target = tweet['user']['id_str_h']
                    source = raw_tweet_dict[source_tw_id]['user']['id_str_h']
                    if target not in d_users or source not in d_users:
                        continue
                    type_t = 'r'
                else:
                    continue
            elif 'quoted_status' in tweet:
                target = tweet['user']['id_str_h']
                source = tweet['quoted_status']['user']['id_str_h']
                if target not in d_users or source not in d_users:
                    continue
                type_t = 'q'
            if tweet_id not in d_friendship_data:
                if source is not None and target is not None:
                    d_friendship_data[tweet_id] = [source, target, type_t, time_t]
            else:
                logging.error('Conflict tweet id exists! %s' % tweet)
            if i % 5000 == 0 and i >= 5000:
                logging.debug('%s raw tweets have been scanned in %s seconds.' % (i, str(time.time() - start)))
            i += 1
        logging.debug('Replied and quotes have been scanned in %s seconds.' % str(time.time() - start))
    raw_in_fd.close()

    start = time.time()
    i = 0
    with open(g_retweet_file_path, 'r') as re_in_fd:
        l_lines = re_in_fd.readlines()
        for j in range(1, len(l_lines)):
            retweet = json.loads(l_lines[j])
            ret_id = retweet['tweet_id_h']
            reted_id = retweet['retweeted_from_tweet_id_h']
            if ret_id not in raw_tweet_dict or reted_id not in raw_tweet_dict:
                continue
            if ret_id not in d_friendship_data:
                source = raw_tweet_dict[reted_id]['user']['id_str_h']
                target = raw_tweet_dict[ret_id]['user']['id_str_h']
                if target not in d_users or source not in d_users:
                    continue
                type_t = 't'
                time_t = translate_time(raw_tweet_dict[ret_id]['created_at'])
                d_friendship_data[ret_id] = [source, target, type_t, time_t]
            if i % 5000 == 0 and i >= 5000:
                logging.debug('%s retweets have been scanned in %s seconds.' % (i, time.time() - start))
            i += 1
        logging.debug('Retweets have been scanned in %s seconds.' % str(time.time() - start))
    re_in_fd.close()

    start = time.time()
    with open(g_output_file_path_format.format(task_name, task_name), 'w+') as out_fd:
        json.dump(d_friendship_data, out_fd, indent=4)
    out_fd.close()
    logging.debug('Friendship data has been written in %s seconds.' % str(time.time() - start))


def main():
    for week in g_l_weeks:
        d_users = load_users(week)
        construct_fsgraph(week, d_users)
        logging.debug('%s fsgraph data is done.' % week)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()