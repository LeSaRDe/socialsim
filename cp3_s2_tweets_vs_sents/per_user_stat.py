import json
import logging
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils


g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
            '20180527_20180603', '20180610_20180617', '20180624_20180701',
            '20180708_20180715', '20180722_20180729', '20180805_20180812',
            '20180819_20180826']
#'I_Wry8ROzibHG44UBImpiQ', 'a0fYqjn3qCvgH6MYNqZRew', 'fDk3wnVbzNUk9l-47tt8UQ',
g_l_top_users= [ 'b0Z0FKyh_ciPQJx0Onzbqg']
g_in_data_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/tw_wh_data_{1}_10_sample_dict.json'
g_output_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/tw_wh_data_10_sample_top_user_stat_{0}.json'



def main():
    # d_user_stat = dict()
    for user_id in g_l_top_users:
        d_per_user_stat = dict()
        for week in g_l_weeks:
            d_per_user_stat[week] = [0, 0, 0, 0, 0, 0, 0]
            with open(g_in_data_format.format(week, week), 'r') as in_fd:
                tw_json = json.load(in_fd)
                tw_i = 0
                for tw_id in tw_json:
                    if tw_json[tw_id]['user']['id_str_h'] != user_id:
                        continue
                    if data_preprocessing_utils.get_tweet_type(tw_json[tw_id]) == 't':
                        d_per_user_stat[week][0] += 1
                    elif data_preprocessing_utils.get_tweet_type(tw_json[tw_id]) == 'q':
                        d_per_user_stat[week][1] += 1
                    elif data_preprocessing_utils.get_tweet_type(tw_json[tw_id]) == 'r':
                        d_per_user_stat[week][2] += 1
                    elif data_preprocessing_utils.get_tweet_type(tw_json[tw_id]) == 'n':
                        d_per_user_stat[week][3] += 1
                    else:
                        raise Exception('Wrong type of tweet occurs!')
                    if 'extension' in tw_json[tw_id] and 'sentiment_scores' in tw_json[tw_id]['extension']:
                        l_sents = json.loads(tw_json[tw_id]['extension']['sentiment_scores'])
                    else:
                        logging.debug('This tweet does not have extension or sentiment_scores: %s' % tw_json[tw_id])
                        l_sents = {'positive': 0, 'neutral': 0, 'negative': 0}
                        continue
                    if l_sents is None or len(l_sents) != 3:
                        raise Exception('No sentiment available to this tweet: %s' % tw_id)
                    d_per_user_stat[week][4] += l_sents['positive']
                    d_per_user_stat[week][5] += l_sents['neutral']
                    d_per_user_stat[week][6] += l_sents['negative']
                    tw_i += 1
                if tw_i > 0:
                    d_per_user_stat[week][4] = d_per_user_stat[week][4] / tw_i
                    d_per_user_stat[week][5] = d_per_user_stat[week][5] / tw_i
                    d_per_user_stat[week][6] = d_per_user_stat[week][6] / tw_i
            in_fd.close()
        # d_user_stat[user_id] = d_per_user_stat
        with open(g_output_format.format(user_id), 'w+') as out_fd:
            for week in d_per_user_stat:
                line = week + ' '
                values = ' '.join([str(field) for field in d_per_user_stat[week]])
                line += values
                out_fd.write(line + '\n')
        out_fd.close()



if __name__ == '__main__':
    main()