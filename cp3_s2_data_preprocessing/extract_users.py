import json
import logging


g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
            '20180527_20180603', '20180610_20180617', '20180624_20180701',
            '20180708_20180715', '20180722_20180729', '20180805_20180812',
            '20180819_20180826']
g_path_prefix_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/'
g_raw_data_file_path_format = g_path_prefix_format + 'tw_wh_data_{1}_timesorted_dict.json'
g_output_file_path_format = g_path_prefix_format + 'users_{1}_full.txt'


def extract_users(task_name):
    d_users = dict()
    with open(g_raw_data_file_path_format.format(task_name, task_name), 'r') as in_fd:
        tw_dict = json.load(in_fd)
        for tw_id in tw_dict:
            user_id = tw_dict[tw_id]['user']['id_str_h']
            if user_id not in d_users:
                d_users[user_id] = None
    in_fd.close()

    with open(g_output_file_path_format.format(task_name, task_name), 'w+') as out_fd:
        l_users = list(d_users.keys())
        out_fd.write('\n'.join(l_users))
    out_fd.close()
    logging.debug('%s is done.' % task_name)


def main():
    for week in g_l_weeks:
        extract_users(week)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()