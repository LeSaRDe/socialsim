import json
import logging
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils
import time
from datetime import datetime, timedelta

g_twitter_data_sorted_by_time_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_by_time_sorted.json'
g_sampled_output_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_20180401_20180408_by_time_sorted.json'

g_start = '20180401000000'
g_end = (datetime.strptime(g_start, '%Y%m%d%H%M%S') + timedelta(days=7)).strftime('%Y%m%d%H%M%S')


def main():
    with open(g_twitter_data_sorted_by_time_file_path, 'r') as in_fd:
        t_start = time.time()
        line = in_fd.readline()
        count = 0
        l_samples = []
        while line:
            json_data = json.loads(line)
            time_str = data_preprocessing_utils.get_user_time(json_data['created_at'].split(' '))
            if time_str > g_end:
                break
            if g_start <= time_str <= g_end:
                l_samples.append(line)
                count += 1
                if count % 5000 == 0 and count > 5000:
                    logging.debug('%s lines have been scanned in %s seconds.' % (count, time.time()-t_start))
            line = in_fd.readline()
    in_fd.close()
    logging.debug('%s lines in total is sampled in %s seconds.' % (len(l_samples), time.time()-t_start))

    if len(l_samples) > 0:
        with open(g_sampled_output_file_path, 'w+') as out_fd:
            out_fd.writelines(l_samples)
        out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()