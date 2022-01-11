import json
import logging


g_raw_data_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/20180401_20180408/tw_wh_data_20180401_20180408_timesorted.json'
g_output_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/20180401_20180408/tw_wh_data_20180401_20180408_timesorted_dict.json'

def main():
    d_output = dict()
    with open(g_raw_data_file_path, 'r') as in_fd:
        l_lines = in_fd.readlines()
        for i in range(0, len(l_lines)):
            json_data = json.loads(l_lines[i])
            id_t = json_data['id_str_h']
            if id_t not in d_output:
                d_output[id_t] = json_data
            else:
                logging.error('Conflict tweet id exists! %s' % id_t)
            if i % 5000 == 0 and i >= 5000:
                logging.debug('%s tweets have been scanned.' % i)
        logging.debug('All tweets have been scanned.')
    in_fd.close()
    with open(g_output_file_path, 'w+') as out_fd:
        json.dump(d_output, out_fd, indent=4)
    out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()