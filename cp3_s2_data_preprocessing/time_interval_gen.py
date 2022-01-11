import logging
from datetime import datetime, timedelta
import json

g_time_int_output_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/time_series_data/time_intervals.txt'
g_time_int_idx_map_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/time_series_data/time_int_idx_map.json'

g_time_stride = timedelta(days=3)
g_time_int_his = timedelta(days=3)
g_time_int_fut = timedelta(days=3)


def time_int_idx_map():
    with open(g_time_int_output_path, 'r') as in_fd:
        with open(g_time_int_idx_map_path, 'w+') as out_fd:
            d_time_int_idx_map = dict()
            time_int_str = in_fd.readline().strip()
            idx = 0
            while time_int_str:
                d_time_int_idx_map[time_int_str] = idx
                time_int_str = in_fd.readline().strip()
                idx += 1
            json.dump(d_time_int_idx_map, out_fd)
            out_fd.close()
        in_fd.close()


# Input: format is YYYYMMDD
# Output: a list of intervals of the format: YYYYMMDD_YYYYMMDD (i.e. start_end)
def main(start_day_str, end_day_str):
    if start_day_str > end_day_str:
        logging.error('Start day is later then end day.')
        return None
    start_day = datetime.strptime(start_day_str, '%Y%m%d')
    end_day = datetime.strptime(end_day_str, '%Y%m%d')
    l_time_ints = []
    cur_day = start_day + g_time_int_his
    while cur_day <= end_day:
        cur_int_start = cur_day - g_time_int_his
        cur_int_end = cur_day + g_time_int_fut
        cur_int_start_str = cur_int_start.strftime('%Y%m%d').strip()
        cur_int_end_str = cur_int_end.strftime('%Y%m%d').strip()
        l_time_ints.append([cur_int_start_str, cur_int_end_str])
        cur_day += g_time_stride
    logging.debug('%s time intervals are generated.' % len(l_time_ints))
    with open(g_time_int_output_path, 'w+') as out_fd:
        l_time_ints_str_ele = [t_int[0] + ':' + t_int[1] for t_int in l_time_ints]
        out_fd.write('\n'.join(l_time_ints_str_ele))
        out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # main('20180401', '20190501')
    time_int_idx_map()