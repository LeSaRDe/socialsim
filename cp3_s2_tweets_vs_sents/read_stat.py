import json


g_stat_data_file_path = 'Tng_an_WH_Twitter_v2_stat.json'
g_out_file_path = 'Tng_an_WH_Twitter_v2_stat_out.txt'


def main():
    with open(g_stat_data_file_path, 'r') as in_fd:
        json_data = json.load(in_fd)
        with open(g_out_file_path, 'w+') as out_fd:
            for item in json_data:
                line = item + ' '
                values = ' '.join([str(field) for field in json_data[item]])
                line += values
                out_fd.write(line + '\n')

        out_fd.close()
    in_fd.close()


if __name__ == '__main__':
    main()