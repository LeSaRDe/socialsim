import json
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import time
import logging

g_json_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_10000_rand.json'
g_output_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_10000_rand_rets.txt'

def read_json(file_path):
    fd = open(file_path, 'r')
    l_lines = fd.readlines()
    l_json_data = []
    for line in l_lines:
        json_data = json.loads(line)
        l_json_data.append(json_data)
    fd.close()
    return l_json_data


def vec_comparison(l_vecs):
    l_comparison_rets = []
    start = time.time()
    for i in range(0, len(l_vecs)-1):
        vec_i = np.asarray(l_vecs[i])
        for j in range(i, len(l_vecs)):
            vec_j = np.asarray(l_vecs[j])
            sim = 1.0 - cosine(vec_i, vec_j)
            l_comparison_rets.append(sim)
            if len(l_comparison_rets) % 10000 == 0:
                logging.debug('vec_comparison: %s finished in %s sec.' % (len(l_comparison_rets), time.time() - start))
    logging.debug('vec_comparison: %s finished in %s sec.' % (len(l_comparison_rets), time.time() - start))
    return l_comparison_rets


def collect_topic_vecs(l_json_data):
    l_topic_vecs = []
    for i in range(0, len(l_json_data)):
        json_data = l_json_data[i]
        if 'socialsim_topic_vector' in json_data['extension']:
            l_topic_vecs.append(json_data['extension']['socialsim_topic_vector'])
    return l_topic_vecs


def main():
    l_json_data = read_json(g_json_path)
    # l_topic_vecs = [json_data['extension']['socialsim_topic_vector'] for json_data in l_json_data]
    l_topic_vecs = collect_topic_vecs(l_json_data)
    logging.debug('%s topic vectors are found.' % len(l_topic_vecs))
    l_comparison_rets = vec_comparison(l_topic_vecs)
    l_comparison_rets_str = [str(ret) for ret in l_comparison_rets]
    str_comparison_rets = ' '.join(l_comparison_rets_str)
    with open(g_output_path, 'w+') as fd_out:
        fd_out.write(str_comparison_rets)
    fd_out.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('start...')
    main()