import logging
import sqlite3
import json
import networkx as nx
import matplotlib.pyplot as plt
import data_preprocessing_utils

g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_raw_tw_data_path = g_path_prefix + 'Tng_an_WH_Twitter_v3.json'
g_cascade_file_path = g_path_prefix + 'wh_tw_v3_cscd.json'
g_cascade_graph_path = g_path_prefix + 'wh_tw_v3_cscd_graph.gml'
g_salient_cascade_file_path = g_path_prefix + 'wh_tw_v3_sal_cscd.json'
g_salient_cascade_graph_path = g_path_prefix + 'wh_tw_v3_sal_cscd_graph.gml'
# g_tw_salience_file_path = g_path_prefix + 'wh_tw_v3_tw_sal.json'
g_salient_tw_text_file_path = g_path_prefix + 'wh_tw_v3_sal_tw_txt.json'

g_cp4_tw_raw_path_prefix = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
g_cp4_tw_raw_files = ['cp4.venezuela.twitter.training.anon.v1.2018-dec.json',
                      'cp4.venezuela.twitter.training.anon.v1.2019-jan-part1.json',
                      'cp4.venezuela.twitter.training.anon.v1.2019-jan-part2.json',
                      'cp4.venezuela.twitter.training.anon.v1.2019-jan-part3.json',
                      'cp4.venezuela.twitter.training.anon.v1.2019-jan-part4.json',
                      'cp4.venezuela.twitter.training.anon.v1.2019-jan-part5.json']


def extract_cascades():
    d_tw = dict()
    with open(g_raw_tw_data_path, 'r') as in_fd:
        tw_ln = in_fd.readline()
        count = 1
        while tw_ln:
            tw_json = json.loads(tw_ln)
            tw_id = tw_json['id_str_h']
            tw_type = data_preprocessing_utils.get_tweet_type(tw_json)
            tw_src_id = None
            if tw_type == 'r':
                tw_src_id = tw_json['in_reply_to_status_id_str_h']
            elif tw_type == 'q':
                if 'quoted_status' in tw_json:
                    tw_src_id = tw_json['quoted_status']['id_str_h']
                else:
                    logging.debug('%s is a quote but has no quoted_status.' % tw_id)
            if tw_id not in d_tw:
                like_cnt = tw_json['favorite_count']
                d_tw[tw_id] = {'r': [], 'q': [], 'sal': like_cnt}
                logging.debug('likes = %s' % like_cnt)
            if tw_src_id is not None and tw_src_id in d_tw:
                if tw_type not in d_tw[tw_src_id]:
                    logging.debug('%s has incorrect records.' % tw_id)
                else:
                    d_tw[tw_src_id][tw_type].append(tw_id)
            tw_ln = in_fd.readline()
            count += 1
            if count % 10000 == 0 and count >= 10000:
                logging.debug('%s tweets have been scanned.' % count)
        logging.debug('%s tweets have been scanned.' % count)
        in_fd.close()

    for tw_id in d_tw:
        d_tw[tw_id]['sal'] += len(d_tw[tw_id]['r'])
        d_tw[tw_id]['sal'] += len(d_tw[tw_id]['q'])

    l_del_tw_ids = []
    for tw_id in d_tw.keys():
        if len(d_tw[tw_id]['r']) + len(d_tw[tw_id]['q']) == 0:
            l_del_tw_ids.append(tw_id)
    for tw_id in l_del_tw_ids:
        del d_tw[tw_id]
    logging.debug('Cascades have been extracted.')

    with open(g_cascade_file_path, 'w+') as out_fd:
        json.dump(d_tw, out_fd)
        out_fd.close()
    logging.debug('Cascade json is done.')


def extract_salient_cascades():
    with open(g_cascade_file_path, 'r') as in_fd:
        d_cscd = json.load(in_fd)
        in_fd.close()
    l_del = []
    for tw_id in d_cscd:
        sal = d_cscd[tw_id]['sal']
        if sal < 10:
            l_del.append(tw_id)
    for del_id in l_del:
        del d_cscd[del_id]
    # TODO
    # Need to uncomment the following block
    #
    # l_del = []
    # for tw_id in d_cscd:
    #     old_sal = d_cscd[tw_id]['sal']
    #     l_del_rpl = []
    #     for rpl_id in d_cscd[tw_id]['r']:
    #         if rpl_id not in d_cscd:
    #             l_del_rpl.append(rpl_id)
    #     d_cscd[tw_id]['r'] = [ele for ele in d_cscd[tw_id]['r'] if ele not in l_del_rpl]
    #     l_del_qt = []
    #     for qt_it in d_cscd[tw_id]['q']:
    #         if qt_it not in d_cscd:
    #             l_del_qt.append(qt_it)
    #     d_cscd[tw_id]['q'] = [ele for ele in d_cscd[tw_id]['q'] if ele not in l_del_qt]
    #     if old_sal - len(l_del_rpl) - len(l_del_qt) < 5:
    #         l_del.append(tw_id)
    # for del_id in l_del:
    #     del d_cscd[del_id]
    #
    with open(g_salient_cascade_file_path, 'w+') as out_fd:
        json.dump(d_cscd, out_fd)
        out_fd.close()
    print()


def build_cascade_graph():
    with open(g_cascade_file_path, 'r') as in_fd:
        d_cscd = json.load(in_fd)
        in_fd.close()
    g_cscd = nx.DiGraph()
    for tw_id in d_cscd:
        for rpl_id in d_cscd[tw_id]['r']:
            g_cscd.add_edge(tw_id, rpl_id, type='r')
        for qt_id in d_cscd[tw_id]['q']:
            g_cscd.add_edge(tw_id, qt_id, type='q')
    nx.write_gml(g_cscd, g_cascade_graph_path)
    logging.debug('Cascade graph is done.')


def draw_cascade_graph():
    g_cscd = nx.read_gml(g_cascade_graph_path)
    # erpl = [(u, v) for (u, v, d) in g_cscd.edges(data=True) if d['type'] == 'r']
    # eqt = [(u, v) for (u, v, d) in g_cscd.edges(data=True) if d['type'] == 'q']
    # pos = nx.spring_layout(g_cscd)
    # nx.draw_networkx_nodes(g_cscd, pos, node_size=100)
    # nx.draw_networkx_edges(g_cscd, pos, edgelist=erpl, edge_color='r', width=2)
    # nx.draw_networkx_edges(g_cscd, pos, edgelist=eqt, edge_color='b', width=2)
    # nx.draw_networkx_labels(g_cscd, pos, font_size=15, font_family='sans-serif')
    nx.draw(g_cscd)
    # plt.axis('off')
    plt.show()


def get_sorted_cascades():
    with open(g_salient_cascade_file_path, 'r') as in_fd:
        d_cscd = json.load(in_fd)
        in_fd.close()
    l_sorted_cscd = sorted(d_cscd.keys(), key=lambda k: d_cscd[k]['sal'], reverse=True)
    return l_sorted_cscd, d_cscd


def find_salient_tw():
    d_tw = dict()
    cnt = 0
    for file in g_cp4_tw_raw_files:
        with open(g_cp4_tw_raw_path_prefix + file, 'r') as in_fd:
            ln = in_fd.readline()
            while ln:
                tw_json = json.loads(ln)
                tw_id = tw_json['id_str_h']
                if tw_id not in d_tw:
                    d_tw[tw_id] = dict()
                    d_tw[tw_id]['r'] = tw_json['reply_count']
                    d_tw[tw_id]['q'] = tw_json['quote_count']
                    d_tw[tw_id]['t'] = tw_json['retweet_count']
                    d_tw[tw_id]['r_q'] = d_tw[tw_id]['r'] + d_tw[tw_id]['q']
                    d_tw[tw_id]['all'] = d_tw[tw_id]['r'] + d_tw[tw_id]['q'] + d_tw[tw_id]['t']
                    cnt += 1
            in_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # extract_cascades()
    # build_cascade_graph()
    # draw_cascade_graph()
    # extract_salient_cascades()
    l_sorted_cscd, d_cscd = get_sorted_cascades()
