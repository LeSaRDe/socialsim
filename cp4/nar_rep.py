import json
import logging
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils
import networkx as nx
import igraph as ig
import sqlite3


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_tw_data_name = 'Tng_an_WH_Twitter_v3.json'
g_all_resp_out = 'all_responses.txt'
g_top_resp_out = 'top_responded_tweets.txt'
g_top_resp_db = 'top_responded_tweets.db'
g_top_tw_graph = 'top_tw_graph.gml'
g_top_tw_graph_img_path = 'top_tw_graph.pdf'
g_top_tw_graph_d3_data = 'top_tw_graph_d3.json'


# replies and quotes are responses
def obtain_top_responded_tweets(en_out=False):
    d_tw = dict()
    with open(g_path_prefix + g_tw_data_name, 'r') as in_fd:
        tw_ln = in_fd.readline()
        count = 1
        while tw_ln:
            if count % 5000 == 0 and count >= 5000:
                logging.debug('%s tweets have been scanned.' % count)
            tw_json = json.loads(tw_ln)
            tw_id = tw_json['id_str_h']
            if tw_id not in d_tw:
                d_tw[tw_id] = dict()
                d_tw[tw_id]['r'] = []
                d_tw[tw_id]['q'] = []
            tw_type = data_preprocessing_utils.get_tweet_type(tw_json)
            if tw_type == 'r':
                tw_src_id = tw_json['in_reply_to_status_id_str_h']
                if tw_src_id not in d_tw:
                    d_tw[tw_src_id] = dict()
                    d_tw[tw_src_id]['r'] = [tw_id]
                    d_tw[tw_src_id]['q'] = []
                else:
                    d_tw[tw_src_id]['r'].append(tw_id)
            elif tw_type == 'q':
                tw_src_id = tw_json['quoted_status']['id_str_h']
                if tw_src_id not in d_tw:
                    d_tw[tw_src_id] = dict()
                    d_tw[tw_src_id]['q'] = [tw_id]
                    d_tw[tw_src_id]['r'] = []
                else:
                    d_tw[tw_src_id]['q'].append(tw_id)
            else:
                tw_ln = in_fd.readline()
                continue
            tw_ln = in_fd.readline()
        in_fd.close()

    if en_out:
        with open(g_all_resp_out, 'w+') as out_fd:
            json.dump(d_tw, out_fd)
            out_fd.close()
        logging.debug('top_responded_tweets is written.')


def retrieve_top_responded_tweets(en_out=False):
    with open(g_all_resp_out, 'r') as in_fd:
        top_resp_json = json.load(in_fd)
        l_top_resp_json = sorted([[tid, top_resp_json[tid]['r'], top_resp_json[tid]['q']] for tid in top_resp_json],
                                 key=lambda k : len(k[1])+len(k[2]), reverse=True)
        in_fd.close()

    l_top_resp_sorted = []
    for top_resp in l_top_resp_json:
        l_tw_r = top_resp[1]
        l_tw_q = top_resp[2]
        l_tw_r_sorted = sorted(l_tw_r, key=lambda k : len(top_resp_json[k]['r']) + len(top_resp_json[k]['q']), reverse=True)
        l_tw_q_sorted = sorted(l_tw_q, key=lambda k : len(top_resp_json[k]['r']) + len(top_resp_json[k]['q']), reverse=True)
        l_top_resp_sorted.append([top_resp[0], l_tw_r_sorted, l_tw_q_sorted])

    if en_out:
        with open(g_top_resp_out, 'w+') as out_fd:
            for tw in l_top_resp_sorted:
                ln = tw[0] + '|' + ' '.join(tw[1]) + '|' + ' '.join(tw[2])
                out_fd.write(ln)
                out_fd.write('\n')
            out_fd.close()

    return l_top_resp_sorted


def read_top_resp_to_db():
    with open(g_top_resp_out, 'r') as in_fd:
        db_conn = sqlite3.connect(g_top_resp_db)
        db_cur = db_conn.cursor()
        ln = in_fd.readline()
        sql_str = '''insert into tw_top_resp values (?, ?, ?, ?)'''
        count = 1
        while ln:
            l_fields = ln.strip().split('|')
            tw_src = l_fields[0]
            tw_replies = l_fields[1]
            tw_quotes = l_fields[2]
            tw_resp_cnt = len(tw_replies.split(' ')) + len(tw_quotes.split(' '))
            db_cur.execute(sql_str, (tw_src, tw_replies, tw_quotes, tw_resp_cnt))
            count += 1
            if count % 5000 == 0 and count >= 5000:
                db_conn.commit()
                logging.debug('%s top resp records have been written.' % count)
            ln = in_fd.readline()
        db_conn.commit()
        logging.debug('%s top resp records have been written.' % count)
        in_fd.close()
        db_conn.close()


def retrieve_replies_n_quotes(tid):
    db_conn = sqlite3.connect(g_top_resp_db)
    db_cur = db_conn.cursor()
    sql_str = '''select l_replies, l_quotes, total_resp from tw_top_resp where tid=?'''
    db_cur.execute(sql_str, (tid,))
    rec = db_cur.fetchone()
    if rec[0] is '':
        l_replies = []
    else:
        l_replies = [reply.strip() for reply in rec[0].strip().split(' ')]
    if rec[1] is '':
        l_quotes = []
    else:
        l_quotes = [quote.strip() for quote in rec[1].strip().split(' ')]
    total_resp = rec[2]
    db_conn.close()
    return l_replies, l_quotes, total_resp


def build_top_resp_graph(sample_graph, tid, l_replies, l_quotes):
    # sample_graph.add_node(tid)
    for reply in l_replies:
        sample_graph.add_edge(tid, reply, type='r')
        l_r, l_q, cnt = retrieve_replies_n_quotes(reply)
        logging.debug('%s responses to %s.' % (cnt, reply))
        sample_graph = build_top_resp_graph(sample_graph, reply, l_r, l_q)

    for quote in l_quotes:
        sample_graph.add_edge(tid, quote, type='q')
        l_r, l_q, cnt = retrieve_replies_n_quotes(quote)
        logging.debug('%s responses to %s.' % (cnt, quote))
        sample_graph = build_top_resp_graph(sample_graph, quote, l_r, l_q)

    return sample_graph


def retrieve_top_resp_sample(en_out=False):
    sample_graph = nx.DiGraph()
    db_conn = sqlite3.connect(g_top_resp_db)
    db_cur = db_conn.cursor()
    sql_str = '''select tid from tw_top_resp order by total_resp desc limit 1'''
    db_cur.execute(sql_str)
    rec = db_cur.fetchone()
    tid = rec[0]
    l_replies, l_quotes, total_resp = retrieve_replies_n_quotes(tid)
    logging.debug('%s responses to %s.' % (total_resp, tid))
    sample_graph = build_top_resp_graph(sample_graph, tid, l_replies, l_quotes)

    if en_out:
        nx.write_gml(sample_graph, g_top_tw_graph)

    return sample_graph


def plot_sample_graph():
    ig_tw = ig.load(g_top_tw_graph)
    ig.plot(ig_tw, g_top_tw_graph_img_path, bbox=(7680, 7680), vertex_label_size=0, margin=50)


def sample_graph_to_d3_json():
    sample_graph = nx.read_gml(g_top_tw_graph)
    nodes = [{'name': node} for node in sample_graph.nodes()]
    links = [{'source': edge[0], 'target': edge[1]} for edge in sample_graph.edges()]
    with open(g_top_tw_graph_d3_data, 'w+') as out_fd:
        json.dump({'nodes': nodes, 'links': links}, out_fd, indent=4)
        out_fd.close()


def main():
    logging.basicConfig(level=logging.DEBUG)
    # obtain_top_responded_tweets(True)
    # retrieve_top_responded_tweets(True)
    # read_top_resp_to_db()
    # sample_graph = retrieve_top_resp_sample(True)
    # plot_sample_graph()
    sample_graph_to_d3_json()


if __name__ == '__main__':
    main()