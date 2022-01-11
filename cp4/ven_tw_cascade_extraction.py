import json
import sqlite3
import logging
from datetime import datetime
from os import walk, path
import os
import networkx as nx
import matplotlib.pyplot as plt
import time


version = 'v2-1'
g_ven_tw_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
g_ven_tw_rawdata_folder = g_ven_tw_folder + 'tw_raw_data/'
g_ven_tw_reply_quote_cas_path = g_ven_tw_folder + 'cp4.venezuela.twitter.training.anon.v2-1.reply-cascades.json'
g_ven_tw_data_list = ['cp4.venezuela.twitter.training.anon.v2-1.2018-dec.json',
                      'cp4.venezuela.twitter.training.anon.v2-1.2019-jan-part1.json',
                      'cp4.venezuela.twitter.training.anon.v2-1.2019-jan-part2.json',
                      'cp4.venezuela.twitter.training.anon.v2-1.2019-jan-part3.json',
                      'cp4.venezuela.twitter.training.anon.v2-1.2019-jan-part4.json',
                      'cp4.venezuela.twitter.training.anon.v2-1.2019-jan-part5.json',
                      'cp4.venezuela.twitter.training.anon.v2-1.reply-cascades.json']
g_ven_tw_db = g_ven_tw_folder + 'ven_tw_en_' + version + '.db'
g_ven_tw_cas_db = g_ven_tw_folder + 'ven_tw_en_cas_' + version + '.db'
g_ven_tw_cas_full_graph = g_ven_tw_folder + 'ven_tw_en_cas_full_graph_' + version + '.gml'
g_ven_tw_cas_folder = g_ven_tw_folder + 'cascades/'
g_ven_tw_cas_graph_format = g_ven_tw_cas_folder + '{0}_cas.gml'
g_ven_tw_cas_graph_img_format = g_ven_tw_cas_folder + '{0}_cas.png'
g_ven_orig_en_tw_ids_path = g_ven_tw_folder + 'ven_tw_orig_en_ids_' + version + '.txt'
g_ven_cas_graph_stats_format = g_ven_tw_folder + 'ven_en_tw_cas_stats_{0}_' + version + '.txt'
g_ven_tw_resp_by_users_db = g_ven_tw_folder + 'ven_tw_resp_' + version + '.db'


def get_tw_type(tw_json):
    if 'in_reply_to_status_id_str_h' in tw_json \
            and tw_json['in_reply_to_status_id_str_h'] != '' \
            and not tw_json['in_reply_to_status_id_str_h'] is None:
        t_type = 'r'
    elif 'retweeted_status' in tw_json:
        t_type = 't'
    elif 'quoted_status' in tw_json:
        t_type = 'q'
    else:
        t_type = 'n'
    return t_type


def get_tw_lang(tw_json):
    return tw_json['lang']


def get_tw_usr(tw_json):
    return tw_json['user']['id_str_h']


def get_tw_id(tw_json):
    return tw_json['id_str_h']


def get_tw_orig_raw_txt(tw_json, tw_type):
    if tw_type == 'n' or tw_type == 'r' or tw_type == 'q':
        if 'full_text_m' in tw_json:
            return tw_json['full_text_m']
        else:
            return tw_json['text_m']


def get_tw_raw_txt(tw_json, tw_type, tw_lang):
    if tw_type == 'n' or tw_type == 'r' or tw_type == 'q':
        if tw_lang == 'en':
            if 'full_text_m' in tw_json:
                return tw_json['full_text_m']
            else:
                return tw_json['text_m']
        else:
            if 'google_translation_m' in tw_json['extension']:
                return tw_json['extension']['google_translation_m']
            else:
                return None
    elif tw_type == 't':
        if tw_lang == 'en':
            if 'full_text_m' in tw_json['retweeted_status']:
                return tw_json['retweeted_status']['full_text_m']
            else:
                return tw_json['retweeted_status']['text_m']
        else:
            if 'google_translation_m' in tw_json['extension']:
                return tw_json['extension']['google_translation_m']
            else:
                return None
    # elif tw_type == 'q':
    #     if 'full_text_m' in tw_json['quoted_status']:
    #         return tw_json['quoted_status']['full_text_m']
    #     else:
    #         return tw_json['quoted_status']['text_m']
    else:
        return None


def get_tw_src(tw_json, tw_type):
    src_id = None
    if tw_type == 'n':
        return None
    elif tw_type == 'r':
        src_id = tw_json['in_reply_to_status_id_str_h']
    elif tw_type == 'q':
        src_id = tw_json['quoted_status_id_str_h']
    elif tw_type == 't':
        src_id = tw_json['retweeted_status']['id_str_h']
    return src_id


def translate_month(month_str):
    month = None
    if month_str == 'Jan':
        month = '01'
    elif month_str == 'Feb':
        month = '02'
    elif month_str == 'Mar':
        month = '03'
    elif month_str == 'Apr':
        month = '04'
    elif month_str == 'May':
        month = '05'
    elif month_str == 'Jun':
        month = '06'
    elif month_str == 'Jul':
        month = '07'
    elif month_str == 'Aug':
        month = '08'
    elif month_str == 'Sep':
        month = '09'
    elif month_str == 'Oct':
        month = '10'
    elif month_str == 'Nov':
        month = '11'
    elif month_str == 'Dec':
        month = '12'
    else:
        logging.error('Wrong month exists! user_time = %s' % month_str)
        raise Exception('Wrong month exists! user_time = %s' % month_str)
    return month


def get_tw_datetime(tw_json):
    date_fields = [item.strip() for item in tw_json['created_at'].split(' ')]
    mon_str = translate_month(date_fields[1])
    day_str = date_fields[2]
    year_str = date_fields[5]
    time_str = ''.join([item.strip() for item in date_fields[3].split(':')])
    return year_str + mon_str + day_str + time_str


def find_all_en_tw():
    timer_start = time.time()
    db_conn = sqlite3.connect(g_ven_tw_db)
    db_cur = db_conn.cursor()
    sql_str = '''create table if not exists ven_tw_en (tw_id text primary key, usr_id text not null, tw_type text not null, tw_src_id text, tw_datetime text, raw_txt text)'''
    db_cur.execute(sql_str)
    s_rec_ids = set([])

    total_cnt = 0
    rec_cnt = 0
    sql_str = '''insert into ven_tw_en (tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt) values (?, ?, ?, ?, ?, ?)'''
    for data_file in g_ven_tw_data_list:
        data_path = g_ven_tw_folder + data_file
        with open(data_path, 'r') as in_fd:
            for tw_ln in in_fd:
                total_cnt += 1
                tw_json = json.loads(tw_ln)
                tw_lang = get_tw_lang(tw_json)
                tw_id = get_tw_id(tw_json)
                if tw_id in s_rec_ids:
                    continue
                if tw_lang == 'en' \
                        or (tw_lang != 'en'
                            and 'google_translation_m' in tw_json['extension']
                            and tw_json['extension']['google_translation_m'] is not None):
                    tw_usr = get_tw_usr(tw_json)
                    tw_type = get_tw_type(tw_json)
                    tw_src_id = get_tw_src(tw_json, tw_type)
                    tw_datetime = get_tw_datetime(tw_json)
                    tw_raw_txt = get_tw_raw_txt(tw_json, tw_type, tw_lang)
                    db_cur.execute(sql_str, (tw_id, tw_usr, tw_type, tw_src_id, tw_datetime, tw_raw_txt))
                    s_rec_ids.add(tw_id)
                    rec_cnt += 1
                if rec_cnt % 10000 == 0 and rec_cnt >= 10000:
                    db_conn.commit()
                    logging.debug('%s recs out of %s tws.' % (rec_cnt, total_cnt))
            in_fd.close()
            db_conn.commit()
            logging.debug('%s recs out of %s tws.' % (rec_cnt, total_cnt))
    logging.debug('%s recs out of %s tws in %s secs.' % (rec_cnt, total_cnt, str(time.time() - timer_start)))
    db_conn.close()


def add_retweet_texts():
    timer_start = time.time()
    db_conn = sqlite3.connect(g_ven_tw_db)
    db_cur = db_conn.cursor()
    s_rec_ids = set([])
    sql_str = '''select tw_id from ven_tw_en'''
    db_cur.execute(sql_str)
    l_recs = db_cur.fetchall()
    for rec in l_recs:
        tw_id = rec[0]
        s_rec_ids.add(tw_id)

    total_cnt = 0
    rec_cnt = 0
    insert_sql_str = '''insert into ven_tw_en (tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt) values (?, ?, ?, ?, ?, ?)'''
    update_sql_str = '''update ven_tw_en set raw_txt = ? where tw_id = ?'''
    for data_file in g_ven_tw_data_list:
        data_path = g_ven_tw_rawdata_folder + data_file
        with open(data_path, 'r') as in_fd:
            for tw_ln in in_fd:
                total_cnt += 1
                tw_json = json.loads(tw_ln)
                tw_type = get_tw_type(tw_json)
                if tw_type != 't':
                    continue
                tw_lang = get_tw_lang(tw_json)
                tw_id = get_tw_id(tw_json)
                if tw_lang == 'en' \
                        or (tw_lang != 'en'
                            and 'google_translation_m' in tw_json['extension']
                            and tw_json['extension']['google_translation_m'] is not None):
                    tw_usr = get_tw_usr(tw_json)
                    tw_src_id = get_tw_src(tw_json, tw_type)
                    tw_datetime = get_tw_datetime(tw_json)
                    tw_raw_txt = get_tw_raw_txt(tw_json, tw_type, tw_lang)
                    if tw_id in s_rec_ids:
                        db_cur.execute(update_sql_str, (tw_raw_txt, tw_id))
                    else:
                        db_cur.execute(insert_sql_str, (tw_id, tw_usr, tw_type, tw_src_id, tw_datetime, tw_raw_txt))
                        s_rec_ids.add(tw_id)
                    rec_cnt += 1
                if rec_cnt % 10000 == 0 and rec_cnt >= 10000:
                    db_conn.commit()
                    logging.debug('%s recs out of %s tws.' % (rec_cnt, total_cnt))
            in_fd.close()
            db_conn.commit()
            logging.debug('%s recs out of %s tws.' % (rec_cnt, total_cnt))
    logging.debug('%s recs out of %s tws in %s secs.' % (rec_cnt, total_cnt, str(time.time() - timer_start)))
    db_conn.close()


def find_all_orig_en_tw_ids():
    s_en_tw_ids = set([])
    total_cnt = 0
    rec_cnt = 0
    for data_file in g_ven_tw_data_list:
        data_path = g_ven_tw_folder + data_file
        with open(data_path, 'r') as in_fd:
            for tw_ln in in_fd:
                total_cnt += 1
                tw_json = json.loads(tw_ln)
                tw_lang = get_tw_lang(tw_json)
                tw_id = get_tw_id(tw_json)
                if tw_id in s_en_tw_ids:
                    continue
                if tw_lang == 'en':
                    s_en_tw_ids.add(tw_id)
                    rec_cnt += 1
                if rec_cnt % 10000 == 0 and rec_cnt >= 10000:
                    logging.debug('%s recs out of %s tws.' % (rec_cnt, total_cnt))
            in_fd.close()
    logging.debug('%s recs out of %s tws.' % (rec_cnt, total_cnt))
    with open(g_ven_orig_en_tw_ids_path, 'w+') as out_fd:
        out_str = '\n'.join(s_en_tw_ids)
        out_fd.write(out_str)
        out_fd.close()


def find_all_en_tw_for_cas():
    db_conn = sqlite3.connect(g_ven_tw_cas_db)
    db_cur = db_conn.cursor()
    sql_str = '''create table if not exists ven_tw_en (tw_id text primary key, usr_id text not null, tw_type text not null, tw_src_id text, tw_datetime text, raw_txt text, replies text, quotes text)'''
    db_cur.execute(sql_str)

    cnt = 0
    sql_str = '''insert into ven_tw_en (tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt, replies, quotes) values (?, ?, ?, ?, ?, ?, null, null)'''
    with open(g_ven_tw_reply_quote_cas_path, 'r') as in_fd:
        for tw_ln in in_fd:
            tw_json = json.loads(tw_ln)
            if get_tw_lang(tw_json) == 'en':
                tw_id = get_tw_id(tw_json)
                tw_usr = get_tw_usr(tw_json)
                tw_type = get_tw_type(tw_json)
                tw_src_id = get_tw_src(tw_json, tw_type)
                tw_datetime = get_tw_datetime(tw_json)
                tw_raw_txt = get_tw_raw_txt(tw_json, tw_type)
                db_cur.execute(sql_str, (tw_id, tw_usr, tw_type, tw_src_id, tw_datetime, tw_raw_txt))
                cnt += 1
            if cnt % 10000 == 0 and cnt >= 10000:
                db_conn.commit()
                logging.debug('%s tws.' % cnt)
        in_fd.close()
    db_conn.commit()
    logging.debug('%s tws.' % cnt)
    db_conn.close()


def merge_cas_db_to_en_db():
    en_db_conn = sqlite3.connect(g_ven_tw_db)
    en_db_cur = en_db_conn.cursor()
    en_insert_sql_str = '''insert into ven_tw_en (tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt) values (?, ?, ?, ?, ?, ?)'''
    en_query_sql_str = '''select tw_id from ven_tw_en where tw_id = ?'''
    cas_db_conn = sqlite3.connect(g_ven_tw_cas_db)
    cas_db_cur = cas_db_conn.cursor()
    cas_sql_str = '''select tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt from ven_tw_en'''
    cas_db_cur.execute(cas_sql_str)
    l_cas_recs = cas_db_cur.fetchall()
    cnt = 0
    for cas_rec in l_cas_recs:
        tw_id = cas_rec[0]
        usr_id = cas_rec[1]
        tw_type = cas_rec[2]
        tw_src_id = cas_rec[3]
        tw_datetime = cas_rec[4]
        raw_txt = cas_rec[5]
        en_db_cur.execute(en_query_sql_str, (tw_id,))
        en_rec = en_db_cur.fetchone()
        if en_rec is None:
            en_db_cur.execute(en_insert_sql_str, (tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt))
            cnt += 1
        if cnt % 1000 and cnt >= 1000:
            en_db_conn.commit()
            logging.debug('%s tws added.' % cnt)
    en_db_conn.commit()
    logging.debug('%s tws added.' % cnt)
    en_db_conn.close()
    cas_db_conn.close()


def update_replies_quotes():
    db_conn = sqlite3.connect(g_ven_tw_cas_db)
    db_cur = db_conn.cursor()
    update_sql_str = '''update ven_tw_en set replies = ?, quotes= ? where tw_id = ?'''
    query_sql_str = '''select tw_id, tw_src_id, tw_type from ven_tw_en where tw_type = "r" or tw_type = "q"'''
    check_sql_str = '''select tw_id, replies, quotes from ven_tw_en where tw_id = ?'''
    db_cur.execute(query_sql_str)
    replies_quotes = db_cur.fetchall()
    cnt = 0
    for resp in replies_quotes:
        resp_src_id = resp[1]
        db_cur.execute(check_sql_str, (resp_src_id,))
        check_rec = db_cur.fetchone()
        if check_rec is None:
            continue
        resp_id = resp[0]
        resp_type = resp[2]
        replies_str = check_rec[1]
        quotes_str = check_rec[2]
        if resp_type == 'r':
            if replies_str is None:
                l_replies = []
            else:
                l_replies = [item.strip() for item in replies_str.split('|')]
            if resp_id not in l_replies:
                l_replies.append(resp_id)
            if len(l_replies) == 0:
                replies_str = None
            else:
                replies_str = '|'.join(l_replies)
        elif resp_type == 'q':
            if quotes_str is None:
                l_quotes = []
            else:
                l_quotes = [item.strip() for item in quotes_str.split('|')]
            if resp_id not in l_quotes:
                l_quotes.append(resp_id)
            if len(l_quotes) == 0:
                quotes_str = None
            else:
                quotes_str = '|'.join(l_quotes)
        else:
            raise Exception('Unexpected resp_type: %s' % resp_type)
        db_cur.execute(update_sql_str, (replies_str, quotes_str, resp_src_id))
        cnt += 1
        if cnt % 1000 and cnt >= 1000:
            db_conn.commit()
            logging.debug('%s tws updated.' % cnt)
    db_conn.commit()
    logging.debug('%s tws updated.' % cnt)
    db_conn.close()


def find_all_en_cas():
    timer_start = time.time()
    cas_db_conn = sqlite3.connect(g_ven_tw_cas_db)
    cas_db_cur = cas_db_conn.cursor()
    cas_sql_str = '''create table if not exists ven_tw_en_cas (tw_id text primary key, replies text, quotes text)'''
    cas_db_cur.execute(cas_sql_str)

    cas_insert_sql_str = '''insert into ven_tw_en_cas (tw_id, replies, quotes) values (?, ?, ?)'''
    cas_update_sql_str = '''update ven_tw_en_cas set replies = ?, quotes= ? where tw_id = ?'''
    cas_query_sql_str = '''select tw_id, replies, quotes from ven_tw_en_cas where tw_id = ?'''

    en_db_conn = sqlite3.connect(g_ven_tw_db)
    en_db_cur = en_db_conn.cursor()
    en_query_sql_str = '''select tw_id, tw_src_id, tw_type from ven_tw_en where tw_type = "r" or tw_type = "q"'''
    en_check_sql_str = '''select tw_id from ven_tw_en where tw_id = ?'''

    cas_cnt = 0
    cas_tw_cnt = 0
    en_db_cur.execute(en_query_sql_str)
    en_recs = en_db_cur.fetchall()
    for en_rec in en_recs:
        tw_id = en_rec[0]
        tw_src_id = en_rec[1]
        tw_type = en_rec[2]
        cas_db_cur.execute(cas_query_sql_str, (tw_src_id,))
        cas_rec = cas_db_cur.fetchone()
        if cas_rec is None:
            en_db_cur.execute(en_check_sql_str, (tw_src_id,))
            en_check_rec = en_db_cur.fetchone()
            if en_check_rec is not None:
                if tw_type == 'r':
                    cas_db_cur.execute(cas_insert_sql_str, (tw_src_id, tw_id, None))
                    cas_cnt += 1
                    cas_tw_cnt += 1
                elif tw_type == 'q':
                    cas_db_cur.execute(cas_insert_sql_str, (tw_src_id, None, tw_id))
                    cas_cnt += 1
                    cas_tw_cnt += 1
                else:
                    raise Exception('Unexpected tw_type %s with %s.' % (tw_type, tw_id))
        else:
            caw_tw_id = cas_rec[0]
            cas_replies_str = cas_rec[1]
            cas_quotes_str = cas_rec[2]
            if tw_type == 'r':
                if cas_replies_str is None:
                    cas_replies_str = tw_id.strip()
                else:
                    cas_replies_str += '|'
                    cas_replies_str += tw_id.strip()
            elif tw_type == 'q':
                if cas_quotes_str is None:
                    cas_quotes_str = tw_id.strip()
                else:
                    cas_quotes_str += '|'
                    cas_quotes_str += tw_id.strip()
            else:
                raise Exception('Unexpected tw_type %s with %s.' % (tw_type, tw_id))
            cas_db_cur.execute(cas_update_sql_str, (cas_replies_str, cas_quotes_str, caw_tw_id))
            cas_tw_cnt += 1
        if cas_tw_cnt % 5000 and cas_tw_cnt >= 5000:
            cas_db_conn.commit()
            logging.debug('%s cas and %s cas_tw have been written.' % (cas_cnt, cas_tw_cnt))
    cas_db_conn.commit()
    logging.debug('%s cas and %s cas_tw have been written.' % (cas_cnt, cas_tw_cnt))
    en_db_conn.close()
    cas_db_conn.close()
    logging.debug('%s cas and %s cas_tw have been written in % secs.'
                  % (cas_cnt, cas_tw_cnt, str(time.time() - timer_start)))


def build_cas_graph():
    '''
    As we need to capture all complete cascades, we have to put all cascades stored in ven_tw_en_cas together so that
    we can extract complete cascades from the connected components.
    '''
    cas_graph = nx.DiGraph()
    en_db_conn = sqlite3.connect(g_ven_tw_db)
    en_db_cur = en_db_conn.cursor()
    en_sql_str = '''select tw_datetime from ven_tw_en where tw_id = ?'''
    cas_db_conn = sqlite3.connect(g_ven_tw_cas_db)
    cas_db_cur = cas_db_conn.cursor()
    cas_sql_str = '''select tw_id, replies, quotes from ven_tw_en_cas'''
    cas_db_cur.execute(cas_sql_str)
    l_cas_recs = cas_db_cur.fetchall()
    for cas_rec in l_cas_recs:
        cas_tw_id = cas_rec[0]
        en_db_cur.execute(en_sql_str, (cas_tw_id,))
        cas_tw_datetime = en_db_cur.fetchone()[0]
        # cas_tw_datetime = cas_rec[1]
        replies_str = cas_rec[1]
        quotes_str = cas_rec[2]
        if replies_str is None and quotes_str is None:
            continue
        cas_graph.add_node(cas_tw_id, datetime=cas_tw_datetime)
        if replies_str is not None:
            l_cas_replies = [item.strip() for item in replies_str.split('|')]
            for r_tw_id in l_cas_replies:
                en_db_cur.execute(en_sql_str, (r_tw_id,))
                r_tw = en_db_cur.fetchone()
                if r_tw is None:
                    continue
                r_tw_datetime = r_tw[0]
                cas_graph.add_node(r_tw_id, datetime=r_tw_datetime)
                cas_graph.add_edge(cas_tw_id, r_tw_id, type='r')
        if quotes_str is not None:
            l_cas_quotes = [item.strip() for item in quotes_str.split('|')]
            for q_tw_id in l_cas_quotes:
                en_db_cur.execute(en_sql_str, (q_tw_id,))
                q_tw = en_db_cur.fetchone()
                if q_tw is None:
                    continue
                q_tw_datetime = q_tw[0]
                cas_graph.add_node(q_tw_id, datetime=q_tw_datetime)
                cas_graph.add_edge(cas_tw_id, q_tw_id, type='q')
    en_db_conn.close()
    cas_db_conn.close()
    logging.debug(nx.info(cas_graph))
    nx.write_gml(cas_graph, g_ven_tw_cas_full_graph)


def output_nx_cas_graph(nx_graph, fig_path):
    plt.figure(1, figsize=(15, 15), tight_layout={'pad': 1, 'w_pad': 200, 'h_pad': 200, 'rect': None})
    pos = nx.spring_layout(nx_graph)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)
    l_en_nodes = nx_graph.nodes
    d_en_labels = {node: node for node in l_en_nodes}
    d_en_edge_labels = {(edge[0], edge[1]): edge[2]['type'] for edge in nx_graph.edges(data=True)}
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=l_en_nodes, node_color='r', node_size=30)
    nx.draw_networkx_labels(nx_graph, pos, labels=d_en_labels, font_color='r', font_size=20, font_family="sans-serif")
    nx.draw_networkx_edges(nx_graph, pos, width=5, arrowsize=60, edge_color='b')
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=d_en_edge_labels, font_color='k', font_size=20, font_family="sans-serif")
    plt.savefig(fig_path, format="PNG")
    plt.clf()


def get_root(each_cas_graph):
    for node in each_cas_graph.nodes:
        if len(list(each_cas_graph.predecessors(node))) == 0:
            return node


def extract_cas_from_graph():
    cas_graph = nx.read_gml(g_ven_tw_cas_full_graph)
    und_cas_graph = nx.to_undirected(cas_graph)
    cnt = 0
    for comp in nx.connected_components(und_cas_graph):
        each_cas_graph = nx.subgraph(cas_graph, comp)
        logging.debug(nx.info(each_cas_graph))
        root = get_root(each_cas_graph)
        if path.exists(g_ven_tw_cas_graph_format.format(root)):
            continue
        nx.write_gml(each_cas_graph, g_ven_tw_cas_graph_format.format(root))
        print(nx.info(each_cas_graph))
        # output_nx_cas_graph(each_cas_graph, g_ven_tw_cas_graph_img_format.format(root))
        cnt += 1
    logging.debug('%s cascades in total.' % cnt)


def ven_cas_graph_stats():
    l_degrees = []
    l_longest_path_lens = []
    l_nodes = []
    l_edges = []
    for (dirpath, dirname, filenames) in walk(g_ven_tw_cas_folder):
        for filename in filenames:
            if filename[-8:] != '_cas.gml':
                continue
            cas_graph = nx.read_gml(g_ven_tw_cas_folder + filename)
            l_nodes.append(cas_graph.number_of_nodes())
            l_edges.append(cas_graph.number_of_edges())
            l_degrees.append(sum(dict(cas_graph.degree()).values()) / float(cas_graph.number_of_nodes()))
            l_longest_path_lens.append(nx.algorithms.dag.dag_longest_path_length(cas_graph))
    with open(g_ven_cas_graph_stats_format.format('avgdegrees'), 'w+') as out_fd:
        out_fd.write('\n'.join([str(num) for num in l_degrees]))
        out_fd.close()
    with open(g_ven_cas_graph_stats_format.format('nodes'), 'w+') as out_fd:
        out_fd.write('\n'.join([str(num) for num in l_nodes]))
        out_fd.close()
    with open(g_ven_cas_graph_stats_format.format('edges'), 'w+') as out_fd:
        out_fd.write('\n'.join([str(num) for num in l_edges]))
        out_fd.close()
    with open(g_ven_cas_graph_stats_format.format('longestpathlens'), 'w+') as out_fd:
        out_fd.write('\n'.join([str(num) for num in l_longest_path_lens]))
        out_fd.close()


def build_resp_db_by_users():
    resp_db_conn = sqlite3.connect(g_ven_tw_resp_by_users_db)
    resp_db_cur = resp_db_conn.cursor()
    sql_str = '''create table if not exists ven_resp_by_usrs (usr_id text primary key, retweets text, replies text, quotes text, originals text)'''
    resp_db_cur.execute(sql_str)
    resp_insert_sql_str = '''insert into ven_resp_by_usrs (usr_id, retweets, replies, quotes, originals) values (?,?,?,?,?)'''

    tw_db_conn = sqlite3.connect(g_ven_tw_db)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = '''select usr_id from ven_tw_en'''
    tw_db_cur.execute(sql_str)
    l_recs = tw_db_cur.fetchall()
    s_usr_ids = set([rec[0].strip() for rec in l_recs])

    sql_str = '''select tw_id, tw_type, tw_src_id, tw_datetime from ven_tw_en where usr_id = ?'''
    cnt = 0
    timer_start = time.time()
    for usr_id in s_usr_ids:
        tw_db_cur.execute(sql_str, (usr_id,))
        l_recs = tw_db_cur.fetchall()
        l_retweets = []
        l_replies = []
        l_quotes = []
        l_originals = []
        for rec in l_recs:
            tw_id = rec[0].strip()
            tw_type = rec[1].strip()
            tw_src_id = rec[2].strip() if rec[2] is not None else None
            tw_datetime = rec[3].strip()
            if tw_type == 'n':
                resp_str = tw_id + '|' + tw_id + '|' + tw_datetime
                l_originals.append(resp_str)
            else:
                resp_str = tw_src_id + '|' + tw_id + '|' + tw_datetime
                if tw_type == 't':
                    l_retweets.append(resp_str)
                elif tw_type == 'r':
                    l_replies.append(resp_str)
                elif tw_type == 'q':
                    l_quotes.append(resp_str)
                else:
                    logging.error('Invalid tw_type @ tw_id = %s' % tw_id)
                    continue
        resp_str_retweets = '\n'.join(l_retweets) if len(l_retweets) > 0 else None
        resp_str_replies = '\n'.join(l_replies) if len(l_replies) > 0 else None
        resp_str_quotes = '\n'.join(l_quotes) if len(l_quotes) > 0 else None
        resp_str_originals = '\n'.join(l_originals) if len(l_originals) > 0 else None
        resp_db_cur.execute(resp_insert_sql_str, (usr_id, resp_str_retweets, resp_str_replies, resp_str_quotes,
                                                  resp_str_originals))
        cnt += 1
        if cnt % 50000 == 0 and cnt >= 50000:
            resp_db_conn.commit()
            logging.debug('%s usrs done in %s secs.' % (cnt, time.time() - timer_start))
    resp_db_conn.commit()
    logging.debug('%s usrs done in %s secs.' % (cnt, time.time() - timer_start))
    resp_db_conn.close()


def resp_by_usrs_stats():
    resp_db_conn = sqlite3.connect(g_ven_tw_resp_by_users_db)
    resp_db_cur = resp_db_conn.cursor()
    sql_str = '''select usr_id, retweets, replies, quotes, originals from ven_resp_by_usrs'''
    resp_db_cur.execute(sql_str)
    l_recs = resp_db_cur.fetchall()
    resp_db_conn.close()
    d_usrs = dict()
    for rec in l_recs:
        usr_id = rec[0].strip()
        d_usrs[usr_id] = {'t_cnt': 0, 'r_cnt': 0, 'q_cnt': 0, 'n_cnt':0}
        retweets = rec[1]
        if retweets is not None:
            t_cnt = len([t.strip() for t in retweets.split('\n')])
            d_usrs[usr_id]['t_cnt'] = t_cnt
        replies = rec[2]
        if replies is not None:
            r_cnt = len([r.strip() for r in replies.split('\n')])
            d_usrs[usr_id]['r_cnt'] = r_cnt
        quotes = rec[3]
        if quotes is not None:
            q_cnt = len([q.strip() for q in quotes.split('\n')])
            d_usrs[usr_id]['q_cnt'] = q_cnt
        originals = rec[4]
        if originals is not None:
            n_cnt = len([n.strip() for n in originals.split('\n')])
            d_usrs[usr_id]['n_cnt'] = n_cnt
    total_usrs = len(d_usrs)
    max_resps_per_usr = max([d_usrs[usr_resp]['t_cnt'] + d_usrs[usr_resp]['r_cnt'] + d_usrs[usr_resp]['q_cnt'] + d_usrs[usr_resp]['n_cnt'] for usr_resp in d_usrs])
    avg_resps_per_usr = sum([d_usrs[usr_resp]['t_cnt'] + d_usrs[usr_resp]['r_cnt'] + d_usrs[usr_resp]['q_cnt'] + d_usrs[usr_resp]['n_cnt'] for usr_resp in d_usrs]) / total_usrs
    max_ts_per_usr = max([d_usrs[usr_resp]['t_cnt'] for usr_resp in d_usrs])
    avg_ts_per_usr = sum([d_usrs[usr_resp]['t_cnt'] for usr_resp in d_usrs]) / total_usrs
    max_rs_per_usr = max([d_usrs[usr_resp]['r_cnt'] for usr_resp in d_usrs])
    avg_rs_per_usr = sum([d_usrs[usr_resp]['r_cnt'] for usr_resp in d_usrs]) / total_usrs
    max_qs_per_usr = max([d_usrs[usr_resp]['q_cnt'] for usr_resp in d_usrs])
    avg_qs_per_usr = sum([d_usrs[usr_resp]['q_cnt'] for usr_resp in d_usrs]) / total_usrs
    max_ns_per_usr = max([d_usrs[usr_resp]['n_cnt'] for usr_resp in d_usrs])
    avg_ns_per_usr = sum([d_usrs[usr_resp]['n_cnt'] for usr_resp in d_usrs]) / total_usrs
    with open(g_ven_tw_folder + 'ven_tw_resp_by_usrs_stats_v2-1.json', 'w+') as out_fd:
        json.dump(d_usrs, out_fd)
        out_fd.close()
    with open(g_ven_tw_folder + 'ven_tw_resp_by_usrs_stats_v2-1.txt', 'w+') as out_fd:
        out_fd.write('total_usrs: %s' % total_usrs)
        out_fd.write('\n')
        out_fd.write('max_resps_per_usr: %s' % max_resps_per_usr)
        out_fd.write('\n')
        out_fd.write('avg_resps_per_usr: %s' % avg_resps_per_usr)
        out_fd.write('\n')
        out_fd.write('max_ts_per_usr: %s' % max_ts_per_usr)
        out_fd.write('\n')
        out_fd.write('avg_ts_per_usr: %s' % avg_ts_per_usr)
        out_fd.write('\n')
        out_fd.write('max_rs_per_usr: %s' % max_rs_per_usr)
        out_fd.write('\n')
        out_fd.write('avg_rs_per_usr: %s' % avg_rs_per_usr)
        out_fd.write('\n')
        out_fd.write('max_qs_per_usr: %s' % max_qs_per_usr)
        out_fd.write('\n')
        out_fd.write('avg_qs_per_usr: %s' % avg_qs_per_usr)
        out_fd.write('\n')
        out_fd.write('max_ns_per_usr: %s' % max_ns_per_usr)
        out_fd.write('\n')
        out_fd.write('avg_ns_per_usr: %s' % avg_ns_per_usr)
        out_fd.write('\n')
        out_fd.close()


def select_usrs_by_resps():
    l_sel_usrs = []
    with open(g_ven_tw_folder + 'ven_tw_resp_by_usrs_stats_v2-1.json', 'r') as in_fd:
        d_usrs = json.load(in_fd)
        in_fd.close()
    for usr_id in d_usrs:
        total_resps = d_usrs[usr_id]['t_cnt'] + d_usrs[usr_id]['r_cnt'] + d_usrs[usr_id]['q_cnt'] + d_usrs[usr_id]['n_cnt']
        if 50 <= total_resps <= 100:
            l_sel_usrs.append(usr_id)
    with open(g_ven_tw_folder + 'ven_tw_resp_sel_usrs_50_100.txt', 'w+') as out_fd:
        out_str = '\n'.join(l_sel_usrs)
        out_fd.write(out_str)
        out_fd.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Extract all English and translated tweets from the raw data, and store them into a database.
    # find_all_en_tw()
    # Extract all cascades from the above tweets, and store them into another database.
    # find_all_en_cas()
    # Build the whole graph for all cascades
    # build_cas_graph()
    # extract_cas_from_graph()
    # ven_cas_graph_stats()

    # add_retweet_texts()

    # build_resp_db_by_users()
    # resp_by_usrs_stats()
    select_usrs_by_resps()

    # with open(g_ven_tw_folder + 'cp4.venezuela.twitter.training.anon.v2-1.2018-dec.json', 'r') as in_fd:
    #     for ln in in_fd:
    #         tw_json = json.loads(ln)
    #         tw_type = get_tw_type(tw_json)
    #         tw_lang = get_tw_lang(tw_json)
    #         if tw_lang != 'en':
    #             print('-----start-----')
    #             print(get_tw_raw_txt(tw_json, tw_type, tw_lang))
    #             print('----------')
    #             print(get_tw_orig_raw_txt(tw_json, tw_type))
    #             print('-----end-----')

    # For Semantic Unit Verification
    # find_all_orig_en_tw_ids()