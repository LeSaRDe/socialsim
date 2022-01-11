import json
import logging
from datetime import datetime, timedelta
import os
import time
import sqlite3
import multiprocessing
import threading
import math
from gensim.models import KeyedVectors
import numpy as np
import data_preprocessing_utils

g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_raw_data_path = g_path_prefix + 'Tng_an_WH_Twitter_v2.json'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_time_int_path = g_time_series_data_path_prefix + 'time_intervals.txt'
g_time_series_data_format = g_time_series_data_path_prefix + '{0}/{1}_nec_data'
g_time_series_data_db_format = g_time_series_data_path_prefix + '{0}/{1}_nec_data.db'
g_raw_data_db_path = g_time_series_data_path_prefix + 'wh_tw_raw.db'
g_time_series_data_txt_by_uid_db_format = g_time_series_data_path_prefix + '{0}/{1}_txt_by_uid.db'
g_retweet_file_path = g_path_prefix + 'Tng_an_Retweet_Chain_WH.json'
g_retweet_db_path = g_time_series_data_path_prefix + 'wh_tw_ret.db'
g_time_format = '%Y%m%d'
g_lexvec_model_path = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.300d.W+C.pos.vectors'
g_unique_raw_cln_by_uid_json_path = g_time_series_data_path_prefix + 'unique_raw_cln_by_uid.json'


def read_n_write_nec_data(l_time_ints):
    # {20180401_20180405 : [{necessary tweet data dict}, ...], ...}
    d_nec_data = {time_int[0].strftime(g_time_format) + '_' + time_int[1].strftime(g_time_format): []
                  for time_int in l_time_ints}
    data_preprocessing_utils.text_clean_init()
    count = 0
    timer_start = time.time()
    with open(g_raw_data_path, 'r') as in_fd:
        t_json_str = in_fd.readline()
        while t_json_str:
            time_int_str, t_nec_data = read_nec_data_for_each_tweet(t_json_str, l_time_ints)
            if time_int_str is None or time_int_str not in d_nec_data:
                # logging.debug('%s is not in any time interval.' % time_int_str)
                t_json_str = in_fd.readline()
                count += 1
                if count % 50000 == 0 and count >= 50000:
                    logging.debug('%s tweets have been scanned in %s.' % (count, str(time.time() - timer_start)))
                continue
                # raise Exception('%s is not in any time interval.' % time_int_str)
            d_nec_data[time_int_str].append(t_nec_data)
            t_json_str = in_fd.readline()
            count += 1
            if count % 50000 == 0 and count >= 50000:
                logging.debug('%s tweets have been scanned in %s.' % (count, str(time.time() - timer_start)))
        in_fd.close()
    logging.debug('%s tweets have been scanned in %s.' % (count, str(time.time() - timer_start)))

    for time_int in d_nec_data:
        l_nec_data_sorted_by_time = sorted(d_nec_data[time_int], key=lambda t: t['time'])
        with open(g_time_series_data_format.format(time_int, time_int), 'w+') as out_fd:
            for nec_data in l_nec_data_sorted_by_time:
                json.dump(nec_data, out_fd)
                out_fd.write('\n')
            out_fd.close()
        logging.debug('%s necessary data has been written.' % time_int)


# {'uid' : str, 'tid' : str, 'time' : 'YYYYMMDDHHMMSS', 'type' : 'n'|'r'|'t'|'q', 'src' : uid|'', 'raw' : [sent_strs],
# 'raw_cln' : [sent_strs]|[], 'org' : [sent_strs]|[], 'org_cln' : [sent_strs]|[]}
# 'n'|'r'|'t'|'q' -- new post|reply|retweet|quote
# src -- only available for reply, retweet and quote. means 'from'.
# raw -- new post: text_m, reply: text_m, retweet: retweet.text_m + quote.text_m (if any), quote: text_m + quote.text_m
# org -- new post: text_m, reply: text_m, retweet: '', quote: text_m
def read_nec_data_for_each_tweet(t_json_str, l_time_ints):
    t_json = json.loads(t_json_str)
    d_nec_data = dict()
    d_nec_data['uid'] = t_json['user']['id_str_h']
    d_nec_data['tid'] = t_json['id_str_h']
    d_nec_data['time'] = data_preprocessing_utils.get_user_time(t_json['created_at'])
    d_nec_data['type'] = data_preprocessing_utils.get_tweet_type(t_json)
    d_nec_data['src'] = get_src_uid(t_json, d_nec_data['type'])
    d_nec_data['raw'] = data_preprocessing_utils.get_raw_text(t_json, d_nec_data['type'])
    l_raw_cln = []
    for dirty_sent in d_nec_data['raw']:
        l_cln_sents = data_preprocessing_utils.text_clean(dirty_sent)
        l_raw_cln += l_cln_sents
    d_nec_data['raw_cln'] = l_raw_cln
    d_nec_data['org'] = data_preprocessing_utils.get_org_text(t_json, d_nec_data['type'])
    l_org_cln = []
    for dirty_sent in d_nec_data['org']:
        l_cln_sents = data_preprocessing_utils.text_clean(dirty_sent)
        l_org_cln += l_cln_sents
    d_nec_data['org_cln'] = l_org_cln
    time_int_str = get_time_int_str(d_nec_data['time'], l_time_ints)
    return time_int_str, d_nec_data


def get_time_int_str(time_str, l_time_ints):
    moment = datetime.strptime(time_str, '%Y%m%d%H%M%S')
    time_int_str = None
    for time_int in l_time_ints:
        start_time = time_int[0]
        end_time = time_int[1]
        if start_time <= moment <= end_time:
            time_int_str = start_time.strftime(g_time_format) + '_' + end_time.strftime(g_time_format)
            break
    return time_int_str


# we take the src anyway regardless of the presence of the src user in the considered time interval
def get_src_uid(t_json, t_type):
    src_uid = ''
    if t_type == 'r':
        src_uid = t_json['in_reply_to_user_id_str_h']

        # CAUTION:
        # it happens that the tweet specified by 'in_reply_to_status_id_h' doesn't exist.
        # however, 'in_reply_to_user_id_str_h' usually exists regardless of the existence of the replied tweet.
        # we use 'in_reply_to_user_id_str_h' to fetch our 'src_uid', though doing this may potentially lead to more
        # low text similarities between users as the user specified by 'src_uid' may not have any tweet at all.

        # src_tid = t_json['in_reply_to_status_id_h']
        # db_conn = sqlite3.connect(g_raw_data_db_path)
        # db_cur = db_conn.cursor()
        # sql_str = '''SELECT t_json from wh_twitter WHERE tid = ?'''
        # db_cur.execute(sql_str, (src_tid, ))
        # src_rec = db_cur.fetchone()
        # if src_rec is not None:
        #     src_t_json = json.loads(src_rec[0])
        #     src_uid = src_t_json['user']['id_str_h']
        # db_conn.close()
    elif t_type == 'q':
        src_uid = t_json['quoted_status']['user']['id_str_h']
    elif t_type == 't':
        ret_tid = t_json['id_str_h']
        ret_uid = t_json['user']['id_str_h']
        db_conn = sqlite3.connect(g_retweet_db_path)
        db_cur = db_conn.cursor()
        sql_str = '''SELECT ret_uid, reted_uid FROM wh_retweet WHERE ret_tid = ?'''
        db_cur.execute(sql_str, (ret_tid,))
        ret_rec = db_cur.fetchone()
        if ret_rec is not None:
            if ret_uid != ret_rec[0]:
                raise Exception('%s has a conflict retweet record.' % ret_tid)
            else:
                src_uid = ret_rec[1]
                if src_uid is None:
                    src_uid = ''
        else:
            logging.debug('%s is not contained in the retweet records.' % ret_tid)
        db_conn.close()

    return src_uid


# not_in_data: 'ret'|'reted'|'both'|'none'
def load_retweet_data():
    ret_db_conn = sqlite3.connect(g_retweet_db_path)
    ret_db_cur = ret_db_conn.cursor()
    sql_str = '''CREATE TABLE IF NOT EXISTS wh_retweet (ret_tid TEXT PRIMARY KEY NOT NULL, reted_tid TEXT NOT NULL, \
     src_tid TEXT NOT NULL, not_in_data TEXT NOT NULL, ret_uid TEXT, reted_uid TEXT, src_uid TEXT)'''
    ret_db_cur.execute(sql_str)
    ret_db_conn.commit()
    raw_db_conn = sqlite3.connect(g_raw_data_db_path)
    raw_db_cur = raw_db_conn.cursor()
    with open(g_retweet_file_path, 'r') as in_fd:
        count = 0
        ret_json_str = in_fd.readline()
        while ret_json_str:
            sql_str = '''SELECT t_json FROM wh_twitter WHERE tid = ?'''
            ret_json = json.loads(ret_json_str)
            not_in_data = 'none'
            ret_uid = None
            reted_uid = None
            src_uid = None
            ret_tid = ret_json['tweet_id_h']
            reted_tid = ret_json['retweeted_from_tweet_id_h']
            src_tid = ret_json['source_tweet_id_h']
            raw_db_cur.execute(sql_str, (ret_tid,))
            ret_t_rec = raw_db_cur.fetchone()
            if ret_t_rec is None:
                not_in_data = 'ret'
            else:
                ret_t_json = json.loads(ret_t_rec[0])
                ret_uid = ret_t_json['user']['id_str_h']
            raw_db_cur.execute(sql_str, (reted_tid,))
            reted_t_rec = raw_db_cur.fetchone()
            if reted_t_rec is None:
                if not_in_data == 'ret':
                    not_in_data = 'both'
                else:
                    not_in_data = 'reted'
            else:
                reted_t_json = json.loads(reted_t_rec[0])
                reted_uid = reted_t_json['user']['id_str_h']
            raw_db_cur.execute(sql_str, (src_tid,))
            src_t_rec = raw_db_cur.fetchone()
            if src_t_rec is not None:
                src_t_json = json.loads(src_t_rec[0])
                src_uid = src_t_json['user']['id_str_h']
            sql_str = '''INSERT INTO wh_retweet VALUES (?, ?, ?, ?, ?, ?, ?)'''
            ret_db_cur.execute(sql_str, (ret_tid, reted_tid, src_tid, not_in_data, ret_uid, reted_uid, src_uid))
            # ret_db_conn.commit()
            ret_json_str = in_fd.readline()
            count += 1
            if count % 10000 == 0 and count >= 10000:
                ret_db_conn.commit()
                logging.debug('%s retweet records have been handled.' % count)
        ret_db_conn.commit()
        logging.debug('%s retweet records have been handled.' % count)
        in_fd.close()
    ret_db_conn.close()
    raw_db_conn.close()


def create_time_int_folders(l_time_ints):
    for time_int in l_time_ints:
        time_int_str = time_int[0].strftime(g_time_format) + '_' + time_int[1].strftime(g_time_format)
        if not os.path.exists(g_time_series_data_path_prefix + time_int_str):
            os.mkdir(g_time_series_data_path_prefix + time_int_str)
    logging.debug('Time interval folders have been created.')


# l_time_ints: [[datetime, datetime], ...]
def read_time_ints():
    l_time_ints = []
    with open(g_time_int_path, 'r') as in_fd:
        l_lines = in_fd.readlines()
        in_fd.close()
        for line in l_lines:
            start_day_str, end_day_str = line.split(':')
            l_time_ints.append([datetime.strptime(start_day_str.strip(), g_time_format),
                                datetime.strptime(end_day_str.strip(), g_time_format)])
    logging.debug('%s time intervals are read.' % len(l_time_ints))
    return l_time_ints


def write_raw_data_to_db():
    with open(g_raw_data_path, 'r') as in_fd:
        db_conn = sqlite3.connect(g_raw_data_db_path)
        db_cur = db_conn.cursor()
        sql_str = '''CREATE TABLE IF NOT EXISTS wh_twitter (tid TEXT PRIMARY KEY NOT NULL, t_json JSON NOT NULL)'''
        db_cur.execute(sql_str)
        db_conn.commit()
        sql_str = '''INSERT INTO wh_twitter VALUES (?, ?)'''
        count = 0
        t_json_str = in_fd.readline()
        while t_json_str:
            t_json = json.loads(t_json_str)
            tid = t_json['id_str_h']
            db_cur.execute(sql_str, (tid, t_json_str))
            count += 1
            if count % 5000 == 0 and count >= 5000:
                db_conn.commit()
                logging.debug('%s tweets have been written to DB.' % count)
            t_json_str = in_fd.readline()
        db_conn.commit()
        logging.debug('%s tweets have been written to DB.' % count)
        db_conn.close()
        in_fd.close()


def time_int_to_time_int_str(time_int):
    return time_int[0].strftime(g_time_format) + '_' + time_int[1].strftime(g_time_format)


def read_nec_data_to_db(l_time_ints):
    # l_time_ints = read_time_ints()
    for time_int in l_time_ints:
        time_int_str = time_int_to_time_int_str(time_int)
        with open(g_time_series_data_format.format(time_int_str, time_int_str), 'r') as in_fd:
            db_conn = sqlite3.connect(g_time_series_data_db_format.format(time_int_str, time_int_str))
            db_cur = db_conn.cursor()
            # raw, raw_cln, org, and org_cln are sentences joined by '\n' or a mere ''.
            sql_str = '''CREATE TABLE IF NOT EXISTS wh_nec_data (tid TEXT PRIMARY KEY, uid TEXT NOT NULL, ''' \
                      '''time TEXT NOT NULL, type TEXT NOT NULL, src TEXT, raw TEXT, raw_cln TEXT, org TEXT, org_cln TEXT)'''
            db_cur.execute(sql_str)
            db_conn.commit()
            nec_data_str = in_fd.readline()
            count = 0
            while nec_data_str:
                nec_data_json = json.loads(nec_data_str)
                sql_str = '''INSERT INTO wh_nec_data(tid, uid, time, type, src, raw, raw_cln, org, org_cln) ''' \
                          '''VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
                tid = nec_data_json['tid']
                uid = nec_data_json['uid']
                time = nec_data_json['time']
                type = nec_data_json['type']
                src = nec_data_json['src']
                raw = '\n'.join(nec_data_json['raw'])
                raw_cln = '\n'.join(nec_data_json['raw_cln'])
                org = '\n'.join(nec_data_json['org'])
                org_cln = '\n'.join(nec_data_json['org_cln'])
                db_cur.execute(sql_str, (tid, uid, time, type, src, raw, raw_cln, org, org_cln))
                count += 1
                if count % 1000 == 0 and count >= 1000:
                    db_conn.commit()
                    logging.debug('%s nec data records have been written into %s DB.' % (count, time_int_str))
                nec_data_str = in_fd.readline()
            db_conn.commit()
            logging.debug('%s nec data records have been written into %s DB.' % (count, time_int_str))
            logging.debug('%s DB is done.' % time_int_str)
            db_conn.close()
            in_fd.close()


def load_lexvec_model():
    lexvec_model = KeyedVectors.load_word2vec_format(g_lexvec_model_path, binary=False)
    return lexvec_model


def add_text_vect_to_db(l_time_ints, txt2vec_model):
    for time_int in l_time_ints:
        time_int_str = time_int_to_time_int_str(time_int)
        db_conn = sqlite3.connect(g_time_series_data_db_format.format(time_int_str, time_int_str))
        db_cur = db_conn.cursor()
        sql_str = '''SELECT COUNT(*) FROM pragma_table_info("wh_nec_data") WHERE name="raw_cln_vec"'''
        db_cur.execute(sql_str)
        exist_raw_cln_vec = db_cur.fetchone()[0]
        if exist_raw_cln_vec == 0:
            sql_str = '''ALTER TABLE wh_nec_data ADD COLUMN raw_cln_vec TEXT'''
            db_cur.execute(sql_str)
            db_conn.commit()
        sql_str = '''SELECT tid, raw_cln FROM wh_nec_data'''
        db_cur.execute(sql_str)
        l_nec_data = db_cur.fetchall()
        count = 0
        for nec_data in l_nec_data:
            tid = nec_data[0]
            l_raw_cln_sents = nec_data[1].split('\n')
            l_raw_cln_sent_vecs = []
            for raw_cln_sent in l_raw_cln_sents:
                raw_cln_sent_vec = np.zeros((300,))
                l_words = [word.strip() for word in raw_cln_sent.split(' ')]
                for word in l_words:
                    try:
                        word_vec = txt2vec_model.wv[word.lower()]
                    except:
                        word_vec = None
                    if word_vec is not None:
                        raw_cln_sent_vec += word_vec
                l_raw_cln_sent_vecs.append(raw_cln_sent_vec)
            raw_cln_vec = sum(l_raw_cln_sent_vecs)
            raw_cln_vec_str = ','.join([str(ele) for ele in raw_cln_vec])
            sql_str = '''UPDATE wh_nec_data SET raw_cln_vec = ? WHERE tid = ?'''
            db_cur.execute(sql_str, (raw_cln_vec_str, tid))
            count += 1
            if count % 5000 == 0 and count >= 5000:
                db_conn.commit()
                logging.debug('%s text vectors have been written to %s DB.' % (count, time_int_str))
        db_conn.commit()
        logging.debug('%s text vectors have been written to %s DB.' % (count, time_int_str))
        logging.debug('%s DB text vectors are done.' % time_int_str)
        db_conn.close()


def build_txt_by_uid_db(l_time_ints):
    for time_int in l_time_ints:
        time_int_str = time_int_to_time_int_str(time_int)
        txt_db_conn = sqlite3.connect(g_time_series_data_txt_by_uid_db_format.format(time_int_str, time_int_str))
        txt_db_cur = txt_db_conn.cursor()
        sql_str = '''CREATE TABLE IF NOT EXISTS wh_txt_by_uid (uid TEXT PRIMARY KEY, raw TEXT, raw_cln TEXT,''' \
                  '''raw_cln_lexvec TEXT)'''
        txt_db_cur.execute(sql_str)
        txt_db_conn.commit()
        nec_db_conn = sqlite3.connect(g_time_series_data_db_format.format(time_int_str, time_int_str))
        nec_db_cur = nec_db_conn.cursor()
        sql_str = '''SELECT uid FROM wh_nec_data'''
        nec_db_cur.execute(sql_str)
        l_uids = [rec[0] for rec in nec_db_cur.fetchall()]
        count = 0
        for uid in set(l_uids):
            sql_str = '''SELECT raw, raw_cln, raw_cln_vec FROM wh_nec_data WHERE uid=?'''
            nec_db_cur.execute(sql_str, (uid,))
            l_recs = nec_db_cur.fetchall()
            l_raw_sents = []
            l_raw_cln_sents = []
            l_raw_cln_vecs = []
            for rec in l_recs:
                l_raw_sents.append(rec[0])
                l_raw_cln_sents.append(rec[1])
                l_raw_cln_vecs.append(rec[2])
            txt_raw = '\n'.join(l_raw_sents)
            txt_raw_cln = '\n'.join(l_raw_cln_sents)
            raw_cln_vec = []
            for raw_cln_vec_str in l_raw_cln_vecs:
                raw_cln_vec += [float(ele.strip()) for ele in raw_cln_vec_str.split(',')]
            txt_raw_cln_vec_str = ','.join([str(ele) for ele in raw_cln_vec])

            sql_str = '''INSERT INTO wh_txt_by_uid VALUES (?, ?, ?, ?)'''
            txt_db_cur.execute(sql_str, (uid, txt_raw, txt_raw_cln, txt_raw_cln_vec_str))
            count += 1
            if count % 1000 and count >= 1000:
                txt_db_conn.commit()
                logging.debug('%s users have been done for %s DB.' % (str(count), time_int_str))
        txt_db_conn.commit()
        txt_db_conn.close()
        logging.debug('%s users have been done for %s DB.' % (str(count), time_int_str))
        logging.debug('%s DB is done.' % time_int_str)


def remove_add_stopwords(l_time_ints):
    for time_int in l_time_ints:
        time_int_str = time_int_to_time_int_str(time_int)
        txt_db_conn = sqlite3.connect(g_time_series_data_txt_by_uid_db_format.format(time_int_str, time_int_str))
        txt_db_cur = txt_db_conn.cursor()
        sql_str = '''SELECT uid, raw_cln FROM wh_txt_by_uid'''
        txt_db_cur.execute(sql_str)
        l_recs = txt_db_cur.fetchall()
        count = 0
        for rec in l_recs:
            uid = rec[0]
            dirty_raw_cln = rec[1]
            l_dirty_sents = dirty_raw_cln.split('\n')
            l_clean_sents = []
            for dirty_sent in l_dirty_sents:
                clean_sent = ' '.join([word for word in dirty_sent.split(' ')
                                       if not data_preprocessing_utils.is_stop_word(word)])
                l_clean_sents.append(clean_sent)
            clean_raw_cln = '\n'.join(l_clean_sents)
            sql_str = '''UPDATE wh_txt_by_uid SET raw_cln = ? WHERE uid = ?'''
            txt_db_cur.execute(sql_str, (clean_raw_cln, uid))
            count += 1
            if count % 1000 == 0 and count >= 1000:
                txt_db_conn.commit()
                logging.debug('%s user texts have been cleaned for %s.' % (str(count), time_int_str))
        txt_db_conn.commit()
        logging.debug('%s user texts have been cleaned for %s.' % (str(count), time_int_str))
        logging.debug('%s text clean is done.' % time_int_str)
        txt_db_conn.close()


def compute_txt_vecs(l_time_ints, txt2vec_model):
    for time_int in l_time_ints:
        time_int_str = time_int_to_time_int_str(time_int)
        txt_db_conn = sqlite3.connect(g_time_series_data_txt_by_uid_db_format.format(time_int_str, time_int_str))
        txt_db_cur = txt_db_conn.cursor()
        sql_str = '''SELECT uid, raw_cln FROM wh_txt_by_uid'''
        txt_db_cur.execute(sql_str)
        l_recs = txt_db_cur.fetchall()
        for rec in l_recs:
            uid = rec[0]
            raw_cln = rec[1]
            l_raw_cln_sents = raw_cln.split('\n')
            l_raw_cln_sent_vecs = []
            count = 0
            for raw_cln_sent in l_raw_cln_sents:
                raw_cln_sent_vec = np.zeros((300,))
                l_words = [word.strip() for word in raw_cln_sent.split(' ')]
                for word in l_words:
                    try:
                        word_vec = txt2vec_model.wv[word.lower()]
                    except:
                        word_vec = None
                    if word_vec is not None:
                        raw_cln_sent_vec += word_vec
                l_raw_cln_sent_vecs.append(raw_cln_sent_vec)
            raw_cln_vec = sum(l_raw_cln_sent_vecs)
            raw_cln_vec_str = ','.join([str(ele) for ele in raw_cln_vec])
            sql_str = '''UPDATE wh_txt_by_uid SET raw_cln_lexvec = ? WHERE uid = ?'''
            txt_db_cur.execute(sql_str, (raw_cln_vec_str, uid))
            count += 1
            if count % 5000 == 0 and count >= 5000:
                txt_db_conn.commit()
                logging.debug('%s text vectors have been written to %s DB.' % (count, time_int_str))
        txt_db_conn.commit()
        logging.debug('%s text vectors have been written to %s DB.' % (count, time_int_str))
        logging.debug('%s DB text vectors are done.' % time_int_str)
        txt_db_conn.close()


def add_clean_txt_by_uid_db(l_time_ints):
    for time_int in l_time_ints:
        time_int_str = time_int_to_time_int_str(time_int)


def unique_txt_by_uid():
    l_time_ints = read_time_ints()
    d_txt_by_tid = dict()
    for time_int in l_time_ints:
        time_int_str = time_int_to_time_int_str(time_int)
        nec_db_conn = sqlite3.connect(g_time_series_data_db_format.format(time_int_str, time_int_str))
        nec_db_cur = nec_db_conn.cursor()
        sql_str = '''SELECT tid, uid, raw_cln FROM wh_nec_data'''
        nec_db_cur.execute(sql_str)
        l_nec_recs = nec_db_cur.fetchall()
        for nec_rec in l_nec_recs:
            tid = nec_rec[0]
            uid = nec_rec[1]
            raw_cln = nec_rec[2]
            if tid not in d_txt_by_tid:
                d_txt_by_tid[tid] = {'uid': uid, 'raw_cln': raw_cln}
        nec_db_conn.close()
        logging.debug('%s is done with reading texts.' % time_int_str)
    logging.debug('All time intervals are done with reading texts.')

    d_txt_by_uid = dict()
    for tid in d_txt_by_tid:
        uid = d_txt_by_tid[tid]['uid']
        if uid not in d_txt_by_uid:
            d_txt_by_uid[uid] = d_txt_by_tid[tid]['raw_cln']
        else:
            d_txt_by_uid[uid] += '\n'
            d_txt_by_uid[uid] += d_txt_by_tid[tid]['raw_cln']

    with open(g_unique_raw_cln_by_uid_json_path, 'w+') as out_fd:
        json.dump(d_txt_by_uid, out_fd)
        out_fd.close()


def operate_db_multithreads(op_func, add_args=None):
    # def read_nec_data_to_db_multithreads():
    l_time_ints = read_time_ints()
    batch_size = math.ceil(len(l_time_ints) / multiprocessing.cpu_count())
    l_l_time_ints = []
    for i in range(0, len(l_time_ints), batch_size):
        if i + batch_size < len(l_time_ints):
            l_l_time_ints.append(l_time_ints[i:i + batch_size])
        else:
            l_l_time_ints.append(l_time_ints[i:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_time_ints:
        if add_args is not None:
            t = threading.Thread(target=op_func, args=(l_each_batch,) + add_args)
        else:
            t = threading.Thread(target=op_func, args=(l_each_batch,))
        t.setName('nec_to_db_t' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('Thread %s is finished.' % t.getName())

    logging.debug('All nec data has been written to DB.')


def main():
    # write_raw_data_to_db()
    # load_retweet_data()
    # l_time_ints = read_time_ints()
    # create_time_int_folders(l_time_ints)
    # read_n_write_nec_data(l_time_ints)
    # operate_db_multithreads(read_nec_data_to_db)
    # lexvec_model = load_lexvec_model()
    # operate_db_multithreads(add_text_vect_to_db, (lexvec_model,))
    # operate_db_multithreads(build_txt_by_uid_db)
    # unique_txt_by_uid()
    # data_preprocessing_utils.text_clean_init()
    # operate_db_multithreads(remove_add_stopwords)
    lexvec_model = load_lexvec_model()
    operate_db_multithreads(compute_txt_vecs, (lexvec_model,))
    

def test_main():
    data_preprocessing_utils.text_clean_init()
    db_conn = sqlite3.connect(g_time_series_data_path_prefix + 'wh_tw_nec_data.db')
    db_cur = db_conn.cursor()
    sql_str = '''SELECT raw from wh_nec_data WHERE tid=?'''
    db_cur.execute(sql_str, ('N3xUDMpz1rOg3d73lRWDxg',))
    raw_txt = db_cur.fetchone()[0]
    clean_txt = data_preprocessing_utils.text_clean(raw_txt)
    print(clean_txt)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # main()
    test_main()