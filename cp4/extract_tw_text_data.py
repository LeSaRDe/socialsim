'''
OBJECTIVES:
    Extract text and related data from the raw input Twitter data, and store the extracted data into a sqlite database.
NOTE:
    As in the challenge the Twitter data may have been stored in another database, and we may have to drop using the
    database built in this file, we wrap a set of APIs for other modules for retrieving Twitter data such as raw texts.
    Those APIs need to be wrapped for both our own database and the external database.
'''

import json
import logging
from os import walk, path
import time
import threading
import multiprocessing
import math
import sys
import traceback
from datetime import datetime
import csv

import sqlite3
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import global_settings


############################################################
#   Tweet Object Parsing
############################################################
# def get_tw_src_info():
#     logging.debug('[get_tw_src_info] Starts...')
#     for filename in global_settings.g_tw_raw_data_file_list:
#         if l_sel_ds is not None:
#             if filename not in l_sel_ds:
#                 continue



def get_tw_type(tw_json):
    '''
    'r' for reply, 'q' for quote, 't' for retweet, and 'n' for original
    '''
    if 'in_reply_to_status_id_str_h' in tw_json \
            and tw_json['in_reply_to_status_id_str_h'] != '' \
            and tw_json['in_reply_to_status_id_str_h'] is not None:
            # and str(tw_json['in_reply_to_status_id_str_h']) != 'nan' \
        t_type = 'r'
    elif 'retweeted_status' in tw_json \
            and tw_json['retweeted_status'] is not None:
            # and str(tw_json['retweeted_status']) != 'nan':
        t_type = 't'
    elif 'quoted_status' in tw_json \
            and tw_json['quoted_status'] is not None:
            # and str(tw_json['quoted_status']) != 'nan':
        t_type = 'q'
    else:
        t_type = 'n'
    return t_type


def get_tw_lang(tw_json):
    # if str(tw_json['lang']) == 'nan':
    #     return None
    return tw_json['lang']


def get_tw_usr_id(tw_json):
    # if str(tw_json['user']) == 'nan' or str(tw_json['user']['id_str_h']) == 'nan':
    #     return None
    if 'user' not in tw_json:
        return None
    return tw_json['user']['id_str_h']


# def get_src_usr_id(tw_json):
#     tw_type = get_tw_type(tw_json)
#     src_uid = ''
#     if tw_type == 'r':
#         src_uid = tw_json['in_reply_to_user_id_str_h']
#     elif tw_type == 'q':
#         src_uid = tw_json['quoted_status']['user']['id_str_h']
#     elif tw_type == 't':
#         ret_tid = tw_json['id_str_h']
#         ret_uid = tw_json['user']['id_str_h']
#         db_conn = sqlite3.connect(g_retweet_db_path)
#         db_cur = db_conn.cursor()
#         sql_str = '''SELECT ret_uid, reted_uid FROM wh_retweet WHERE ret_tid = ?'''
#         db_cur.execute(sql_str, (ret_tid,))
#         ret_rec = db_cur.fetchone()
#         if ret_rec is not None:
#             if ret_uid != ret_rec[0]:
#                 raise Exception('%s has a conflict retweet record.' % ret_tid)
#             else:
#                 src_uid = ret_rec[1]
#                 if src_uid is None:
#                     src_uid = ''
#         else:
#             logging.debug('%s is not contained in the retweet records.' % ret_tid)
#         db_conn.close()
#
#     return src_uid


def get_tw_id(tw_json):
    # if str(tw_json['id_str_h']) == 'nan':
    #     return None
    return tw_json['id_str_h']


def get_tw_raw_txt(tw_json, tw_type, tw_lang):
    '''
    For replies, quotes and originals, the raw texts to be returns are the originals without the texts that replied or
    quoted. For retweets, the raw texts are exactly what are retweeted.
    Only English texts and translated texts are considered.
    The returned value can be None.
    '''
    if tw_type == 'n' or tw_type == 'r' or tw_type == 'q':
        if tw_lang == 'en':
            if 'full_text_m' in tw_json:
                    # and str(tw_json['full_text_m']) != 'nan':
                return tw_json['full_text_m']
            # elif str(tw_json['text_m']) != 'nan':
            #     return tw_json['text_m']
            elif 'text_m' in tw_json:
                return tw_json['text_m']
            else:
                return None
        else:
            if 'extension' in tw_json and 'google_translation_m' in tw_json['extension']:
                    # and str(tw_json['extension']['google_translation_m']) != 'nan':
                    # str(tw_json['extension']) != 'nan' \
                return tw_json['extension']['google_translation_m']
            else:
                return None
    elif tw_type == 't':
        if tw_lang == 'en':
            if 'retweeted_status' in tw_json and 'full_text_m' in tw_json['retweeted_status']:
                    # str(tw_json['retweeted_status']) != 'nan' \
                    # and 'full_text_m' in tw_json['retweeted_status'] \
                    # and str(tw_json['retweeted_status']['full_text_m']) != 'nan':
                return tw_json['retweeted_status']['full_text_m']
            elif 'retweeted_status' in tw_json and 'text_m' in tw_json['retweeted_status']:
                    # str(tw_json['retweeted_status']) != 'nan' \
                    # and str(tw_json['retweeted_status']['text_m']) != 'nan':
                return tw_json['retweeted_status']['text_m']
            else:
                return None
        else:
            if 'extension' in tw_json and 'google_translation_m' in tw_json['extension']:
                    # str(tw_json['extension']) != 'nan' \
                    # and 'google_translation_m' in tw_json['extension'] \
                    # and str(tw_json['extension']['google_translation_m']) != 'nan':
                return tw_json['extension']['google_translation_m']
            else:
                return None
    else:
        return None


def get_tw_src_id(tw_json, tw_type):
    src_id = None
    if tw_type == 'n':
        return None
    elif tw_type == 'r' and 'in_reply_to_status_id_str_h' in tw_json:
            # and str(tw_json['in_reply_to_status_id_str_h']) != 'nan':
        src_id = tw_json['in_reply_to_status_id_str_h']
    elif tw_type == 'q' and 'quoted_status_id_str_h' in tw_json:
        # and str(tw_json['quoted_status_id_str_h']) != 'nan':
        src_id = tw_json['quoted_status_id_str_h']
    elif tw_type == 't' and 'retweeted_status' in tw_json and 'id_str_h' in tw_json['retweeted_status']:
        # and str(tw_json['retweeted_status']) != 'nan':
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
        raise Exception('Wrong month exists! user_time = %s' % month_str)
    return month


def get_tw_datetime(tw_json):
    '''
    Converte the datetime in the raw tweet object to the formart: YYYYMMDDHHMMSS
    '''
    if 'created_at' not in tw_json:
            # str(tw_json['created_at']) == 'nan':
        return None
    date_fields = [item.strip() for item in tw_json['created_at'].split(' ')]
    mon_str = translate_month(date_fields[1])
    day_str = date_fields[2]
    year_str = date_fields[5]
    time_str = ''.join([item.strip() for item in date_fields[3].split(':')])
    return year_str + mon_str + day_str + time_str


def get_tw_nars(tw_json, d_nar_to_code):
    '''
    We only use supervised narratives. Manual narratives are not exactly consistent with supervised ones.
    Particularly, 'assembly' only appears in manual not in supervised. This is weird.
    '''
    if 'extension' in tw_json and tw_json['extension'] is not None:
        # and str(tw_json['extension']) != 'nan':
        # if 'manual_narratives' in tw_json['extension'] \
        #         and tw_json['extension']['manual_narratives'] is not None \
        #         and str(tw_json['extension']['manual_narratives']) != 'nan':
        #     l_nar_strs = tw_json['extension']['manual_narratives']
        #     l_nars = [d_nar_to_code[nar_str] for nar_str in l_nar_strs]
        #     return l_nars
        if 'supervised_narratives' in tw_json['extension'] \
                and tw_json['extension']['supervised_narratives'] is not None:
                # and str(tw_json['extension']['supervised_narratives']) != 'nan':
            l_nar_strs = tw_json['extension']['supervised_narratives']
            l_nars = [d_nar_to_code[nar_str] for nar_str in l_nar_strs]
            return l_nars
        else:
            return None
    else:
        return None


def get_tw_stances(tw_json):
    if 'extension' in tw_json and tw_json['extension'] is not None:
        if 'supervised_stance' in tw_json['extension'] and tw_json['extension']['supervised_stance'] is not None:
            return [tw_json['extension']['supervised_stance']['am'],
                    tw_json['extension']['supervised_stance']['pm'],
                    tw_json['extension']['supervised_stance']['?']]
        else:
            return None
    else:
        return None


def get_tw_sentiments(tw_json):
    if 'extension' in tw_json and tw_json['extension'] is not None:
        if 'sentiment_scores' in tw_json['extension'] \
                and tw_json['extension']['sentiment_scores'] is not None \
                and tw_json['extension']['sentiment_scores'] != 'N/A':
            return [tw_json['extension']['sentiment_scores']['positive'],
                    tw_json['extension']['sentiment_scores']['negative'],
                    tw_json['extension']['sentiment_scores']['neutral']]
        else:
            return None
    else:
        return None


############################################################
#   Tweet Object Parsing
############################################################
def tw_raw_data_to_parsed_db():
    '''
    Read in Twitter raw data, extract tw_id, tw_type, usr_id, tw_src_id, tw_datetime and raw_txt, and store these
    into a database.
    '''
    timer_start = time.time()
    db_conn = sqlite3.connect(global_settings.g_tw_en_db)
    db_cur = db_conn.cursor()
    sql_str = """create table if not exists ven_tw_en (tw_id text primary key, usr_id text not null, 
                tw_type text not null, tw_src_id text, tw_datetime text, raw_txt text)"""
    db_cur.execute(sql_str)
    s_rec_ids = set([])

    total_cnt = 0
    rec_cnt = 0
    sql_str = """insert into ven_tw_en (tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt) 
                values (?, ?, ?, ?, ?, ?)"""
    for data_file in global_settings.g_tw_raw_data_file_list:
        data_path = global_settings.g_tw_raw_data_folder + data_file
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
                            and str(tw_json['extension']) != 'nan'
                            and 'google_translation_m' in tw_json['extension']
                            and str(tw_json['extension']['google_translation_m']) != 'nan'
                            and tw_json['extension']['google_translation_m'] is not None):
                    tw_usr = get_tw_usr_id(tw_json)
                    tw_type = get_tw_type(tw_json)
                    tw_src_id = get_tw_src_id(tw_json, tw_type)
                    tw_datetime = get_tw_datetime(tw_json)
                    tw_raw_txt = get_tw_raw_txt(tw_json, tw_type, tw_lang)
                    if tw_id is None or tw_lang is None or tw_usr is None or tw_type is None or tw_datetime is None or tw_raw_txt is None:
                        continue
                    db_cur.execute(sql_str, (tw_id, tw_usr, tw_type, tw_src_id, tw_datetime, tw_raw_txt))
                    s_rec_ids.add(tw_id)
                    rec_cnt += 1
                if rec_cnt % 10000 == 0 and rec_cnt >= 10000:
                    db_conn.commit()
                    index_sql_str = '''drop index if exists idx_ven_tw_en_tw_id'''
                    db_cur.execute(index_sql_str)
                    db_conn.commit()
                    index_sql_str = '''create index idx_ven_tw_en_tw_id on ven_tw_en(tw_id)'''
                    db_cur.execute(index_sql_str)
                    db_conn.commit()
                    logging.debug('[tw_raw_data_to_parsed_db]: %s recs out of %s tws in %s secs.'
                                  % (rec_cnt, total_cnt, time.time() - timer_start))
            in_fd.close()
            db_conn.commit()
            index_sql_str = '''drop index if exists idx_ven_tw_en_tw_id'''
            db_cur.execute(index_sql_str)
            db_conn.commit()
            index_sql_str = '''create index idx_ven_tw_en_tw_id on ven_tw_en(tw_id)'''
            db_cur.execute(index_sql_str)
            db_conn.commit()
            logging.debug('[tw_raw_data_to_parsed_db]: %s recs out of %s tws in %s secs.'
                          % (rec_cnt, total_cnt, time.time() - timer_start))
    logging.debug('[tw_raw_data_to_parsed_db]: %s recs out of %s tws in %s secs.' %
                  (rec_cnt, total_cnt, str(time.time() - timer_start)))
    db_conn.close()
    logging.debug('[tw_raw_data_to_parsed_db]: All done in %s secs.' % str(time.time() - timer_start))


# @profile
def parse_one_tw_rec(tw_json, d_nar_to_code):
    if tw_json is None:
        raise Exception('[parse_one_tw_rec] tw_json is None.')
    tw_lang = get_tw_lang(tw_json)
    tw_id = get_tw_id(tw_json)
    if tw_lang == 'en' \
            or (tw_lang != 'en' and 'extension' in tw_json
                # and str(tw_json['extension']) != 'nan'
                and 'google_translation_m' in tw_json['extension']
                # and str(tw_json['extension']['google_translation_m']) != 'nan'
                and tw_json['extension']['google_translation_m'] is not None):
        tw_usr = get_tw_usr_id(tw_json)
        if tw_usr is None:
            return None
        tw_type = get_tw_type(tw_json)
        if tw_type is None:
            return None
        tw_src_id = get_tw_src_id(tw_json, tw_type)
        if tw_type != 'n' and tw_src_id is None:
            return None
        tw_datetime = get_tw_datetime(tw_json)
        if tw_datetime is None:
            return None
        tw_raw_txt = get_tw_raw_txt(tw_json, tw_type, tw_lang)
        if tw_raw_txt is None:
            return None
        l_nars = get_tw_nars(tw_json, d_nar_to_code)
        if l_nars is None or len(l_nars) <= 0:
            return None
        return (tw_id, tw_usr, tw_type, tw_src_id, tw_datetime, tw_raw_txt, l_nars)
    else:
        return None


# @profile
def tw_raw_data_parse_single_thread(l_raw_recs, d_nar_to_code, t_id):
    logging.debug('[tw_raw_data_parse_single_thread] Thread %s: Starts with %s raw recs.' % (t_id, len(l_raw_recs)))
    timer_start = time.time()

    l_ready_recs = []
    valid_cnt = 1
    for raw_rec in l_raw_recs:
        try:
            tw_json = json.loads(raw_rec)
            tw_parsed_fields = parse_one_tw_rec(tw_json, d_nar_to_code)
            if tw_parsed_fields is not None:
                l_ready_recs.append(tw_parsed_fields)
                valid_cnt += 1
            else:
                continue
            if valid_cnt % 10000 == 0 and valid_cnt >= 10000:
                logging.debug('[tw_raw_data_parse_single_thread] Thread %s: valid_cnt = %s in %s secs.'
                              % (t_id, valid_cnt, time.time() - timer_start))
        except Exception as err:
            logging.error('[tw_raw_data_parse_single_thread] Thread %s: Exception %s @tw_id = %s'
                          % (t_id, err, get_tw_id(tw_json)))
            traceback.print_exc()
            pass
    out_df = pd.DataFrame(l_ready_recs, columns=['tw_id', 'usr_id', 'tw_type', 'tw_src_id', 'tw_datetime', 'raw_txt', 'l_nars'])
    out_df.to_pickle(global_settings.g_tw_raw_data_int_file_format.format(str(t_id)))

    logging.debug('[tw_raw_data_parse_single_thread] Thread %s: All done %s secs.'
                  % (t_id, time.time() - timer_start))


def tw_raw_data_parse_multithread(raw_rec_path, d_nar_to_code, num_threads, job_id):
    logging.debug('[tw_raw_data_parse_multithread] Starts with %s' % raw_rec_path)
    timer_start = time.time()
    num_threads = int(num_threads)

    with open(raw_rec_path, 'r') as in_fd:
        l_raw_recs = in_fd.readlines()
        in_fd.close()
    num_raw_recs = len(l_raw_recs)
    logging.debug('[tw_raw_data_parse_multithread] Load in %s with %s raw recs in % secs'
                  % (raw_rec_path, num_raw_recs, time.time() - timer_start))

    batch_size = math.ceil(num_raw_recs / num_threads)
    l_tasks = []
    for i in range(0, num_raw_recs, batch_size):
        if i + batch_size < num_raw_recs:
            l_tasks.append(l_raw_recs[i:i + batch_size])
        else:
            l_tasks.append(l_raw_recs[i:])
    logging.debug('[tw_raw_data_parse_multithread] %s tasks to go.' % str(len(l_tasks)))

    l_threads = []
    t_id = 0
    for l_each_batch in l_tasks:
        t = threading.Thread(target=tw_raw_data_parse_single_thread,
                             args=(l_each_batch, d_nar_to_code, str(job_id) + '_' + str(t_id)))
        t.setName('t_mul_task_' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('[tw_raw_data_parse_multithread] Thread %s is finished.' % t.getName())
    logging.debug('[tw_raw_data_parse_multithread] %s done in %s sec.'
                  % (raw_rec_path, str(time.time() - timer_start)))


def tw_raw_data_parse_wrapper(num_threads, job_id, l_sel_ds=None):
    logging.debug('[tw_raw_data_parse_wrapper] Starts Job %s...' % str(job_id))
    timer_start = time.time()

    with open(global_settings.g_tw_narrative_to_code_file, 'r') as in_fd:
        d_nar_to_code = json.load(in_fd)
        in_fd.close()
    logging.debug('[tw_raw_data_parse_wrapper] Load in g_tw_narrative_to_code_file')

    raw_data_id = 0
    for filename in global_settings.g_tw_raw_data_file_list:
        if l_sel_ds is not None:
            if filename not in l_sel_ds:
                continue
        dirpath = global_settings.g_tw_raw_data_folder
        tw_raw_data_parse_multithread(dirpath + filename, d_nar_to_code, num_threads, job_id + '_' + str(raw_data_id))
        raw_data_id += 1
    logging.debug('[tw_raw_data_parse_wrapper] Job %s: All done in %s secs' % (job_id, time.time() - timer_start))


def tw_raw_data_int_to_db(ds_name):
    logging.debug('[tw_raw_data_int_to_db] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = """create table if not exists cp4.mf3jh_ven_tw_en_all (tw_id char(22) primary key, usr_id char(22), 
    tw_type char(1), tw_src_id char(22), tw_datetime char(14), raw_txt text, clean_txt text, nars integer array, tref boolean);"""
    tw_db_cur.execute(tw_db_sql_str)
    tw_db_conn.commit()

    tw_db_sql_str = """insert into cp4.mf3jh_ven_tw_en_all (tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt, clean_txt, nars, tref)
    values (%s,%s,%s,%s,%s,%s,%s,%s, %s) on conflict do nothing"""

    l_tw_ids = []
    cnt = 0
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_raw_data_int_folder):
        for filename in filenames:
            if filename[:14] != 'ven_tw_en_int_' or filename[-7:] != '.pickle':
                continue
            df_recs = pd.read_pickle(dirpath + filename)
            for rec in df_recs.values:
                tw_id = rec[0]
                usr_id = rec[1]
                tw_type = rec[2]
                tw_src_id = rec[3]
                tw_datetime = rec[4]
                raw_txt = rec[5]
                nars = rec[6]
                tw_db_cur.execute(tw_db_sql_str, (tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt, None, nars, False))
                l_tw_ids.append(tw_id)
                cnt += 1
                if cnt % 50000 == 0 and cnt >= 50000:
                    tw_db_conn.commit()
                    logging.debug('[tw_raw_data_int_to_db] %s recs committed in %s secs.'
                                  % (cnt, time.time() - timer_start))
    tw_db_conn.commit()
    logging.debug('[tw_raw_data_int_to_db] All %s recs committed in %s secs.'
                  % (cnt, time.time() - timer_start))
    tw_db_cur.close()
    tw_db_conn.close()

    with open(global_settings.g_tw_raw_data_int_tw_ids_file_format.format(ds_name), 'w+') as out_fd:
        out_str = '\n'.join(l_tw_ids)
        out_fd.write(out_str)
        out_fd.close()
    logging.debug('[tw_raw_data_int_to_db] All %s recs committed in %s secs.'
                  % (cnt, time.time() - timer_start))


def update_tref(l_tw_ids):
    logging.debug('[update_tref] Starts with %s tw_ids...' % str(len(l_tw_ids)))
    timer_start = time.time()

    s_unk_tw_ids = set(l_tw_ids)
    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = '''select tw_id, tw_type, tw_src_id from cp4.mf3jh_ven_tw_en_all'''
    tw_db_cur.execute(tw_db_sql_str)
    l_recs = tw_db_cur.fetchall()
    s_exist_tw_ids = set([rec[0] for rec in l_recs])

    tw_db_up_sql_str = '''update cp4.mf3jh_ven_tw_en_all set tref = true where tw_id = %s'''
    cnt = 0
    for rec in l_recs:
        tw_id = rec[0]
        if tw_id not in s_unk_tw_ids:
            continue
        tw_type = rec[1]
        tw_src_id = rec[2]
        if tw_type == 't' and tw_src_id in s_exist_tw_ids:
            tw_db_cur.execute(tw_db_up_sql_str, (tw_id,))
            cnt += 1
            if cnt % 10000 == 0 and cnt >= 10000:
                tw_db_conn.commit()
                logging.debug('[update_tref] %s recs updated in %s secs.' % (cnt, time.time() - timer_start))
    tw_db_conn.commit()
    logging.debug('[update_tref] All %s recs updated in %s secs.' % (cnt, time.time() - timer_start))

    tw_db_cur.close()
    tw_db_conn.close()


############################################################
#   Data Table for Users
############################################################
def udt_parse_one_tw_rec(tw_json, d_nar_to_code, dt_start, dt_end):
    if tw_json is None:
        raise Exception('[udt_parse_one_tw_rec] tw_json is None.')
    tw_id = get_tw_id(tw_json)
    tw_usr = get_tw_usr_id(tw_json)
    if tw_usr is None:
        return None
    tw_type = get_tw_type(tw_json)
    if tw_type is None:
        return None
    tw_src_id = get_tw_src_id(tw_json, tw_type)
    tw_datetime = get_tw_datetime(tw_json)
    if tw_datetime is None:
        return None
    if dt_start is not None and tw_datetime < dt_start:
        return None
    if dt_end is not None and tw_datetime > dt_end:
        return None
    l_nars = get_tw_nars(tw_json, d_nar_to_code)
    if l_nars is not None and len(l_nars) <= 0:
        l_nars = None
    l_stances = get_tw_stances(tw_json)
    l_sentiments = get_tw_sentiments(tw_json)
    return (tw_id, tw_usr, tw_type, tw_src_id, tw_datetime, l_nars, l_stances, l_sentiments)


def udt_tw_raw_data_parse_single_thread(l_raw_recs, d_nar_to_code, dt_start, dt_end, t_id):
    logging.debug('[udt_tw_raw_data_parse_single_thread] Thread %s: Starts with %s raw recs.' % (t_id, len(l_raw_recs)))
    timer_start = time.time()

    l_ready_recs = []
    valid_cnt = 1
    for raw_rec in l_raw_recs:
        try:
            tw_json = json.loads(raw_rec)
            tw_parsed_fields = udt_parse_one_tw_rec(tw_json, d_nar_to_code, dt_start, dt_end,)
            if tw_parsed_fields is not None:
                l_ready_recs.append(tw_parsed_fields)
                valid_cnt += 1
            else:
                continue
            if valid_cnt % 10000 == 0 and valid_cnt >= 10000:
                logging.debug('[udt_tw_raw_data_parse_single_thread] Thread %s: valid_cnt = %s in %s secs.'
                              % (t_id, valid_cnt, time.time() - timer_start))
        except Exception as err:
            logging.error('[udt_tw_raw_data_parse_single_thread] Thread %s: Exception %s @tw_id = %s'
                          % (t_id, err, get_tw_id(tw_json)))
            traceback.print_exc()
            pass
    out_df = pd.DataFrame(l_ready_recs, columns=['tw_id', 'usr_id', 'tw_type', 'tw_src_id', 'tw_datetime', 'l_nars',
                                                 'l_stances', 'l_sentiments'])
    out_df.to_pickle(global_settings.g_udt_tw_raw_data_int_file_format.format(str(t_id)))

    logging.debug('[udt_tw_raw_data_parse_single_thread] Thread %s: All done %s secs.'
                  % (t_id, time.time() - timer_start))


def udt_tw_raw_data_parse_multithread(raw_rec_path, d_nar_to_code, dt_start, dt_end, num_threads, job_id):
    logging.debug('[udt_tw_raw_data_parse_multithread] Starts with %s' % raw_rec_path)
    timer_start = time.time()
    num_threads = int(num_threads)

    with open(raw_rec_path, 'r') as in_fd:
        l_raw_recs = in_fd.readlines()
        in_fd.close()
    num_raw_recs = len(l_raw_recs)
    logging.debug('[udt_tw_raw_data_parse_multithread] Load in %s with %s raw recs in % secs'
                  % (raw_rec_path, num_raw_recs, time.time() - timer_start))

    batch_size = math.ceil(num_raw_recs / int(num_threads))
    l_tasks = []
    for i in range(0, num_raw_recs, batch_size):
        if i + batch_size < num_raw_recs:
            l_tasks.append(l_raw_recs[i:i + batch_size])
        else:
            l_tasks.append(l_raw_recs[i:])
    logging.debug('[udt_tw_raw_data_parse_multithread] %s tasks to go.' % str(len(l_tasks)))

    l_threads = []
    t_id = 0
    for l_each_batch in l_tasks:
        t = threading.Thread(target=udt_tw_raw_data_parse_single_thread,
                             args=(l_each_batch, d_nar_to_code, dt_start, dt_end, str(job_id) + '_' + str(t_id)))
        t.setName('t_mul_task_' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('[udt_tw_raw_data_parse_multithread] Thread %s is finished.' % t.getName())
    logging.debug('[udt_tw_raw_data_parse_multithread] %s done in %s sec.'
                  % (raw_rec_path, str(time.time() - timer_start)))


def udt_tw_raw_data_parse_wrapper(num_threads, job_id, dt_start, dt_end, l_skips=None):
    logging.debug('[udt_tw_raw_data_parse_wrapper] Starts Job %s...' % str(job_id))
    timer_start = time.time()

    with open(global_settings.g_tw_narrative_to_code_file, 'r') as in_fd:
        d_nar_to_code = json.load(in_fd)
        in_fd.close()
    logging.debug('[udt_tw_raw_data_parse_wrapper] Load in g_tw_narrative_to_code_file')

    raw_data_id = 0
    for filename in global_settings.g_tw_raw_data_file_list:
        if l_skips is not None:
            if filename in l_skips:
                raw_data_id += 1
                continue
        dirpath = global_settings.g_tw_raw_data_folder
        udt_tw_raw_data_parse_multithread(dirpath + filename,
                                          d_nar_to_code,
                                          dt_start,
                                          dt_end,
                                          num_threads,
                                          job_id + '_' + str(raw_data_id))
        raw_data_id += 1
    logging.debug('[udt_tw_raw_data_parse_wrapper] Job %s: All done in %s secs' % (job_id, time.time() - timer_start))


def build_udt_tw_data():
    logging.debug('[build_udt_tw_data] Starts...')
    timer_start = time.time()
    l_tw_recs = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_udt_int_folder):
        for filename in filenames:
            if filename[:15] != 'udt_ven_tw_int_' or filename[-7:] != '.pickle':
                continue
            df_tw_recs = pd.read_pickle(dirpath + filename)
            l_tw_recs.append(df_tw_recs)
    df_tw_recs_all = pd.concat(l_tw_recs)
    df_tw_recs_all.to_pickle(global_settings.g_udt_tw_data_file)
    logging.debug('[build_udt_tw_data] Output udt_tw_raw_data_int_to_out in %s secs.' % str(time.time() - timer_start))


def build_udt_tw_src_trg_data_single_thread(l_tasks, d_udt_tw_data, t_id):
    logging.debug('[build_udt_tw_src_trg_data_single_thread] Thread %s: Starts with %s tasks...'
                  % (str(t_id), len(l_tasks)))
    timer_start = time.time()

    # 'tw_id', 'usr_id', 'tw_type', 'tw_src_id', 'tw_datetime', 'l_nars','l_stances', 'l_sentiments'
    l_ready_recs = []
    cnt = 0
    for tw_id in l_tasks:
        tw_data = d_udt_tw_data[tw_id]
        trg_tw_id = tw_id
        trg_usr_id = tw_data['usr_id']
        trg_tw_type = tw_data['tw_type']
        trg_tw_datetime = tw_data['tw_datetime']
        trg_l_nars = tw_data['l_nars']
        trg_l_stances = tw_data['l_stances']
        trg_l_sentiments = tw_data['l_sentiments']
        trg_usr_com_id = None
        trg_usr_mid = None
        trg_usr_com_member = None

        src_tw_id = tw_data['tw_src_id']
        # src_tw_data = df_udt_tw_data.loc[src_tw_id] if src_tw_id in df_udt_tw_data.index else None
        if src_tw_id in d_udt_tw_data:
            src_tw_data = d_udt_tw_data[src_tw_id]
            src_tw_type = src_tw_data['tw_type']
            src_usr_id = src_tw_data['usr_id']
            src_tw_datetime = src_tw_data['tw_datetime']
            src_l_nars = src_tw_data['l_nars']
            src_l_stances = src_tw_data['l_stances']
            src_l_sentiments = src_tw_data['l_sentiments']
        else:
            src_tw_type = None
            src_usr_id = None
            src_tw_datetime = None
            src_l_nars = None
            src_l_stances = None
            src_l_sentiments = None
        src_usr_com_id = None
        src_usr_mid = None
        src_usr_com_member = None

        ready_rec = (trg_tw_id, trg_usr_id, trg_usr_mid, trg_usr_com_id, trg_usr_com_member, trg_tw_type,
                     trg_tw_datetime, trg_l_nars, trg_l_stances, trg_l_sentiments,
                     src_tw_id, src_usr_id, src_usr_mid, src_usr_com_id, src_usr_com_member, src_tw_type,
                     src_tw_datetime, src_l_nars, src_l_stances, src_l_sentiments)
        l_ready_recs.append(ready_rec)
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[build_udt_tw_src_trg_data_single_thread] Thread %s: %s ready_recs in %s secs.'
                          % (t_id, cnt, time.time() - timer_start))
    logging.debug('[build_udt_tw_src_trg_data_single_thread] Thread %s: %s ready_recs in %s secs.'
                  % ( t_id, cnt, time.time() - timer_start))

    df_out = pd.DataFrame(l_ready_recs, columns=['trg_tw_id', 'trg_usr_id', 'trg_usr_mid', 'trg_usr_com_id',
                                                 'trg_usr_com_member', 'trg_tw_type', 'trg_tw_datetime', 'trg_l_nars',
                                                 'trg_l_stances', 'trg_l_sentiments',
                                                 'src_tw_id', 'src_usr_id', 'src_usr_mid', 'src_usr_com_id',
                                                 'src_usr_com_member', 'src_tw_type', 'src_tw_datetime', 'src_l_nars',
                                                 'src_l_stances', 'src_l_sentiments'])
    df_out.to_pickle(global_settings.g_udt_tw_srg_trg_data_int_file_format.format(str(t_id)))
    logging.debug('[build_udt_tw_src_trg_data_single_thread] Thread %s: All done in %s secs.'
                  % (t_id, str(time.time() - timer_start)))


# @profile
def build_udt_tw_src_trg_data_multithread(num_threads, job_id):
    logging.debug('[build_udt_tw_src_trg_data_multithread] Starts...')
    timer_start = time.time()

    df_udt_tw_data = pd.read_pickle(global_settings.g_udt_tw_data_file)
    df_udt_tw_data = df_udt_tw_data.set_index('tw_id')
    logging.debug('[build_udt_tw_src_trg_data_multithread] Load g_udt_tw_data_file with %s recs in %s secs.'
                  % (len(df_udt_tw_data), time.time() - timer_start))

    d_udt_tw_data = dict()
    for tw_id, tw_data in df_udt_tw_data.iterrows():
        d_udt_tw_data[tw_id] = tw_data
    num_tasks = len(d_udt_tw_data)
    logging.debug('[build_udt_tw_src_trg_data_multithread] Convert df_udt_tw_data to d_udt_tw_data in %s secs.'
                  % str(time.time() - timer_start))

    l_task_tw_ids = list(d_udt_tw_data.keys())
    batch_size = math.ceil(num_tasks / int(num_threads))
    l_tasks = []
    for i in range(0, num_tasks, batch_size):
        if i + batch_size < num_tasks:
            l_tasks.append(l_task_tw_ids[i:i + batch_size])
        else:
            l_tasks.append(l_task_tw_ids[i:])
    logging.debug('[build_udt_tw_src_trg_data_multithread] %s tasks to go.' % str(len(l_tasks)))

    l_threads = []
    t_id = 0
    for batch in l_tasks:
        t = threading.Thread(target=build_udt_tw_src_trg_data_single_thread,
                             args=(batch, d_udt_tw_data, str(job_id) + '_' + str(t_id)))
        t.setName('t_mul_task_' + str(job_id) + '_' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('[build_udt_tw_src_trg_data_multithread] Thread %s is finished.' % t.getName())
    logging.debug('[build_udt_tw_src_trg_data_multithread] All done in %s sec.'
                  % str(time.time() - timer_start))


def udt_tw_src_trg_data_int_to_out():
    logging.debug('[udt_tw_src_trg_data_int_to_out] Starts...')
    timer_start = time.time()

    l_udt_tw_src_trg_df = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_udt_tw_srg_trg_data_int_folder):
        for filename in filenames:
            if filename[:24] != 'udt_tw_src_trg_data_int_' or filename[-7:] != '.pickle':
                continue
            df_udt_tw_src_trg = pd.read_pickle(dirpath + filename)
            l_udt_tw_src_trg_df.append(df_udt_tw_src_trg)

    df_out = pd.concat(l_udt_tw_src_trg_df)
    df_out.to_pickle(global_settings.g_udt_tw_src_trg_data_file)
    logging.debug('[udt_tw_src_trg_data_int_to_out] All done in %s secs.' % str(time.time() - timer_start))


def fill_mdid_com_id_to_udt_tw_srg_trg_data():
    logging.debug('[fill_mdid_com_id_to_udt_tw_srg_trg_data] Starts...')
    timer_start = time.time()

    df_udt_tw_src_trg = pd.read_pickle(global_settings.g_udt_tw_src_trg_data_file)
    logging.debug('[fill_mdid_com_id_to_udt_tw_srg_trg_data] Load g_udt_tw_src_trg_data_file with %s recs in %s secs.'
                  % (len(df_udt_tw_src_trg), time.time() - timer_start))

    df_usr_com_info = get_usr_com_info()
    d_usr_com_info = df_usr_com_info.set_index('usr_id').to_dict(orient='index')
    cnt = 0
    for idx, src_trg_rec in df_udt_tw_src_trg.iterrows():
        trg_usr_id = src_trg_rec['trg_usr_id']
        if trg_usr_id in d_usr_com_info:
            src_trg_rec['trg_usr_mid'] = d_usr_com_info[trg_usr_id]['mid']
            src_trg_rec['trg_usr_com_id'] = d_usr_com_info[trg_usr_id]['com_id']
            src_trg_rec['trg_usr_com_member'] = d_usr_com_info[trg_usr_id]['com_member']
        src_usr_id = src_trg_rec['src_usr_id']
        if src_usr_id in d_usr_com_info:
            src_trg_rec['src_usr_mid'] = d_usr_com_info[src_usr_id]['mid']
            src_trg_rec['src_usr_com_id'] = d_usr_com_info[src_usr_id]['com_id']
            src_trg_rec['src_usr_com_member'] = d_usr_com_info[src_usr_id]['com_member']
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[fill_mdid_com_id_to_udt_tw_srg_trg_data] %s recs scanned in %s secs.'
                          % (cnt, time.time() - timer_start))
    logging.debug('[fill_mdid_com_id_to_udt_tw_srg_trg_data] %s recs scanned in %s secs.'
                  % (cnt, time.time() - timer_start))
    df_udt_tw_src_trg.to_pickle(global_settings.g_udt_tw_src_trg_data_file)
    logging.debug('[fill_mdid_com_id_to_udt_tw_srg_trg_data] All done in %s secs.' % str(time.time() - timer_start))


def udt_tw_srg_trg_data_to_db():
    logging.debug('[udt_tw_srg_trg_data_to_db] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = """create table if not exists cp4.mf3jh_udt_tw_src_trg_data 
                        (
                            trg_tw_id char(22) primary key, 
                            trg_usr_id char(22), 
                            trg_usr_mid text, 
                            trg_usr_com_id text,
                            trg_usr_com_member char(1), 
                            trg_tw_type char(1), 
                            trg_tw_datetime char(14), 
                            trg_l_nars integer[],
                            trg_l_stances real[], 
                            trg_l_sentiments real[],
                            src_tw_id char(22), 
                            src_usr_id char(22), 
                            src_usr_mid text, 
                            src_usr_com_id text,
                            src_usr_com_member char(1), 
                            src_tw_type char(1), 
                            src_tw_datetime char(14), 
                            src_l_nars integer[],
                            src_l_stances real[], 
                            src_l_sentiments real[]
                        )"""
    tw_db_cur.execute(tw_db_sql_str)
    tw_db_conn.commit()

    tw_db_sql_str = """insert into cp4.mf3jh_udt_tw_src_trg_data (trg_tw_id, trg_usr_id, trg_usr_mid, trg_usr_com_id,
                        trg_usr_com_member, trg_tw_type, trg_tw_datetime, trg_l_nars, trg_l_stances, trg_l_sentiments,
                        src_tw_id, src_usr_id, src_usr_mid, src_usr_com_id, src_usr_com_member, src_tw_type, 
                        src_tw_datetime, src_l_nars, src_l_stances, src_l_sentiments) values (%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) on conflict do nothing"""

    df_udt_tw_src_trg = pd.read_pickle(global_settings.g_udt_tw_src_trg_data_file)
    logging.debug('[udt_tw_srg_trg_data_to_db] Load g_udt_tw_src_trg_data_file with %s recs in %s secs.'
                  % (len(df_udt_tw_src_trg), time.time() - timer_start))

    cnt = 0
    for idx, rec in df_udt_tw_src_trg.iterrows():
        trg_tw_id = rec['trg_tw_id']
        trg_usr_id = rec['trg_usr_id']
        trg_usr_mid = rec['trg_usr_mid']
        trg_usr_com_id = rec['trg_usr_com_id']
        trg_usr_com_member = rec['trg_usr_com_member']
        trg_tw_type = rec['trg_tw_type']
        trg_tw_datetime = rec['trg_tw_datetime']
        trg_l_nars = rec['trg_l_nars']
        trg_l_stances = rec['trg_l_stances']
        trg_l_sentiments = rec['trg_l_sentiments']
        src_tw_id = rec['src_tw_id']
        src_usr_id = rec['src_usr_id']
        src_usr_mid = rec['src_usr_mid']
        src_usr_com_id = rec['src_usr_com_id']
        src_usr_com_member = rec['src_usr_com_member']
        src_tw_type = rec['src_tw_type']
        src_tw_datetime = rec['src_tw_datetime']
        src_l_nars = rec['src_l_nars']
        src_l_stances = rec['src_l_stances']
        src_l_sentiments = rec['src_l_sentiments']
        tw_db_cur.execute(tw_db_sql_str, (trg_tw_id, trg_usr_id, trg_usr_mid, trg_usr_com_id, trg_usr_com_member,
                                          trg_tw_type, trg_tw_datetime, trg_l_nars, trg_l_stances, trg_l_sentiments,
                                          src_tw_id, src_usr_id, src_usr_mid, src_usr_com_id, src_usr_com_member,
                                          src_tw_type, src_tw_datetime, src_l_nars, src_l_stances, src_l_sentiments))
        cnt += 1
        if cnt % 100000 == 0 and cnt >= 100000:
            tw_db_conn.commit()
            logging.debug('[udt_tw_srg_trg_data_to_db] %s recs committed in %s secs.'
                          % (cnt, time.time() - timer_start))
    tw_db_conn.commit()
    logging.debug('[udt_tw_srg_trg_data_to_db] %s recs committed in %s secs.'
                  % (cnt, time.time() - timer_start))
    tw_db_cur.close()
    tw_db_conn.close()
    logging.debug('[udt_tw_srg_trg_data_to_db] All done.')


def nar_codes_to_nar_vec(l_nar_codes, nar_vec_len):
    nar_vec = np.zeros(nar_vec_len)
    for nar_code in l_nar_codes:
        nar_vec[int(nar_code)] = 1
    return nar_vec


def datetime_diff(dt_start_str, dt_end_str):
    dt_fmt = '%Y%m%d%H%M%S'
    dt_start = datetime.strptime(dt_start_str, dt_fmt)
    dt_end = datetime.strptime(dt_end_str, dt_fmt)
    if dt_end < dt_start:
        raise Exception('[datetime_diff] Invalid datetime duration, dt_start_str=%s, dt_end_str=%s'
                        % (dt_start_str, dt_end_str))
    return (dt_end - dt_start).total_seconds()


# @profile
def build_udt_usr_data_single_thread(task_id, nar_vec_len, p_id):
    logging.debug('[build_udt_usr_data_single_thread] Thread %s: Starts...' % p_id)
    timer_start = time.time()

    """'tw_id', 'usr_id', 'tw_type', 'tw_src_id', 'tw_datetime', 'l_nars', 'l_stances', 'l_sentiments'"""
    df_udt_tw_groupby_usr = pd.read_pickle(global_settings.g_udt_usr_data_task_file_format.format(str(task_id)))
    logging.debug('[build_udt_usr_data_single_thread] Load in task %s in %s secs: %s usrs to go.'
                  % (task_id, time.time() - timer_start, len(df_udt_tw_groupby_usr)))

    l_usr_recs = []
    cnt = 0
    for usr_df_rec in df_udt_tw_groupby_usr.values:
        usr_id = usr_df_rec[0]
        df_usr_data = usr_df_rec[1]
        tw_cnt = len(df_usr_data)
        tw_t_cnt = len(df_usr_data.loc[df_usr_data['tw_type'] == 't'])
        tw_r_cnt = len(df_usr_data.loc[df_usr_data['tw_type'] == 'r'])
        tw_q_cnt = len(df_usr_data.loc[df_usr_data['tw_type'] == 'q'])
        tw_n_cnt = len(df_usr_data.loc[df_usr_data['tw_type'] == 'n'])
        top_act_ratio = max(tw_t_cnt, tw_r_cnt, tw_q_cnt, tw_n_cnt) / float(tw_cnt)
        l_types = ['t', 'r', 'q', 'n']
        top_act_type = l_types[np.argmax([tw_t_cnt, tw_r_cnt, tw_q_cnt, tw_n_cnt])]
        l_stances = df_usr_data.loc[df_usr_data['l_stances'].notnull()]['l_stances']
        if len(l_stances) > 0:
            avg_stances = np.average(np.stack(l_stances.to_list()), axis=0)
            avg_am = avg_stances[0]
            avg_pm = avg_stances[1]
            avg_un = avg_stances[2]
        else:
            avg_am = None
            avg_pm = None
            avg_un = None
        l_sentiments = df_usr_data.loc[df_usr_data['l_sentiments'].notnull()]['l_sentiments']
        if len(l_sentiments) > 0:
            avg_sentiments = np.average(np.stack(l_sentiments.to_list()), axis=0)
            avg_pos = avg_sentiments[0]
            avg_neg = avg_sentiments[1]
            avg_neu = avg_sentiments[2]
        else:
            avg_pos = None
            avg_neg = None
            avg_neu = None
        nar_cnt = len(df_usr_data.loc[df_usr_data['l_nars'].notnull()])
        nar_miss_ratio = float(tw_cnt - nar_cnt) / tw_cnt
        l_nars = df_usr_data.loc[df_usr_data['l_nars'].notnull()]['l_nars']
        if len(l_nars) > 0:
            nar_sum_vec = np.sum(np.stack(l_nars.apply(nar_codes_to_nar_vec, args=(nar_vec_len,)).to_list()), axis=0)
        else:
            nar_sum_vec = None
        tw_dt_start = min(df_usr_data.loc[df_usr_data['tw_type'].notnull()]['tw_datetime'].to_list())
        tw_dt_end = max(df_usr_data.loc[df_usr_data['tw_type'].notnull()]['tw_datetime'].to_list())
        tw_dt_dur = datetime_diff(tw_dt_start, tw_dt_end)
        usr_rec = (usr_id, tw_cnt, tw_t_cnt, tw_r_cnt, tw_q_cnt, tw_n_cnt, top_act_type, top_act_ratio, avg_am, avg_pm,
                   avg_un, avg_pos, avg_neg, avg_neu, nar_sum_vec, nar_miss_ratio, tw_dt_start, tw_dt_end, tw_dt_dur)
        l_usr_recs.append(usr_rec)
        cnt += 1
        if cnt % 1000 == 0 and cnt > 1000:
            logging.debug('[build_udt_usr_data_single_thread] Thread %s: Build %s usr recs in %s secs.'
                          % (p_id, cnt, time.time() - timer_start))
    logging.debug('[build_udt_usr_data_single_thread] Thread %s: Build all %s usr recs in %s secs.'
                  % (p_id, cnt, time.time() - timer_start))

    df_udt_usr = pd.DataFrame(l_usr_recs, columns=['usr_id', 'tw_cnt', 'tw_t_cnt', 'tw_r_cnt', 'tw_q_cnt', 'tw_n_cnt',
                                               'top_act_type', 'top_act_ratio', 'avg_am', 'avg_pm', 'avg_un',
                                               'avg_pos', 'avg_neg', 'avg_neu', 'nar_sum_vec', 'nar_miss_ratio',
                                               'tw_dt_start', 'tw_dt_end', 'tw_dt_dur'])
    df_udt_usr.to_pickle(global_settings.g_udt_usr_data_int_file_format.format(str(p_id)))
    logging.debug('[build_udt_usr_data_single_thread] Thread %s: All done with %s recs in %s secs.'
                  % (p_id, len(df_udt_usr), time.time() - timer_start))


def build_udt_usr_data_multithread(num_threads, job_id, en_build_task=False):
    logging.debug('[build_udt_usr_data_multithread] Starts...')
    timer_start = time.time()

    if en_build_task:
        num_usr_ids = udt_tw_data_groupby_usr(num_threads)
        logging.debug('[build_udt_usr_data_multithread] %s tasks to go.' % str(num_usr_ids))
    else:
        num_usr_ids = int(num_threads)

    with open(global_settings.g_tw_code_to_narrative_file, 'r') as in_fd:
        d_code_to_nar = json.load(in_fd)
        in_fd.close()
    nar_vec_len = max([int(code) for code in list(d_code_to_nar.keys())]) + 1
    logging.debug('[build_udt_usr_data_multithread] nar_vec_len = %s' % str(nar_vec_len))

    l_procs = []
    p_id = 0
    for task_id in range(num_usr_ids):
        t = multiprocessing.Process(target=build_udt_usr_data_single_thread,
                                    args=(task_id, nar_vec_len, str(job_id) + '_' + str(p_id)))
        t.name = 't_mul_task_' + str(p_id)
        t.start()
        l_procs.append(t)
        p_id += 1

    while len(l_procs) > 0:
        for t in l_procs:
            if t.is_alive():
                t.join(1)
            else:
                l_procs.remove(t)
                logging.debug('[build_udt_usr_data_multithread] Thread %s is finished.' % t.name)
    logging.debug('[build_udt_usr_data_multithread] All done in %s sec.'
                  % str(time.time() - timer_start))


def udt_tw_data_groupby_usr(num_threads):
    logging.debug('[udt_tw_data_groupby_usr] Starts...')
    timer_start = time.time()

    df_tw_recs_all = pd.read_pickle(global_settings.g_udt_tw_data_file)
    l_usr_ids = list(set(df_tw_recs_all['usr_id'].to_list()))
    num_usr_ids = len(l_usr_ids)

    batch_size = math.ceil(num_usr_ids / int(num_threads))
    l_tasks = []
    for i in range(0, num_usr_ids, batch_size):
        if i + batch_size < num_usr_ids:
            l_tasks.append(l_usr_ids[i:i + batch_size])
        else:
            l_tasks.append(l_usr_ids[i:])
    logging.debug('[udt_tw_data_groupby_usr] Request %s tasks.' % str(len(l_tasks)))

    d_usr_groupby = {usr_id: [] for usr_id in l_usr_ids}
    logging.debug('[udt_tw_data_groupby_usr] Load g_udt_tw_data_file in %s secs.' % str(time.time() - timer_start))

    cnt = 0
    for tw_rec in df_tw_recs_all.values:
        usr_id = tw_rec[1]
        d_usr_groupby[usr_id].append(tw_rec)
        cnt += 1
        if cnt % 100000 == 0 and cnt >= 100000:
            logging.debug('[udt_tw_data_groupby_usr] Scan %s tw_recs in %s secs.' % (cnt, time.time() - timer_start))
    logging.debug('[udt_tw_data_groupby_usr] Scan All %s tw_recs in %s secs.' % (cnt, time.time() - timer_start))

    cnt = 0
    for usr_id in d_usr_groupby:
        df_udt_tw_per_usr = pd.DataFrame(d_usr_groupby[usr_id],
                                         columns=['tw_id', 'usr_id', 'tw_type', 'tw_src_id', 'tw_datetime', 'l_nars',
                                                 'l_stances', 'l_sentiments'])
        d_usr_groupby[usr_id] = df_udt_tw_per_usr
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[udt_tw_data_groupby_usr] %s usr dfs ready in %s secs.' % (cnt, time.time() - timer_start))
    logging.debug('[udt_tw_data_groupby_usr] All %s usr dfs ready in %s secs.' % (cnt, time.time() - timer_start))

    for task_id, l_task_usr_ids in enumerate(l_tasks):
        l_task_recs = []
        for usr_id in l_task_usr_ids:
            task_rec = (usr_id, d_usr_groupby[usr_id])
            l_task_recs.append(task_rec)
        df_task = pd.DataFrame(l_task_recs, columns=['usr_id', 'udt_tw'])
        df_task.to_pickle(global_settings.g_udt_usr_data_task_file_format.format(str(task_id)))
        logging.debug('[udt_tw_data_groupby_usr] Output df_task:%s in %s secs.' % (task_id, time.time() - timer_start))

    logging.debug('[udt_tw_data_groupby_usr] All done in %s secs.' % str(time.time() - timer_start))
    return len(l_tasks)


def udt_usr_data_int_to_out():
    logging.debug('[udt_usr_data_int_to_out] Starts...')
    timer_start = time.time()
    l_udt_usr_dfs = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_udt_int_folder):
        for filename in filenames:
            if filename[:17] != 'udt_usr_data_int_' or filename[-7:] != '.pickle':
                continue
            df_udt_usr = pd.read_pickle(dirpath + filename)
            l_udt_usr_dfs.append(df_udt_usr)
    out_df = pd.concat(l_udt_usr_dfs)
    out_df.to_pickle(global_settings.g_udt_usr_data_file)
    logging.debug('[udt_usr_data_int_to_out] All done in %s secs.' % str(time.time() - timer_start))


def get_usr_com_info(com_id):
    logging.debug('[get_usr_com_info] Starts...')
    timer_start = time.time()

    # l_recs = []
    # for com_file in global_settings.g_udt_community_file_list:
    #     com_id = com_file[10:-10]
    #     with open(global_settings.g_udt_community_folder + com_file, 'r') as in_fd:
    #         csv_reader = csv.reader(in_fd, delimiter=',')
    #         row_cnt = 0
    #         for row in csv_reader:
    #             if row_cnt == 0:
    #                 row_cnt += 1
    #                 continue
    #             usr_id = row[0]
    #             mid = row[1]
    #             com_member = row[2]
    #             l_recs.append((usr_id, mid, com_id, com_member))
    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = """select id, community from cp4.sna where community = %s"""
    tw_db_cur.execute(sql_str, (str(com_id),))
    l_recs = tw_db_cur.fetchall()
    logging.debug('[get_usr_com_info] Fetch %s recs in %s secs.' % (len(l_recs), time.time() - timer_start))

    # df_usr_com_info = pd.DataFrame(l_recs, columns=['usr_id', 'mid', 'com_id', 'com_member'])
    df_usr_com_info = pd.DataFrame(l_recs, columns=['usr_id', 'com_id'])
    # df_usr_com_info.drop_duplicates(subset=['usr_id'], inplace=True, ignore_index=True)
    logging.debug('[get_usr_com_info] All done with %s usr com recs in %s secs.'
                  % (len(df_usr_com_info), time.time() - timer_start))
    return df_usr_com_info


def build_udt_for_community(df_usr_com_info, com_id):
    logging.debug('[build_udt_for_community] Starts with %s usrs considered...' % str(len(df_usr_com_info)))
    timer_start = time.time()

    df_udt_usr = pd.read_pickle(global_settings.g_udt_usr_data_file)
    logging.debug('[build_udt_for_community] Load g_udt_usr_data_file with %s usrs in %s secs.'
                  % (len(df_udt_usr), str(time.time() - timer_start)))

    df_udt_com = df_udt_usr.join(df_usr_com_info.set_index('usr_id'), on='usr_id', how='inner')
    df_udt_com.to_pickle(global_settings.g_udt_com_data_file_format.format(com_id))
    logging.debug('[build_udt_for_community] All done with %s usrs in %s secs.'
                  % (len(df_udt_com), time.time() - timer_start))


def udt_to_db(com_id):
    logging.debug('[udt_to_db] Starts...')

    df_udt = pd.read_pickle(global_settings.g_udt_com_data_file_format.format(str(com_id)))
    df_udt['nar_sum_vec'] = df_udt['nar_sum_vec'].apply(lambda v: v.tolist() if v is not None else None)
    logging.debug('[udt_to_db] Load g_udt_com_data_file with %s recs.' % str(len(df_udt)))

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = """create table if not exists cp4.mf3jh_udt_{0}
                (
                    usr_id char(22) primary key, 
                    tw_cnt integer, 
                    tw_t_cnt integer,
                    tw_r_cnt integer, 
                    tw_q_cnt integer, 
                    tw_n_cnt integer,
                    top_act_type char(1), 
                    top_act_ratio real, 
                    avg_am real, 
                    avg_pm real, 
                    avg_un real,
                    avg_pos real,
                    avg_neg real, 
                    avg_neu real, 
                    nar_sum_vec integer array, 
                    nar_miss_ratio real,
                    tw_dt_start char(14), 
                    tw_dt_end char(14), 
                    tw_dt_dur integer, 
                    com_id text
                );""".format(str(com_id))
    tw_db_cur.execute(sql_str)
    tw_db_conn.commit()
    sql_str = """insert into cp4.mf3jh_udt_{0} (usr_id, tw_cnt, tw_t_cnt, tw_r_cnt, tw_q_cnt, tw_n_cnt, top_act_type, 
                top_act_ratio, avg_am, avg_pm, avg_un, avg_pos, avg_neg, avg_neu, nar_sum_vec, nar_miss_ratio,
                tw_dt_start, tw_dt_end, tw_dt_dur, com_id) 
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""".format(str(com_id))
    cnt = 0
    for rec in df_udt.values.tolist():
        tw_db_cur.execute(sql_str, rec)
        cnt += 1
    tw_db_conn.commit()
    logging.debug('[udt_to_db] Commit %s recs.' % cnt)
    tw_db_cur.close()
    tw_db_conn.close()
    logging.debug('[udt_to_db] All done with %s recs.' % str(cnt))


def udt_tw_src_trg_data_with_sna_com_id():
    logging.debug('[udt_tw_src_trg_data_with_sna_com_id] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()

    sql_str = """select id, community from cp4_pipeline.updated_sna"""
    tw_db_cur.execute(sql_str)
    l_sna_recs = tw_db_cur.fetchall()
    d_sna = dict()
    for sna_rec in l_sna_recs:
        usr_id = sna_rec[0]
        com_id = sna_rec[1]
        if com_id is not None:
            d_sna[usr_id] = str(int(com_id))
    logging.debug('[udt_tw_src_trg_data_with_sna_com_id] Load sna with %s eff usrs in %s secs.'
                  % (len(d_sna), time.time() - timer_start))

    sql_str = """select 
                       trg_tw_id, 
                       trg_usr_id, 
                       src_usr_id 
                from cp4.mf3jh_udt_tw_src_trg_data"""
    tw_db_cur.execute(sql_str)
    l_src_trg_recs = tw_db_cur.fetchall()
    l_ready_recs = []
    cnt = 0
    ready_cnt = 0
    for src_trg_rec in l_src_trg_recs:
        cnt += 1
        if cnt % 50000 == 0 and cnt >= 50000:
            logging.debug('[udt_tw_src_trg_data_with_sna_com_id] %s src_trg recs scanned with %s ready recs in %s secs.'
                          % (cnt, ready_cnt, time.time() - timer_start))
        trg_tw_id = src_trg_rec[0]
        trg_usr_id = src_trg_rec[1]
        src_usr_id = src_trg_rec[2]
        trg_usr_sna_com_id = None
        if trg_usr_id in d_sna:
            trg_usr_sna_com_id = d_sna[trg_usr_id]
        src_usr_sna_com_id = None
        if src_usr_id in d_sna:
            src_usr_sna_com_id = d_sna[src_usr_id]
        if trg_usr_sna_com_id is None and src_usr_sna_com_id is None:
            continue
        l_ready_recs.append((trg_tw_id, trg_usr_id, trg_usr_sna_com_id, src_usr_id, src_usr_sna_com_id))
        ready_cnt += 1
    logging.debug('[udt_tw_src_trg_data_with_sna_com_id] All %s src_trg recs scanned with %s ready recs in %s secs.'
                  % (cnt, ready_cnt, time.time() - timer_start))

    df_out = pd.DataFrame(l_ready_recs, columns=['trg_tw_id', 'trg_usr_id', 'trg_usr_sna_com_id',
                                                 'src_usr_id', 'src_usr_sna_com_id'])
    df_out.to_pickle(global_settings.g_udt_tw_src_trg_sna_com_id_file)

    tw_db_cur.close()
    tw_db_conn.close()
    logging.debug('[udt_tw_src_trg_data_with_sna_com_id] All done in %s secs.' % str(time.time() - timer_start))


def udt_tw_src_trg_sna_com_id_to_db():
    logging.debug('[udt_tw_src_trg_sna_com_id_to_db] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    sql_str = """create table if not exists cp4.mf3jh_udt_tw_src_trg_sna_com_id (
                trg_tw_id char(22) primary key, 
                trg_usr_id char(22), 
                trg_usr_sna_com_id integer, 
                src_usr_id char(22), 
                src_usr_sna_com_id integer)"""
    tw_db_cur.execute(sql_str)
    tw_db_conn.commit()
    logging.debug('[udt_tw_src_trg_sna_com_id_to_db] create cp4.mf3jh_udt_tw_src_trg_sna_com_id')


    sql_str = """insert into cp4.mf3jh_udt_tw_src_trg_sna_com_id (trg_tw_id, trg_usr_id, trg_usr_sna_com_id, 
                src_usr_id, src_usr_sna_com_id) values (%s,%s,%s,%s,%s) on conflict do nothing"""

    df_src_trg_sna = pd.read_pickle(global_settings.g_udt_tw_src_trg_sna_com_id_file)
    cnt = 0
    for idx, rec in df_src_trg_sna.iterrows():
        trg_tw_id = rec[0]
        trg_usr_id = rec[1]
        trg_usr_sna_com_id = int(rec[2]) if rec[2] is not None else None
        src_usr_id = rec[3]
        src_usr_sna_com_id = int(rec[4]) if rec[4] is not None else None
        tw_db_cur.execute(sql_str, (trg_tw_id, trg_usr_id, trg_usr_sna_com_id, src_usr_id, src_usr_sna_com_id))
        cnt += 1
        if cnt % 100000 == 0 and cnt >= 100000:
            tw_db_conn.commit()
            logging.debug('[udt_tw_src_trg_sna_com_id_to_db] %s recs committed in %s secs.'
                          % (cnt, time.time() - timer_start))
    tw_db_conn.commit()
    logging.debug('[udt_tw_src_trg_sna_com_id_to_db] All %s recs committed in %s secs.'
                  % (cnt, time.time() - timer_start))

    tw_db_cur.close()
    tw_db_conn.close()
    logging.debug('[udt_tw_src_trg_sna_com_id_to_db] All done.')


def udt_tw_src_trg_sna_data(trg_tw_start_dt, trg_tw_end_dt, sna_file):
    logging.debug('[udt_tw_src_trg_sna_data] Starts...')
    timer_start = time.time()

    df_udt_tw_src_trg = pd.read_pickle(global_settings.g_udt_tw_src_trg_data_file)
    df_udt_tw_src_trg = df_udt_tw_src_trg.set_index('trg_tw_id')
    logging.debug('[udt_tw_src_trg_sna_data] Load df_udt_tw_src_trg with %s recs.' % str(len(df_udt_tw_src_trg)))

    d_sna = dict()
    if sna_file is not None:
        with open(sna_file, 'r') as in_fd:
            csv_reader = csv.reader(in_fd, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    usr_id = row[0]
                    com_id = row[2]
                    try:
                        d_sna[usr_id] = str(int(com_id))
                    except:
                        d_sna[usr_id] = None
            in_fd.close()
        logging.debug('[udt_tw_src_trg_sna_data] Load d_sna from sna_file with %s recs.' % str(len(d_sna)))
    else:
        tw_db_conn = psycopg2.connect(host='postgis1',
                                      port=5432,
                                      dbname='socialsim',
                                      user=global_settings.g_postgis1_username,
                                      password=global_settings.g_postgis1_password)
        tw_db_cur = tw_db_conn.cursor()
        sql_str = """select id, community from cp4_pipeline.updated_sna"""
        tw_db_cur.execute(sql_str)
        l_sna_recs = tw_db_cur.fetchall()
        for sna_rec in l_sna_recs:
            usr_id = sna_rec[0]
            com_id = sna_rec[1]
            if com_id is not None:
                try:
                    d_sna[usr_id] = str(int(com_id))
                except:
                    d_sna[usr_id] = None
        tw_db_cur.close()
        tw_db_conn.close()
        logging.debug('[udt_tw_src_trg_sna_data] Load d_sna from db with %s recs.' % str(len(d_sna)))

    l_ready_recs = []
    cnt = 0
    for trg_tw_id, src_trg_rec in df_udt_tw_src_trg.iterrows():
        trg_usr_id = src_trg_rec['trg_usr_id']
        trg_tw_type = src_trg_rec['trg_tw_type']
        trg_tw_datetime = src_trg_rec['trg_tw_datetime']
        trg_l_nars = src_trg_rec['trg_l_nars']
        trg_l_stances = src_trg_rec['trg_l_stances']
        trg_l_sentiments = src_trg_rec['trg_l_sentiments']
        src_tw_id = src_trg_rec['src_tw_id']
        src_usr_id = src_trg_rec['src_usr_id']
        src_tw_type = src_trg_rec['src_tw_type']
        src_tw_datetime = src_trg_rec['src_tw_datetime']
        src_l_nars = src_trg_rec['src_l_nars']
        src_l_stances = src_trg_rec['src_l_stances']
        src_l_sentiments = src_trg_rec['src_l_sentiments']

        if trg_tw_datetime < trg_tw_start_dt or trg_tw_datetime > trg_tw_end_dt:
            continue

        trg_usr_sna_com_id = None
        src_usr_sna_com_id = None
        if trg_usr_id in d_sna:
            trg_usr_sna_com_id = d_sna[trg_usr_id]
        if src_usr_id in d_sna:
            src_usr_sna_com_id = d_sna[src_usr_id]

        ready_rec = (trg_tw_id, trg_usr_id, trg_usr_sna_com_id, trg_tw_type, trg_tw_datetime, trg_l_nars,
                     trg_l_stances, trg_l_sentiments, src_tw_id, src_usr_id, src_usr_sna_com_id, src_tw_type,
                     src_tw_datetime, src_l_nars, src_l_stances, src_l_sentiments)
        l_ready_recs.append(ready_rec)
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[udt_tw_src_trg_sna_data] %s ready_recs in %s secs.' % (cnt, time.time() - timer_start))
    logging.debug('[udt_tw_src_trg_sna_data] %s ready_recs in %s secs.' % (cnt, time.time() - timer_start))

    df_out = pd.DataFrame(l_ready_recs, columns=['trg_tw_id', 'trg_usr_id', 'trg_usr_sna_com_id', 'trg_tw_type',
                                                 'trg_tw_datetime', 'trg_l_nars', 'trg_l_stances', 'trg_l_sentiments',
                                                 'src_tw_id', 'src_usr_id', 'src_usr_sna_com_id', 'src_tw_type',
                                                 'src_tw_datetime', 'src_l_nars', 'src_l_stances', 'src_l_sentiments'])
    df_out.to_pickle(global_settings.g_udt_tw_src_trg_sna_data_file_format.format(trg_tw_start_dt, trg_tw_end_dt))
    logging.debug('[udt_tw_src_trg_sna_data] All done with %s recs in %s secs.'
                  % (len(df_out), time.time() - timer_start))



############################################################
#   TEST
############################################################
def get_direct_retweet_ratio():
    logging.debug('[get_direct_retweet_ratio] Starts...')
    l_t_files = ['cp4.venezuela.twitter.anon.retweet-reconstruction.2018-dec.json',
                 'cp4.venezuela.twitter.anon.retweet-reconstruction.2019-jan.json']
    cnt_total = 0
    cnt_direct = 0
    for t_file in l_t_files:
        with open(global_settings.g_tw_raw_data_folder + t_file, 'r') as in_fd:
            for ln in in_fd:
                cnt_total += 1
                ln_json = json.loads(ln.strip())
                if ln_json['source_tweet_id_h'] == ln_json['retweeted_from_tweet_id_h']:
                    cnt_direct += 1
            in_fd.close()
    logging.debug('[get_direct_retweet_ratio] cnt_total = %s, cnt_direct = %s' % (cnt_total, cnt_direct))



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'extract_from_raw':
        '''
        Step #1: Extract considered data from raw
            Considered fields: tw_id, usr_id, tw_type, tw_src_id, tw_datetime, raw_txt, nars
            Only English or translated texts are considered. Except tw_src_id, all other fields cannot be None or empty.
            If nars is empty, we don't consider that tw. All restuls are stored in cp4.mf3jh_ven_tw_en_all.
        '''
        ds_name = '0404'
        # num_threads = sys.argv[2]
        # job_id = sys.argv[3]
        # l_sel_ds = None
        # tw_raw_data_parse_wrapper(num_threads, job_id, l_sel_ds)
        tw_raw_data_int_to_db(ds_name)

        '''
        Step #2: Manually create indices for cp4.mf3jh_ven_tw_en_all
            It's just convenient to do this manually. The indices only need to be created once. Then the DBM will 
            update it when the table changes.
        '''
    elif cmd == 'update_tref':
        '''
        Step #3: Extract tw_ids with effective texts
            As many tws are retweets, we don't repeatedly process their texts. Instead, we process their source texts,
            and ref the retweets to their sources if any.
        '''
        l_tw_ids = []
        with open(global_settings.g_tw_raw_data_int_tw_ids_file_format.format(ds_name), 'r') as in_fd:
            for ln in in_fd:
                l_tw_ids.append(ln.strip())
            in_fd.close()
        update_tref(l_tw_ids)

    elif cmd == 'udt':
        num_threads = sys.argv[2]
        job_id = sys.argv[3]
        dt_start = None
        dt_end = None
        l_skips = [
            'collection1_2019-02-08_2019-02-14_twitter_raw_data.json',
            'collection1_2019-02-01_2019-02-07_twitter_raw_data.json',
            'collection1_2019-02-15_2019-02-21_twitter_raw_data.json',
            'cp4.ven.ci2.twitter.v2.2019-02-15_2019-02-21.json',
            'cp4.ven.ci2.twitter.v2.2019-02-22_2019-02-28.json',
            'cp4.ven.ci2.twitter.v2.2019-03-01_2019-03-07.json',
            'cp4.ven.ci2.twitter.v2.2019-03-08_2019-03-14.json',
            'cp4.ven.ci2.twitter.v2.2019-03-15_2019-03-21.json']
        # udt_tw_raw_data_parse_wrapper(num_threads, job_id, dt_start, dt_end, l_skips=l_skips)
        # build_udt_tw_data()
        # build_udt_usr_data_multithread(num_threads, job_id)
        # udt_usr_data_int_to_out()
        # com_id = '151'
        # d_usr_com_info = get_usr_com_info(com_id)
        # build_udt_for_community(d_usr_com_info, com_id)
        # udt_to_db(com_id)
        # build_udt_tw_src_trg_data_multithread(num_threads, job_id)
        # udt_tw_src_trg_data_int_to_out()
        # fill_mdid_com_id_to_udt_tw_srg_trg_data()
        # udt_tw_srg_trg_data_to_db()
        # udt_tw_src_trg_data_with_sna_com_id()
        # udt_tw_src_trg_sna_com_id_to_db()
        trg_tw_start_dt = '20190322000000'
        trg_tw_end_dt = '20190404235959'
        # sna_file = global_settings.g_tw_work_folder + 'Moody_sna_run_0723.csv'
        sna_file = None
        udt_tw_src_trg_sna_data(trg_tw_start_dt, trg_tw_end_dt, sna_file)

    elif cmd == 'test':
        get_direct_retweet_ratio()