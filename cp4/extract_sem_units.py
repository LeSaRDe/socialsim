'''
OBJECTIVES:
    Actual runner of the semantic units extractor.
'''

import logging
import time
import threading
import multiprocessing
import math
from os import walk
import sys

import sqlite3
import psycopg2
import pandas as pd

import global_settings
from semantic_units_extractor import SemUnitsExtractor


def tw_text_clean_single_thread(sem_units_ext, l_tasks, t_id):
    logging.debug('[tw_text_clean_single_thread] Thread %s: Starts with %s tasks...' % (t_id, len(l_tasks)))
    timer_start = time.time()

    l_recs = []
    cnt = 0
    for tw_id, raw_txt in l_tasks:
        clean_txt = sem_units_ext.text_clean(raw_txt)
        if clean_txt == '':
            clean_txt = None
        l_recs.append((tw_id, clean_txt))
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[tw_text_clean_single_thread] Thread %s: %s clean txts do`ne in %s secs.'
                          % (t_id, cnt, time.time() - timer_start))
    logging.debug('[tw_text_clean_single_thread] Thread %s: All %s clean txts done in %s secs.'
                  % (t_id, cnt, time.time() - timer_start))

    out_df = pd.DataFrame(l_recs, columns=['tw_id', 'clean_txt'])
    out_df.to_pickle(global_settings.g_tw_raw_data_clean_txt_int_file_format.format(str(t_id)))
    logging.debug('[tw_text_clean_single_thread] Thread %s: All done in %s secs.' % (t_id, time.time() - timer_start))


def tw_text_clean_single_multithread(sem_units_ext, num_threads, job_id):
    logging.debug('[tw_text_clean_single_multithread] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = '''select tw_id, raw_txt from cp4.mf3jh_ven_tw_en_all where tref = false'''
    tw_db_cur.execute(tw_db_sql_str)
    l_recs = tw_db_cur.fetchall()
    tw_db_cur.close()
    tw_db_conn.close()

    num_tasks = len(l_recs)
    logging.debug('[tw_text_clean_single_multithread] %s tws to clean. Fetching tws done in %s secs.'
                  % (num_tasks, time.time() - timer_start))

    batch_size = math.ceil(num_tasks / int(num_threads))
    l_tasks = []
    for i in range(0, num_tasks, batch_size):
        if i + batch_size < num_tasks:
            l_tasks.append(l_recs[i:i + batch_size])
        else:
            l_tasks.append(l_recs[i:])
    logging.debug('[tw_text_clean_single_multithread] %s tasks to go.' % str(len(l_tasks)))

    l_threads = []
    t_id = 0
    for l_each_batch in l_tasks:
        t = threading.Thread(target=tw_text_clean_single_thread,
                             args=(sem_units_ext, l_each_batch, str(job_id) + '_' + str(t_id)))
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
                logging.debug('[tw_text_clean_single_multithread] Thread %s is finished.' % t.getName())
    logging.debug('[tw_text_clean_single_multithread] All done in %s sec.'
                  % str(time.time() - timer_start))


def clean_txt_int_to_db():
    logging.debug('[clean_txt_int_to_db] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = """update cp4.mf3jh_ven_tw_en_all set clean_txt = %s where tw_id = %s"""

    cnt = 0
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_raw_data_int_folder):
        for filename in filenames:
            if filename[:14] != 'clean_txt_int_' or filename[-7:] != '.pickle':
                continue
            df_recs = pd.read_pickle(dirpath + filename)
            for rec in df_recs.values:
                tw_id = rec[0]
                clean_txt = rec[1]
                tw_db_cur.execute(tw_db_sql_str, (clean_txt, tw_id))
                cnt += 1
                if cnt % 50000 == 0 and cnt >= 50000:
                    tw_db_conn.commit()
                    logging.debug('[clean_txt_int_to_db] %s recs committed in %s secs.'
                                  % (cnt, time.time() - timer_start))
    tw_db_conn.commit()
    logging.debug('[clean_txt_int_to_db] All %s recs committed in %s secs.'
                  % (cnt, time.time() - timer_start))
    tw_db_cur.close()
    tw_db_conn.close()


# def make_sem_unit_tasks_for_ven_tw():
#     en_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
#     en_db_cur = en_db_conn.cursor()
#     en_select_sql_str = '''select tw_id, tw_type, clean_txt from ven_tw_en'''
#     en_query_sql_str = '''select clean_txt from ven_tw_en where tw_id = ?'''
#     en_db_cur.execute(en_select_sql_str)
#     l_tw_recs = en_db_cur.fetchall()
#     l_txt_tasks = []
#     total_cnt = 0
#     eff_cnt = 0
#     for tw_rec in l_tw_recs:
#         total_cnt += 1
#         tw_id = tw_rec[0]
#         tw_type = tw_rec[1]
#         if tw_type == 't':
#             en_db_cur.execute(en_query_sql_str, (tw_id,))
#             rec = en_db_cur.fetchone()
#             if rec is not None:
#                 continue
#         tw_clean_txt = tw_rec[2]
#         if tw_clean_txt is None or tw_clean_txt == '':
#             continue
#         l_txt_tasks.append((tw_id, tw_clean_txt))
#         eff_cnt += 1
#     en_db_conn.close()
#     logging.debug('[make_sem_unit_tasks_for_ven_tw] %s effective tw tasks out of %s tws.' % (eff_cnt, total_cnt))
#     return l_txt_tasks


# def make_sem_unit_retweet_makeup_tasks_for_ven_tw():
#     en_db_conn = sqlite3.connect(global_settings.g_tw_en_db)
#     en_db_cur = en_db_conn.cursor()
#     en_select_sql_str = '''select tw_id, tw_src_id, clean_txt from ven_tw_en where tw_type="t"'''
#     en_query_sql_str = '''select clean_txt from ven_tw_en where tw_id = ?'''
#     en_db_cur.execute(en_select_sql_str)
#     l_tw_recs = en_db_cur.fetchall()
#     l_txt_tasks = []
#     for tw_rec in l_tw_recs:
#         tw_id = tw_rec[0]
#         tw_src_id = tw_rec[1]
#         en_db_cur.execute(en_query_sql_str, (tw_src_id,))
#         rec = en_db_cur.fetchone()
#         if rec is not None:
#             continue
#         tw_clean_txt = tw_rec[2]
#         if tw_clean_txt is None or tw_clean_txt == '':
#             continue
#         l_txt_tasks.append((tw_id, tw_clean_txt))
#     en_db_conn.close()
#     logging.debug('[make_sem_unit_retweet_makeup_tasks_for_ven_tw] %s retweet makeup tasks.' % len(l_txt_tasks))
#     return l_txt_tasks


def extract_tw_sem_units_single_proc(sem_units_ext, l_tasks, proc_id):
    logging.debug('[extract_tw_sem_units_single_proc] Proc %s: Starts with %s tasks.' % (proc_id, len(l_tasks)))
    timer_start = time.time()
    sem_units_ext.sem_unit_extraction_thread(l_tasks, proc_id, None, False)
    logging.debug('[extract_tw_sem_units_single_proc] Proc %s: All done in %s secs.'
                  % (proc_id, time.time() - timer_start))


def extract_tw_sem_units_multiproc(num_procs, job_id):
    logging.debug('[extract_tw_sem_units_multiproc] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = '''select tw_id, clean_txt from cp4.mf3jh_ven_tw_en_all where clean_txt is not null'''
    tw_db_cur.execute(tw_db_sql_str)
    l_sem_units_tasks = tw_db_cur.fetchall()
    num_tasks = len(l_sem_units_tasks)
    logging.debug('[extract_tw_sem_units_multiproc] Fetch %s clean_txts in %s secs.'
                  % (num_tasks, time.time() - timer_start))

    batch_size = math.ceil(num_tasks / int(num_procs))
    l_batches = []
    for i in range(0, num_tasks, batch_size):
        if i + batch_size < num_tasks:
            l_batches.append(l_sem_units_tasks[i:i + batch_size])
        else:
            l_batches.append(l_sem_units_tasks[i:])
    logging.debug('[extract_tw_sem_units_multiproc] %s tasks to go.' % str(len(l_batches)))

    l_procs = []
    t_id = 0
    for l_each_batch in l_batches:
        sem_units_ext = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
        if sem_units_ext is None:
            raise Exception('[extract_tw_sem_units_multiproc] sem_units_ext is None!')
        t = multiprocessing.Process(target=extract_tw_sem_units_single_proc,
                                    args=(sem_units_ext, l_each_batch, str(job_id) + '_' + str(t_id)))
        t.name = 't_mul_task_' + str(t_id)
        t.start()
        l_procs.append(t)
        t_id += 1

    while len(l_procs) > 0:
        for t in l_procs:
            if t.is_alive():
                t.join(1)
            else:
                l_procs.remove(t)
                logging.debug('[extract_tw_sem_units_multiproc] Proc %s is finished.' % t.name)
    logging.debug('[extract_tw_sem_units_multiproc] All done in %s sec.'
                  % str(time.time() - timer_start))

    logging.debug('[extract_tw_sem_units_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def sem_units_int_to_db():
    logging.debug('[sem_units_int_to_db] Starts...')
    timer_start = time.time()

    tw_db_conn = psycopg2.connect(host='postgis1',
                                  port=5432,
                                  dbname='socialsim',
                                  user=global_settings.g_postgis1_username,
                                  password=global_settings.g_postgis1_password)
    tw_db_cur = tw_db_conn.cursor()
    tw_db_sql_str = """create table if not exists cp4.mf3jh_ven_tw_sem_units (tw_id char(22) primary key, 
    cls_json_str text, nps text array)"""
    tw_db_cur.execute(tw_db_sql_str)
    tw_db_conn.commit()

    tw_db_sql_str = """insert into cp4.mf3jh_ven_tw_sem_units (tw_id, cls_json_str, nps) values (%s, %s, %s) 
    on conflict do nothing"""
    cnt = 0
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_sem_units_int_folder):
        for filename in filenames:
            if filename[:14] != 'sem_units_int_' or filename[-7:] != '.pickle':
                continue
            df_recs = pd.read_pickle(dirpath + filename)
            for rec in df_recs.values:
                tw_id = rec[0]
                cls_json_str = rec[1]
                nps_str = rec[2]
                if nps_str is not None:
                    l_nps = [item.strip() for item in nps_str.split('\n')]
                else:
                    l_nps = None
                tw_db_cur.execute(tw_db_sql_str, (tw_id, cls_json_str, l_nps))
                cnt += 1
                if cnt % 50000 == 0 and cnt >= 50000:
                    tw_db_conn.commit()
                    logging.debug('[sem_units_int_to_db] %s recs committed in %s secs.'
                                  % (cnt, time.time() - timer_start))
    tw_db_conn.commit()
    logging.debug('[sem_units_int_to_db] All %s recs committed in %s secs.'
                  % (cnt, time.time() - timer_start))
    tw_db_cur.close()
    tw_db_conn.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'text_clean':
        '''Step #1: Text Clean'''
        num_threads = sys.argv[2]
        job_id = sys.argv[3]
        # sem_units_ext = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
        # if sem_units_ext is None:
        #     raise Exception('[extract_sem_units:__main__] sem_units_ext is None!')
        # tw_text_clean_single_multithread(sem_units_ext, num_threads, job_id)
        # clean_txt_int_to_db()


    elif cmd == 'extract_sem_units':
        '''Step #2: Sem Units Extraction'''
        num_threads = sys.argv[2]
        job_id = sys.argv[3]
        extract_tw_sem_units_multiproc(num_threads, job_id)
        sem_units_int_to_db()

    '''Testing'''
    # tw_db_conn = psycopg2.connect(host='postgis1',
    #                               port=5432,
    #                               dbname='socialsim',
    #                               user=global_settings.g_postgis1_username,
    #                               password=global_settings.g_postgis1_password)
    # tw_db_cur = tw_db_conn.cursor()
    # tw_id = 'ohI-olQHbMQx4OIvKKOj8g'
    # tw_sql_str = '''select clean_txt from cp4.mf3jh_ven_tw_en_all where tw_id = %s'''
    # tw_db_cur.execute(tw_sql_str, (tw_id,))
    # clean_txt = tw_db_cur.fetchone()[0]
    # clean_txt = 'i love pizza.'
    # cls_graph, l_nps = sem_units_ext.extract_sem_units_from_text(clean_txt, 'test_txt')
    # print('cls_graph = ')
    # print(cls_graph.nodes(data=True))
    # print('l_nps = ')
    # print(l_nps)
    # tw_db_cur.close()
    # tw_db_conn.close()