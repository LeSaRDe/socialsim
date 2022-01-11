import logging
import json
import time
import math
import threading
import multiprocessing
import sys

import numpy as np
import psycopg2
import pandas as pd

import global_settings


def get_nar_codes_from_activated_users(tb_name):
    logging.debug('[get_nar_codes_from_activated_users] Starts...')
    db_conn = psycopg2.connect(host='postgis1',
                               port=5432,
                               dbname='socialsim',
                               user=global_settings.g_postgis1_username,
                               password=global_settings.g_postgis1_password)
    db_cur = db_conn.cursor()
    sql_str = """select narratives from {0}""".format(tb_name)
    db_cur.execute(sql_str)
    l_recs = db_cur.fetchall()

    l_nars = []
    for rec in l_recs:
        narratives = rec[0]
        l_nars += narratives
    l_nars = list(set(l_nars))
    logging.debug('[get_nar_codes_from_activated_users] %s nars.' % str(len(l_nars)))

    d_nar = {l_nars[i]: i for i in range(len(l_nars))}
    d_nar_rev = {d_nar[key]: key for key in d_nar}
    with open(global_settings.g_tw_narrative_to_code_file, 'w+') as out_fd:
        json.dump(d_nar, out_fd)
        out_fd.close()
    with open(global_settings.g_tw_code_to_narrative_file, 'w+') as out_fd:
        json.dump(d_nar_rev, out_fd)
        out_fd.close()
    logging.debug('[get_nar_codes_from_activated_users] d_nar is output.')


def get_nar_codes():
    '''
    Don't forget to set up your postgis1 account!
    '''
    db_conn = psycopg2.connect(host='postgis1',
                               port=5432,
                               dbname='socialsim',
                               user=global_settings.g_postgis1_username,
                               password=global_settings.g_postgis1_password)
    db_cur = db_conn.cursor()

    sql_str = """select distinct narrative from cp4.tw_supervisednarratives"""
    db_cur.execute(sql_str)
    l_recs = db_cur.fetchall()
    l_nars = [rec[0] for rec in l_recs]
    d_nar = {l_nars[i]: i for i in range(len(l_nars))}
    d_nar_rev = {d_nar[key]: key for key in d_nar}
    logging.debug('[get_nar_codes] d_nar = %s' % d_nar)
    logging.debug('[get_nar_codes] d_nar_rev = %s' % d_nar_rev)
    db_cur.close()
    db_conn.close()

    with open(global_settings.g_tw_narrative_to_code_file, 'w+') as out_fd:
        json.dump(d_nar, out_fd)
        out_fd.close()
    with open(global_settings.g_tw_code_to_narrative_file, 'w+') as out_fd:
        json.dump(d_nar_rev, out_fd)
        out_fd.close()
    logging.debug('[get_nar_codes] d_nar is output.')


def nar_array_to_nar_bvec(d_nar_to_code, l_nar_strs):
    if len(l_nar_strs) <= 0:
        return np.zeros(len(d_nar_to_code))
    try:
        l_nar_pulses = [d_nar_to_code[nar] for nar in l_nar_strs]
    except:
        return None
    nar_bvec = np.asarray([1 if i in l_nar_pulses else 0 for i in range(len(d_nar_to_code))])
    return nar_bvec


# @profile
def make_full_train_set_single_thread(tb_name, set_name, t_id, en_usr_avg_embed=False, d_nar_to_code=None,
                                      d_usr_avg_embeds=None, offset=0, task_size=0):
    '''
    Include usr_id, usr_role, com_id, nar_vec, usr_avg_embed. All sub train sets should be derived based on this set.
    '''
    timer_start = time.time()

    with open(global_settings.g_tw_narrative_to_code_file, 'r') as in_fd:
        d_nar_to_code = json.load(in_fd)
        in_fd.close()
    logging.debug('[make_full_train_set_single_thread] g_tw_narrative_to_code_file is loaded.')

    if en_usr_avg_embed:
        with open(global_settings.g_tw_embed_usr_avg_embeds_output, 'r') as in_fd:
            d_usr_avg_embeds = json.load(in_fd)
            in_fd.close()
        logging.debug('[make_full_train_set_single_thread] g_tw_embed_usr_avg_embeds_output is loaded in %s secs.'
                      % str(time.time() - timer_start))

    db_conn = psycopg2.connect(host='postgis1',
                               port=5432,
                               dbname='socialsim',
                               user=global_settings.g_postgis1_username,
                               password=global_settings.g_postgis1_password)
    db_cur = db_conn.cursor()
    # sql_str = """select usr_id, role, community, narratives from cp4.{0} offset {1} fetch next {2} rows only""".format(tb_name, offset, task_size)
    sql_str = """select usr_id, role, community, narratives from {0}""".format(tb_name)
    db_cur.execute(sql_str)
    l_recs = db_cur.fetchall()
    db_cur.close()
    db_conn.close()
    l_rows = []
    cnt = 0
    logging.debug('[make_full_train_set_single_thread] Thread %s: Read %s recs from DB.' % (t_id, len(l_recs)))
    for rec in l_recs:
        usr_id = rec[0]
        usr_role = rec[1]
        com_id = rec[2]
        if com_id is None:
            continue
        l_nar_strs = rec[3]
        nar_bvec = nar_array_to_nar_bvec(d_nar_to_code, l_nar_strs)
        if nar_bvec is None:
            continue
        if en_usr_avg_embed:
            try:
                usr_avg_embed = d_usr_avg_embeds[usr_id]
            except:
                raise Exception('[make_full_train_set_for_specific_usrs] Thread %s Unknown user %s' % (t_id, usr_id))
            df_row = (usr_id, usr_role, com_id, nar_bvec, usr_avg_embed)
        else:
            df_row = (usr_id, usr_role, com_id, nar_bvec)
        l_rows.append(df_row)
        cnt += 1
        if cnt % 5000 == 0 and cnt >= 5000:
            logging.debug('[make_full_train_set_for_specific_usrs] Thread %s: %s usr recs done in %s secs.'
                          % (t_id, cnt, time.time() - timer_start))
    if en_usr_avg_embed:
        df_all = pd.DataFrame(l_rows, columns=['usr_id', 'role', 'com_id', 'nar_vec', 'sem_vec'])
    else:
        df_all = pd.DataFrame(l_rows, columns=['usr_id', 'role', 'com_id', 'nar_vec'])
    logging.debug('[make_full_train_set_for_specific_usrs] Thread %s: All %s usr recs done in %s secs.'
                  % (t_id, cnt, time.time() - timer_start))
    out_name = str(set_name)
    df_all.to_pickle(global_settings.g_tw_ae_narrative_rep_train_sets_file_format.format(out_name))
    logging.debug('[make_full_train_set_for_specific_usrs] Thread %s: Output in %s secs.'
                  % (t_id, str(time.time() - timer_start)))


'''DO NOT USE THIS FUNCTION!'''
def make_full_train_set_multithread(op_func, num_threads, tb_name, set_name):
    '''
    WARNING:
        We don't really have to use multithreading as 1) single thread can make a large training set done within
    just 100+ secs, and 2) making training sets may require large memory, and thus multithreading may lead to out of
    memory issue.
    '''
    logging.debug('[make_full_train_set_multithread] Starts...')
    timer_start = time.time()

    db_conn = psycopg2.connect(host='postgis1',
                               port=5432,
                               dbname='socialsim',
                               user=global_settings.g_postgis1_username,
                               password=global_settings.g_postgis1_password)
    db_cur = db_conn.cursor()
    sql_str = '''select count(*) from {0}'''.format(tb_name)
    db_cur.execute(sql_str)
    rec = db_cur.fetchone()
    num_tasks = int(rec[0])
    db_cur.close()
    db_conn.close()

    l_tasks = []
    offset = 0
    task_size = math.ceil(num_tasks / num_threads)
    while offset + task_size < num_tasks:
        l_tasks.append((offset, task_size))
        offset += task_size

    with open(global_settings.g_tw_narrative_to_code_file, 'r') as in_fd:
        d_nar_to_code = json.load(in_fd)
        in_fd.close()
    logging.debug('[make_full_train_set_multithread] g_tw_narrative_to_code_file is loaded.')

    with open(global_settings.g_tw_embed_usr_avg_embeds_activated_usrs_only_output, 'r') as in_fd:
        d_usr_avg_embeds = json.load(in_fd)
        in_fd.close()
    logging.debug('[make_full_train_set_multithread] g_tw_embed_usr_avg_embeds_activated_usrs_only_output is loaded in %s secs.'
                  % str(time.time() - timer_start))

    l_threads = []
    t_id = 0
    for task in l_tasks:
        t = threading.Thread(target=op_func,
                             args=(tb_name,
                                   set_name,
                                   str(t_id),
                                   d_nar_to_code,
                                   d_usr_avg_embeds,
                                   task[0],
                                   task[1]))
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
                logging.debug('[make_full_train_set_multithread] Thread %s is finished.' % t.getName())

    logging.debug('[make_full_train_set_multithread] All done in %s sec for %s tasks.'
                  % (time.time() - timer_start, num_tasks))


def make_sub_train_set_by_com_id(tb_name, l_cols):
    logging.debug('[make_sub_train_set_by_com_id] Starts: tb_name = %s, l_cols = %s' % (tb_name, l_cols))
    try:
        train_set_df = pd.read_pickle(global_settings.g_tw_ae_narrative_rep_train_sets_file_format.format(tb_name))
        for i in range(len(train_set_df)):
            if 'sem_vec' in l_cols:
                train_set_df.at[i, 'sem_vec'] = np.asarray(train_set_df.at[i, 'sem_vec'], dtype=np.float32)
            if 'nar_vec' in l_cols:
                train_set_df.at[i, 'nar_vec'] = np.asarray(train_set_df.at[i, 'nar_vec'])

        l_com_ids = [ele.astype(np.int32) for ele in train_set_df['com_id'].unique() if np.isfinite(ele)]
        logging.debug('[make_sub_train_set_by_com_id] %s com_ids to go.' % len(l_com_ids))
        for com_id in l_com_ids:
            if not np.isfinite(com_id):
                continue
            train_subset_df = train_set_df.loc[train_set_df['com_id'] == com_id]
            train_subset_df = train_subset_df[l_cols]
            train_subset_df.to_pickle(global_settings.g_tw_ae_narrative_rep_train_subset_by_com_id_file_format.
                                      format(tb_name, '#'.join(l_cols), str(com_id)))
            logging.debug('[make_sub_train_set_by_com_id] Output: com_id = %s' % com_id)
    except Exception as err:
        logging.error('[make_sub_train_set] %s' % err)
        return


def sample_read_train_set(train_set_path, l_cols):
    try:
        train_set_df = pd.read_pickle(train_set_path)
        for col in l_cols:
            print('[sample_read_train_set] Read 5 rows at %s.' % col)
            print(train_set_df[[col]])
            print('\n')
    except Exception as err:
        logging.error('[sample_read_train_set] %s' % err)
        return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    '''
    Step #1: Get nar_to_code and code_to_nar mappings
    '''
    if cmd == 'nar_code':
        tb_name = 'cp4_pipeline.activated_users'
        get_nar_codes_from_activated_users(tb_name)
        # get_nar_codes()

    elif cmd == 'train_set':
        tb_name = 'cp4_pipeline.activated_users'
        set_name = tb_name + '#run_0729'
        make_full_train_set_single_thread(tb_name, set_name, 0)

    # l_cols = ['nar_vec', 'sem_vec']
    # make_sub_train_set_by_com_id(tb_name, l_cols)
    # sample_read_train_set(global_settings.g_tw_ae_narrative_rep_train_subset_by_com_id_file_format.
    #                       format(tb_name, '#'.join(l_cols), 1), l_cols)
