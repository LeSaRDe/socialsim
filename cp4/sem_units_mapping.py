import json
import logging
import sqlite3
from os import path, walk
import networkx as nx
import numpy as np
from semantic_unit_extraction import NarrativeAgent


g_work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
g_tw_db_path = g_work_folder + 'ven_tw_en_v2-1.db'
g_top_r_and_q_users = g_work_folder + 'ven_tw_top_r_and_q_users_v2-1.txt'
g_tw_r_and_q_map_dataset_folder = g_work_folder + 'ven_tw_r_and_q/'
g_tw_r_and_q_map_dataset_format = g_tw_r_and_q_map_dataset_folder + 'ven_tw_r_and_q_map_dataset_v2-1_{0}.txt'
g_tw_r_and_q_map_tw_id_dataset_format = g_tw_r_and_q_map_dataset_folder + 'ven_tw_r_and_q_map_tw_id_dataset_v2-1_{0}.txt'
g_sem_units_folder = g_work_folder + 'sem_units_full/'
g_cls_graph_format = g_sem_units_folder + '{0}_cls_graph.gml'
g_nps_format = g_sem_units_folder + '{0}_nps.txt'
g_voc_to_idx_format = g_tw_r_and_q_map_dataset_folder + '{0}_v2i.json'
g_idx_to_voc_format = g_tw_r_and_q_map_dataset_folder + '{0}_i2v.json'
g_phrase_maps_format = g_tw_r_and_q_map_dataset_folder + '{0}_phrase_maps.txt'
g_pulse_maps_format = g_tw_r_and_q_map_dataset_folder + '{0}_pulse_maps.txt'

g_d_v2i_per_usr = dict()
g_d_i2v_per_usr = dict()


def make_sem_units_task_list(nar_agent):
    s_tw_ids = set([])
    for (dirpath, dirname, filenames) in walk(g_tw_r_and_q_map_dataset_folder):
        for filename in filenames:
            if filename[:38] == 'ven_tw_r_and_q_map_tw_id_dataset_v2-1_' and filename[-4:] == '.txt':
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for ln in in_fd:
                        fields = [item.strip() for item in ln.split('|')]
                        s_tw_ids.add(fields[0])
                        s_tw_ids.add(fields[1])
                    in_fd.close()

    l_sem_units_tasks = []
    tw_db_conn = sqlite3.connect(g_tw_db_path)
    tw_db_cur = tw_db_conn.cursor()
    tw_sql_str = '''select raw_txt from ven_tw_en where tw_id = ?'''
    for tw_id in s_tw_ids:
        tw_db_cur.execute(tw_sql_str, (tw_id,))
        rec = tw_db_cur.fetchone()
        if rec is None:
            continue
        raw_txt = rec[0]
        clean_txt = nar_agent.text_clean(raw_txt)
        if clean_txt is None:
            continue
        l_sem_units_tasks.append((tw_id, clean_txt))
    tw_db_conn.close()
    logging.debug('%s semantic units tasks.' % len(l_sem_units_tasks))
    return l_sem_units_tasks


def extract_sem_units(l_sem_units_tasks, nar_agent):
    nar_agent.task_multithreads(nar_agent.sem_unit_extraction_thread,
                                l_sem_units_tasks,
                                10,
                                output_folder=g_sem_units_folder)
    logging.debug('Semantic units tasks are all done!')


def extract_replies_and_quotes_for_one_user(usr_id):
    tw_db_conn = sqlite3.connect(g_tw_db_path)
    tw_db_cur = tw_db_conn.cursor()
    tw_sql_str = '''select tw_id, raw_txt, tw_src_id, tw_datetime from ven_tw_en where usr_id = ? and (tw_type = 'r' or tw_type = 'q') order by tw_datetime'''
    tw_src_sql_str = '''select raw_txt from ven_tw_en where tw_id = ?'''

    tw_db_cur.execute(tw_sql_str, (usr_id,))
    l_recs = tw_db_cur.fetchall()
    dataset = []
    tw_id_dataset = []
    cnt = 0
    for rec in l_recs:
        tw_id = rec[0]
        raw_txt = rec[1]
        if raw_txt is None or raw_txt == '':
            continue
        tw_src_id = rec[2]
        if tw_src_id is None:
            continue
        tw_datetime = rec[3]
        tw_db_cur.execute(tw_src_sql_str, (tw_src_id,))
        rec = tw_db_cur.fetchone()
        if rec is None:
            continue
        tw_src_raw_txt = rec[0]
        if tw_src_raw_txt is None or tw_src_raw_txt == '':
            continue
        data_rec = tw_src_raw_txt + '|' + raw_txt + '|' + tw_datetime
        dataset.append(data_rec)
        tw_id_data_rec = tw_src_id + '|' + tw_id + '|' + tw_datetime
        tw_id_dataset.append(tw_id_data_rec)
        cnt += 1
    tw_db_conn.close()
    logging.debug('%s mappings for user %s.' % (cnt, usr_id))

    with open(g_tw_r_and_q_map_dataset_format.format(usr_id), 'w+') as out_fd:
        out_str = '\n'.join(dataset)
        out_fd.write(out_str)
        out_fd.close()
    with open(g_tw_r_and_q_map_tw_id_dataset_format.format(usr_id), 'w+') as out_fd:
        out_str = '\n'.join(tw_id_dataset)
        out_fd.write(out_str)
        out_fd.close()


def extract_replies_and_quotes_for_batch(l_top_usrs):
    # l_top_usrs = get_r_and_q_top_users(top_num, rq_cnt)
    for usr_id in l_top_usrs:
        extract_replies_and_quotes_for_one_user(usr_id)


def get_r_and_q_top_users(top_num, rq_cnt=None):
    l_top_usrs = []
    if top_num is not None:
        term_cond = 'top'
    elif rq_cnt is not None:
        term_cond = 'rq'
    else:
        raise Exception('Invalid terminal condition!')
    usr_cnt = 0
    with open(g_top_r_and_q_users, 'r') as in_fd:
        for ln in in_fd:
            fields = [item.strip() for item in ln.split('|')]
            if term_cond == 'top':
                if usr_cnt >= top_num:
                    break
                else:
                    l_top_usrs.append(fields[0])
                    usr_cnt += 1
            elif term_cond == 'rq':
                if fields[1] < rq_cnt:
                    break
                else:
                    l_top_usrs.append(fields[0])
                    usr_cnt += 1
        in_fd.close()
    return l_top_usrs


def retrieve_sem_units_for_one_user(usr_id):
    l_sem_units = []
    with open(g_tw_r_and_q_map_tw_id_dataset_format.format(usr_id), 'r') as in_fd:
        for ln in in_fd:
            fields = [item.strip() for item in ln.split('|')]
            src_tw_id = fields[0]
            trg_tw_id = fields[1]
            trg_tw_datetime = fields[2]
            if (not path.exists(g_cls_graph_format.format(src_tw_id))
                    and not path.exists(g_nps_format.format(src_tw_id))) \
                or (not path.exists(g_cls_graph_format.format(trg_tw_id))
                    and not path.exists(g_nps_format.format(trg_tw_id))):
                continue
            src_cls = None
            if path.exists(g_cls_graph_format.format(src_tw_id)):
                src_cls = nx.read_gml(g_cls_graph_format.format(src_tw_id))
            src_nps = None
            if path.exists(g_nps_format.format(src_tw_id)):
                src_nps = []
                with open(g_nps_format.format(src_tw_id), 'r') as in_fd:
                    for ln in in_fd:
                        src_nps.append(ln.strip())
                    in_fd.close()
            trg_cls = None
            if path.exists(g_cls_graph_format.format(trg_tw_id)):
                trg_cls = nx.read_gml(g_cls_graph_format.format(trg_tw_id))
            trg_nps = None
            if path.exists(g_nps_format.format(trg_tw_id)):
                trg_nps = []
                with open(g_nps_format.format(trg_tw_id), 'r') as in_fd:
                    for ln in in_fd:
                        trg_nps.append(ln.strip())
                    in_fd.close()

            sem_units_rec = dict()
            sem_units_rec['src_cls'] = src_cls
            sem_units_rec['src_nps'] = src_nps
            sem_units_rec['trg_cls'] = trg_cls
            sem_units_rec['trg_nps'] = trg_nps
            sem_units_rec['datetime'] = trg_tw_datetime
            l_sem_units.append(sem_units_rec)
    logging.debug('For user %s: %s sem units recs.' % (usr_id, len(l_sem_units)))
    return l_sem_units


def retrieve_sem_units_for_batch(l_usr_ids):
    d_usr_sem_units = dict()
    for usr_id in l_usr_ids:
        l_sem_units = retrieve_sem_units_for_one_user(usr_id)
        d_usr_sem_units[usr_id] = l_sem_units
    return d_usr_sem_units


def extract_phrase_map_and_vocab_for_one_user(usr_id, l_sem_units, en_vocab_out, en_phrase_maps_out):
    l_raw_phrase_maps = []
    s_tokens = set([])
    for sem_units in l_sem_units:
        '''Extract phrases on the src side and the trg side, and extract vocab.'''
        s_src_phrases = set([])
        s_trg_phrases = set([])
        src_cls = sem_units['src_cls']
        if src_cls is not None:
            for node in src_cls.nodes(data=True):
                node_txt = node[1]['txt'].strip()
                l_node_tokens = [token.strip().lower() for token in node_txt.split(' ') if token.strip() != '']
                s_src_phrases.add(' '.join(l_node_tokens))
                for node_token in l_node_tokens:
                    s_tokens.add(node_token)
        l_src_nps = sem_units['src_nps']
        if l_src_nps is not None:
            for src_np in l_src_nps:
                l_np_tokens = [token.strip().lower() for token in src_np.split(' ') if token.strip() != '']
                s_src_phrases.add(' '.join(l_np_tokens))
                for np_token in l_np_tokens:
                    s_tokens.add(np_token)
        trg_cls = sem_units['trg_cls']
        if trg_cls is not None:
            for node in trg_cls.nodes(data=True):
                node_txt = node[1]['txt'].strip()
                l_node_tokens = [token.strip().lower() for token in node_txt.split(' ') if token.strip() != '']
                s_trg_phrases.add(' '.join(l_node_tokens))
                for node_token in l_node_tokens:
                    s_tokens.add(node_token.lower())
        l_trg_nps = sem_units['trg_nps']
        if l_trg_nps is not None:
            for trg_np in l_trg_nps:
                l_np_tokens = [token.strip().lower() for token in trg_np.split(' ') if token.strip() != '']
                s_trg_phrases.add(' '.join(l_np_tokens))
                for np_token in l_np_tokens:
                    s_tokens.add(np_token.lower())

        '''Make raw phrase mappings'''
        trg_datetime = sem_units['datetime']
        for src_p in s_src_phrases:
            for trg_p in s_trg_phrases:
                map_rec = src_p + '|' + trg_p + '|' + trg_datetime
                l_raw_phrase_maps.append(map_rec)

    logging.debug('For user %s: %s tokens in vocab.' % (usr_id, len(s_tokens)))
    logging.debug('For usr %s: %s raw phrase mappings.' % (usr_id, len(l_raw_phrase_maps)))

    if en_vocab_out:
        make_vocab_index(s_tokens, usr_id)
        logging.debug('Vocab indices for %s have been output.' % usr_id)

    if en_phrase_maps_out:
        with open(g_phrase_maps_format.format(usr_id), 'w+') as out_fd:
            out_str = '\n'.join(l_raw_phrase_maps)
            out_fd.write(out_str)
            out_fd.close()
        logging.debug('Phrase maps for %s have been output.' % usr_id)

    return s_tokens, l_raw_phrase_maps


def extract_phrase_map_and_vocab_for_batch(l_usr_ids, d_usr_sem_units, en_vocab_out=True, en_phrase_maps_out=True):
    if l_usr_ids is None or len(l_usr_ids) <= 0 or d_usr_sem_units is None or len(d_usr_sem_units) <= 0:
        raise Exception('Invalid inputs to extract_phrase_map_and_vocab_for_batch!')
    d_out = dict()
    for usr_id in l_usr_ids:
        if usr_id not in d_usr_sem_units:
            raise Exception('%s is not in d_usr_sem_units.' % usr_id)
        s_tokens, l_raw_phrase_maps = extract_phrase_map_and_vocab_for_one_user(usr_id,
                                                                                d_usr_sem_units[usr_id],
                                                                                en_vocab_out,
                                                                                en_phrase_maps_out)
        d_out[usr_id] = {'vocab': s_tokens, 'phrase_maps': l_raw_phrase_maps}
    logging.debug('extract_phrase_map_and_vocab_for_batch is done with %s users.' % len(l_usr_ids))
    return d_out


def convert_phrase_maps_to_pulses_for_one_usr(usr_id, l_raw_phrase_maps, en_pulses_out):
    l_pulses = []
    for phrase_map in l_raw_phrase_maps:
        fields = [item.strip() for item in phrase_map.split('|')]
        src_p = fields[0]
        trg_p = fields[1]
        trg_datetime = fields[2]
        l_src_tokens = [token.strip() for token in src_p.split(' ')]
        l_trg_tokens = [token.strip() for token in trg_p.split(' ')]
        # src_bit_vec = np.zeros(voc_size)
        # trg_bit_vec = np.zeros(voc_size)
        src_pulses = []
        trg_pulses = []
        for src_token in l_src_tokens:
            idx = v2i_token(usr_id, src_token)
            # src_bit_vec[idx] = 1.0
            src_pulses.append(idx)
        for trg_token in l_trg_tokens:
            idx = v2i_token(usr_id, trg_token)
            # trg_bit_vec[idx] = 1.0
            trg_pulses.append(idx)
        # l_bit_maps.append((src_bit_vec, trg_bit_vec, src_pulses, trg_pulses, trg_datetime))
        l_pulses.append((src_pulses, trg_pulses, trg_datetime))

    if en_pulses_out:
        l_pulses_out = []
        for src_pulses, trg_pulses, trg_datetime in l_pulses:
            src_pulses_str = ','.join([str(ele) for ele in src_pulses])
            trg_pulses_str = ','.join([str(ele) for ele in trg_pulses])
            out_str = src_pulses_str + '|' + trg_pulses_str + '|' + trg_datetime
            l_pulses_out.append(out_str)
        with open(g_pulse_maps_format.format(usr_id), 'w+') as out_fd:
            out_fd.write('\n'.join(l_pulses_out))
            out_fd.close()

    return l_pulses


def convert_phrase_maps_to_pulses_for_batch(l_usr_ids, d_usr_vocab_phrase_maps, en_pulses_out=True):
    if l_usr_ids is None or len(l_usr_ids) == 0 or d_usr_vocab_phrase_maps is None or len(d_usr_vocab_phrase_maps) == 0:
        raise Exception('Invalid inputs to convert_phrase_maps_to_pulses_for_batch!')
    d_out = dict()
    for usr_id in l_usr_ids:
        if usr_id not in d_usr_vocab_phrase_maps:
            raise Exception('%s is not in d_usr_vocab_phrase_maps.' % usr_id)
        l_pulses = convert_phrase_maps_to_pulses_for_one_usr(usr_id,
                                                             d_usr_vocab_phrase_maps[usr_id]['phrase_maps'],
                                                             en_pulses_out)
        d_out[usr_id] = l_pulses
    logging.debug('convert_phrase_maps_to_pulses_for_batch is done with %s users.' % len(l_usr_ids))
    return d_out


def make_vocab_index(s_tokens, voc_name):
    d_v2i = dict()
    d_i2v = dict()
    for idx, token in enumerate(s_tokens):
        d_v2i[token] = idx
        d_i2v[idx] = token
    with open(g_voc_to_idx_format.format(voc_name), 'w+') as out_fd:
        json.dump(d_v2i, out_fd)
        out_fd.close()
    with open(g_idx_to_voc_format.format(voc_name), 'w+') as out_fd:
        json.dump(d_i2v, out_fd)
        out_fd.close()


def load_vocab_index_for_one(voc_name):
    global g_d_i2v_per_usr, g_d_v2i_per_usr
    with open(g_voc_to_idx_format.format(voc_name), 'r') as in_fd:
        g_d_v2i_per_usr[voc_name] = json.load(in_fd)
        in_fd.close()
    with open(g_idx_to_voc_format.format(voc_name), 'r') as in_fd:
        g_d_i2v_per_usr[voc_name] = json.load(in_fd)
        in_fd.close()
    if len(g_d_i2v_per_usr) != len(g_d_v2i_per_usr):
        raise Exception('g_d_i2v does not match g_d_v2i on %s!' % voc_name)
    logging.debug('v2i and i2v for %s have been loaded.' % voc_name)
    return len(g_d_i2v_per_usr)


def load_vocab_index_for_batch(l_voc_names):
    for voc_name in l_voc_names:
        load_vocab_index_for_one(voc_name)


def v2i_token(usr_id, token):
    if g_d_v2i_per_usr is None or usr_id not in g_d_v2i_per_usr or g_d_v2i_per_usr[usr_id] is None:
        raise Exception('No g_d_v2i_per_usr for %s!' % usr_id)
    if token not in g_d_v2i_per_usr[usr_id]:
        raise Exception('%s is not in g_d_v2i!')
    return g_d_v2i_per_usr[usr_id][token]


def i2v_token(usr_id, idx):
    if g_d_i2v_per_usr is None or usr_id not in g_d_i2v_per_usr or g_d_i2v_per_usr[usr_id] is None:
        raise Exception('No g_d_i2v!')
    if idx not in g_d_i2v_per_usr[usr_id]:
        raise Exception('%s is not in g_d_i2v' % idx)
    return g_d_i2v_per_usr[usr_id][idx]


def convert_pulses_to_bitvec_for_one_user(usr_id, pulses_str):
    if pulses_str is None or pulses_str == '':
        raise Exception('Invalid pulses_str!')
    if g_d_v2i_per_usr is None or usr_id not in g_d_v2i_per_usr or g_d_v2i_per_usr[usr_id] is None:
        raise Exception('No vocab available for %s.' % usr_id)
    bitvec_size = len(g_d_v2i_per_usr[usr_id])
    bitvec = np.zeros(bitvec_size)
    l_pulses = [int(ele.strip()) for ele in pulses_str.split(',')]
    for pulse in l_pulses:
        bitvec[pulse] = 1
    return bitvec


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    '''Get top replies_n_quotes users'''
    l_top_usrs = get_r_and_q_top_users(10)

    # '''Retrieve replies and quotes for each user'''
    # extract_replies_and_quotes_for_batch(l_top_usrs)
    # '''Extract semantic units for replies and quotes'''
    # nar_agent = NarrativeAgent('agent_config.conf')
    # l_su_tasks = make_sem_units_task_list(nar_agent)
    # extract_sem_units(l_su_tasks, nar_agent)

    '''Retrieve semantic units of replies and quotes for each user'''
    d_usr_sem_units = retrieve_sem_units_for_batch(l_top_usrs)
    '''Build vocab and phrase mappings based on semantic units for each user'''
    d_usr_vocab_phrase_maps = extract_phrase_map_and_vocab_for_batch(l_top_usrs, d_usr_sem_units)
    '''Load vocab for each user'''
    load_vocab_index_for_batch(l_top_usrs)
    '''Convert phrase mappings to bit mappings for each user'''
    convert_phrase_maps_to_pulses_for_batch(l_top_usrs, d_usr_vocab_phrase_maps)
