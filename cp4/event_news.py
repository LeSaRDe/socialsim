import logging
import time
import math
import threading
import multiprocessing
import sys
from os import walk, path
import os
import random
import traceback
from datetime import datetime, timedelta
import re
import csv

import sqlite3
import json
import networkx as nx
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.special import softmax
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import normalize
import psycopg2
from scipy.spatial.distance import jensenshannon

import global_settings
from semantic_units_extractor import SemUnitsExtractor
sys.path.insert(1, global_settings.g_lexvec_model_folder)
import model as lexvec


def event_texts_clean(out_suffix):
    logging.debug('[event_texts_clean] Starts...')
    timer_start = time.time()

    sem_units_ext = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
    logging.debug('[event_texts_clean] Create sem_units_ext')

    l_ready_recs = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_events_raw_texts_folder):
        for filename in filenames:
            if len(filename) != 8:
                continue
            raw_txt = ''
            with open(dirpath + filename, 'r') as in_fd:
                for ln in in_fd:
                    raw_txt += ln
                in_fd.close()
            clean_txt = sem_units_ext.text_clean(raw_txt)
            l_ready_recs.append((filename, clean_txt))
    logging.debug('[event_texts_clean] Clean %s events in %s secs.' % (len(l_ready_recs), time.time() - timer_start))

    df_out = pd.DataFrame(l_ready_recs, columns=['event_id', 'clean_txt'])
    df_out.to_pickle(global_settings.g_events_clean_texts_file_format.format(out_suffix))
    logging.debug('[event_texts_clean] All done in %s secs.' % str(time.time() - timer_start))


def event_sem_units(event_suffix):
    logging.debug('[event_sem_units] Starts...')
    timer_start = time.time()

    sem_units_ext = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
    logging.debug('[event_sem_units] Create sem_units_ext')

    df_clean_txt = pd.read_pickle(global_settings.g_events_clean_texts_file_format.format(event_suffix))
    logging.debug('[event_sem_units] Load in g_events_clean_texts_file')

    l_event_sents_recs = []
    for idx, clean_txt_rec in df_clean_txt.iterrows():
        even_id = clean_txt_rec['event_id']
        clean_txt = clean_txt_rec['clean_txt']
        l_sents = [sent.strip() for sent in clean_txt.split('\n')]
        l_tasks = [(even_id + '_' + str(sent_id), sent_txt) for sent_id, sent_txt in enumerate(l_sents)]
        sem_units_ext.sem_unit_extraction_thread(l_tasks, even_id, global_settings.g_events_sem_units_file_format, False)
        logging.debug('[event_sem_units] sem units done for %s in %s secs.' % (even_id, time.time() - timer_start))
        l_event_sents_recs.append((even_id, {task[0]: task[1] for task in l_tasks}))
    df_event_sents = pd.DataFrame(l_event_sents_recs, columns=['event_id', 'sents'])
    df_event_sents.to_pickle(global_settings.g_events_sents_file_format.format(event_suffix))
    logging.debug('[event_sem_units] Output g_events_sents_file')

    logging.debug('[event_sem_units] All done in %s secs.' % str(time.time() - timer_start))


def extract_phrases_from_cls_json_str(cls_json_str):
    if cls_json_str is None:
        return None
    s_covered_nodes = []
    s_phrases = set([])
    try:
        cls_json = json.loads(cls_json_str)
        cls_graph = nx.adjacency_graph(cls_json)
        for edge in cls_graph.edges(data=True):
            node_1 = edge[0]
            node_2 = edge[1]
            node_1_txt = cls_graph.nodes(data=True)[node_1]['txt']
            node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
            node_2_txt = cls_graph.nodes(data=True)[node_2]['txt']
            node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
            phrase = (node_1_txt, node_2_txt, node_1_pos + '_' + node_2_pos)
            s_covered_nodes.append(node_1)
            s_covered_nodes.append(node_2)
            s_phrases.add(phrase)
        s_covered_nodes = set(s_covered_nodes)
        if len(s_covered_nodes) < len(cls_graph.nodes):
            for node in cls_graph.nodes(data=True):
                if node[0] not in s_covered_nodes:
                    node_txt = node[1]['txt']
                    node_pos = node[1]['pos']
                    s_phrases.add((node_txt, node_pos))
    except Exception as err:
        print('[extract_phrases_from_cls_json_str] %s' % err)
        traceback.print_exc()
    if len(s_phrases) > 0:
        return list(s_phrases)
    return None


def extract_phrase_from_nps_str(nps_str):
    if nps_str is None:
        return None
    l_nps = [noun_phrase.strip() for noun_phrase in nps_str.split('\n')]
    s_phrase = set(l_nps)
    if len(s_phrase) > 0:
        s_phrase = set([(noun_phrase, 'NOUN') for noun_phrase in s_phrase])
        return list(s_phrase)
    return None


def event_su_to_phs(event_suffix):
    logging.debug('[event_su_to_phs] Starts...')
    timer_start = time.time()

    l_ready_recs = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_events_sem_units_folder):
        for filename in filenames:
            if filename[:3] != 'su_' or filename[-7:] != '.pickle':
                continue
            event_id = filename[3:-7]
            df_su = pd.read_pickle(dirpath + filename)
            d_sent_to_phs = dict()
            for idx, su_rec in df_su.iterrows():
                sent_id = su_rec[0]
                cls_json_str = su_rec[1]
                nps_str = su_rec[2]
                l_phs = []
                l_nps = extract_phrase_from_nps_str(nps_str)
                if l_nps is not None:
                    l_phs += l_nps
                l_cls = extract_phrases_from_cls_json_str(cls_json_str)
                if l_cls is not None:
                    l_phs += l_cls
                if len(l_phs) > 0:
                    l_phs = list(set([ph_item[0].lower() for ph_item in l_phs]))
                    d_sent_to_phs[sent_id] = l_phs
            l_ready_recs.append((event_id, d_sent_to_phs))
            logging.debug('[event_su_to_phs] Extract phs for %s in %s secs.' % (event_id, time.time() - timer_start))
    df_out = pd.DataFrame(l_ready_recs, columns=['event_id', 'sent_to_phs'])
    df_out.to_pickle(global_settings.g_events_sent_to_phs_file_format.format(event_suffix))
    logging.debug('[event_su_to_phs] All done in %s secs.' % str(time.time() - timer_start))


def load_lexvec_model():
    # global g_lexvec_model, g_embedding_len
    lexvec_model = lexvec.Model(global_settings.g_lexvec_vect_file_path)
    embedding_len = len(lexvec_model.word_rep('the'))
    logging.debug('[load_lexvec_model] The length of embeddings is %s' % embedding_len)
    return lexvec_model, embedding_len


def phrase_embedding(lexvec_model, embedding_len, phrase_str):
    '''
    Take a phrase (i.e. a sequence of tokens connected by whitespaces) and compute an embedding for it.
    This function acts as a wrapper of word embeddings. Refactor this function to adapt various word embedding models.
    :param
        phrase_str: An input phrase
    :return:
        An embedding for the phrase
    '''
    if global_settings.g_word_embedding_model == 'lexvec':
        if lexvec_model is None:
            raise Exception('lexvec_model is not loaded!')
        phrase_vec = np.zeros(embedding_len)
        l_words = [word.strip().lower() for word in phrase_str.split(' ')]
        for word in l_words:
            word_vec = lexvec_model.word_rep(word)
            phrase_vec += word_vec
        if not np.isfinite(phrase_vec).all():
            logging.error('Invalid embedding for %s!' % phrase_str)
            phrase_vec = np.zeros(embedding_len)
        return phrase_vec


def event_ph_to_embed(event_suffix):
    logging.debug('[event_ph_to_embed] Starts...')
    timer_start = time.time()

    lexvec_model, embedding_len = load_lexvec_model()

    with open(global_settings.g_tw_raw_phrases_phrase_to_id, 'r') as in_fd:
        d_ph_to_id = json.load(in_fd)
        in_fd.close()
    logging.debug('[event_ph_to_embed] Load g_tw_raw_phrases_phrase_to_id with %s phrases in %s secs.'
                  % (len(d_ph_to_id), time.time() - timer_start))

    df_phid_to_embed = pd.read_pickle(global_settings.g_tw_raw_phrases_embeds)
    df_phid_to_embed = df_phid_to_embed.set_index('phid')
    logging.debug('[event_ph_to_embed] Load g_tw_raw_phrases_embeds with %s embeds in %s secs.'
                  % (len(df_phid_to_embed), time.time() - timer_start))

    df_sent_to_phs = pd.read_pickle(global_settings.g_events_sent_to_phs_file_format.format(event_suffix))
    l_event_phs = []
    for idx, sent_rec in df_sent_to_phs.iterrows():
        d_sent_to_phs = sent_rec['sent_to_phs']
        for sent_id in d_sent_to_phs:
            l_event_phs += d_sent_to_phs[sent_id]
    l_event_phs = list(set(l_event_phs))
    logging.debug('[event_ph_to_embed] Extract %s phrases in %s secs.' % (len(l_event_phs), time.time() - timer_start))

    d_event_ph_to_phid = dict()
    l_event_phid_to_embed_recs = []
    event_phid = max(list(d_ph_to_id.values())) + 1
    cnt = 0
    for ph in l_event_phs:
        if ph in d_ph_to_id:
            d_event_ph_to_phid[ph] = int(d_ph_to_id[ph])
            l_event_phid_to_embed_recs.append((d_event_ph_to_phid[ph], df_phid_to_embed.loc[d_event_ph_to_phid[ph]]['embed']))
        else:
            d_event_ph_to_phid[ph] = event_phid
            event_phid += 1
            ph_embed = np.asarray(phrase_embedding(lexvec_model, embedding_len, ph), dtype=np.float32)
            l_event_phid_to_embed_recs.append((d_event_ph_to_phid[ph], ph_embed))
        cnt += 1
        if cnt % 5000 == 0 and cnt >= 5000:
            logging.debug('[event_ph_to_embed] Index and embed %s event phrase in %s secs.'
                          % (cnt, time.time() - timer_start))
    logging.debug('[event_ph_to_embed] Index and embed all %s event phrase in %s secs.'
                  % (cnt, time.time() - timer_start))

    with open(global_settings.g_events_ph_to_id_file_format.format(event_suffix), 'w+') as out_fd:
        json.dump(d_event_ph_to_phid, out_fd)
        out_fd.close()
    logging.debug('[event_ph_to_embed] Output g_events_ph_to_id_file with %s recs in %s secs.'
                  % (len(d_event_ph_to_phid), time.time() - timer_start))

    d_event_phid_to_ph = {d_event_ph_to_phid[ph]: ph for ph in d_event_ph_to_phid}
    with open(global_settings.g_events_id_to_ph_file_format.format(event_suffix), 'w+') as out_fd:
        json.dump(d_event_phid_to_ph, out_fd)
        out_fd.close()
    logging.debug('[event_ph_to_embed] Output g_events_id_to_ph_file with %s recs in %s secs.'
                  % (len(d_event_phid_to_ph), time.time() - timer_start))

    df_event_phid_to_embed = pd.DataFrame(l_event_phid_to_embed_recs, columns=['phid', 'embed'])
    df_event_phid_to_embed.to_pickle(global_settings.g_events_phid_to_embed_file_format.format(event_suffix))
    logging.debug('[event_ph_to_embed] Output g_events_phid_to_embed_file with %s recs in %s secs.'
                  % (len(df_event_phid_to_embed), time.time() - timer_start))

    l_sent_to_phids_recs = []
    for idx, sent_rec in df_sent_to_phs.iterrows():
        event_id = sent_rec['event_id']
        d_sent_to_phs = sent_rec['sent_to_phs']
        d_sent_to_phids = {sent_id: [d_event_ph_to_phid[ph] for ph in d_sent_to_phs[sent_id]] for sent_id in d_sent_to_phs}
        l_sent_to_phids_recs.append((event_id, d_sent_to_phids))
    df_sent_to_phids = pd.DataFrame(l_sent_to_phids_recs, columns=['event_id', 'sent_to_phids'])
    df_sent_to_phids.to_pickle(global_settings.g_events_sent_to_phids_file_format.format(event_suffix))
    logging.debug('[event_ph_to_embed] Output g_events_sent_to_phids_file with %s recs in %s secs.'
                  % (len(df_sent_to_phids), time.time() - timer_start))
    logging.debug('[event_ph_to_embed] All done.')


def event_ph_to_phrase_cluster(num_clusters, event_suffix):
    logging.debug('[event_ph_to_phrase_cluster] Starts...')

    df_phid_to_embed = pd.read_pickle(global_settings.g_events_phid_to_embed_file_format.format(event_suffix))
    logging.debug('[event_ph_to_phrase_cluster] Load g_events_phid_to_embed_file with %s recs.'
                  % str(len(df_phid_to_embed)))

    df_pc_info = pd.read_pickle(global_settings.g_tw_raw_phrases_clustering_info_format.format(str(num_clusters)))
    logging.debug('[event_ph_to_phrase_cluster] Load g_tw_raw_phrases_clustering_info with %s recs'
                  % str(len(df_pc_info)))

    l_ph_c_recs = []
    for idx, ph_embed_rec in df_phid_to_embed.iterrows():
        phid = ph_embed_rec['phid']
        ph_embed = ph_embed_rec['embed']
        l_sim = [0.0] * num_clusters
        for _, pc_vec_rec in df_pc_info.iterrows():
            pcid = int(pc_vec_rec['cid'])
            pc_vec = pc_vec_rec['cvec']
            sim = 1.0 - cosine(ph_embed, pc_vec)
            l_sim[pcid] = sim
        ph_c = np.argmax(l_sim)
        l_ph_c_recs.append((phid, ph_c))
    df_out = pd.DataFrame(l_ph_c_recs, columns=['phid', 'cid'])
    df_out.to_pickle(global_settings.g_events_phid_to_pcid_file_format.format(event_suffix))
    logging.debug('[event_ph_to_phrase_cluster] Output g_events_phid_to_pcid_file with %s recs.' % str(len(df_out)))
    logging.debug('[event_ph_to_phrase_cluster] All done.')


def update_event_sents(event_suffix):
    logging.debug('[update_event_sents] Start...')

    df_event_sents = pd.read_pickle(global_settings.g_events_sents_file_format.format(event_suffix))
    logging.debug('[update_event_sents] Load g_events_sents_file with %s recs.' % str(len(df_event_sents)))

    df_event_sent_to_phids = pd.read_pickle(global_settings.g_events_sent_to_phids_file_format.format(event_suffix))
    df_event_sent_to_phids = df_event_sent_to_phids.set_index('event_id')
    logging.debug('[update_event_sents] Load g_events_sent_to_phids_file with %s recs.'
                  % str(len(df_event_sent_to_phids)))

    for idx, rec in df_event_sents.iterrows():
        event_id = rec['event_id']
        d_sents = rec['sents']
        d_sent_to_phids = df_event_sent_to_phids.loc[event_id]['sent_to_phids']
        l_del_sent_ids = [sent_id for sent_id in d_sents if sent_id not in d_sent_to_phids]
        for sent_id in l_del_sent_ids:
            d_sents.pop(sent_id)
        rec['sents'] = d_sents
        logging.debug('[update_event_sents] %s event removes %s sents.' % (event_id, len(l_del_sent_ids)))
    df_event_sents.to_pickle(global_settings.g_events_sents_file_format.format(event_suffix))
    logging.debug('[update_event_sents] All done.')


def event_sents_to_csv(event_suffix):
    logging.debug('[event_sents_to_csv] Starts...')

    df_event_sents = pd.read_pickle(global_settings.g_events_sents_file_format.format(event_suffix))
    with open(global_settings.g_events_sents_csv_file_format.format(event_suffix), 'w+') as out_fd:
        csv_writer = csv.writer(out_fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'comment_text'])
        for idx, rec in df_event_sents.iterrows():
            d_sents = rec['sents']
            for sent_id in d_sents:
                csv_writer.writerow([sent_id, d_sents[sent_id]])
        out_fd.close()
    logging.debug('[event_sents_to_csv] All done.')


def event_sents_to_pc_embeds(num_clusters, event_suffix):
    logging.debug('[event_sents_to_pc_embeds] Starts...')

    df_phid_to_pcid = pd.read_pickle(global_settings.g_events_phid_to_pcid_file_format.format(event_suffix))
    df_phid_to_pcid = df_phid_to_pcid.set_index('phid')
    logging.debug('[event_sents_to_pc_embeds] Load g_events_phid_to_pcid_file with %s recs.'
                  % str(len(df_phid_to_pcid)))

    df_sent_to_phids = pd.read_pickle(global_settings.g_events_sent_to_phids_file_format.format(event_suffix))
    logging.debug('[event_sents_to_pc_embeds] Load g_events_sent_to_phids_file with %s recs.'
                  % str(len(df_sent_to_phids)))

    l_ready_recs = []
    for _, event_rec in df_sent_to_phids.iterrows():
        event_id = event_rec['event_id']
        d_sent_to_pc_vec = dict()
        d_sent_to_phids = event_rec['sent_to_phids']
        for sent_id in d_sent_to_phids:
            l_phids = d_sent_to_phids[sent_id]
            sent_pc_vec = np.zeros(num_clusters, dtype=np.float32)
            for phid in l_phids:
                ph_pc_vec = np.zeros(num_clusters, dtype=np.float32)
                ph_pc_vec[df_phid_to_pcid.loc[phid]] = 1.0
                sent_pc_vec += ph_pc_vec
            sent_pc_vec = sent_pc_vec / np.sum(sent_pc_vec)
            d_sent_to_pc_vec[sent_id] = sent_pc_vec
        l_ready_recs.append((event_id, d_sent_to_pc_vec))
        logging.debug('[event_sents_to_pc_embeds] %s sent_to_pc_vec for %s done.' % (len(d_sent_to_pc_vec), event_id))
    df_out = pd.DataFrame(l_ready_recs, columns=['event_id', 'sent_to_pcvec'])
    df_out.to_pickle(global_settings.g_events_sent_to_pcvec_file_format.format(event_suffix))
    logging.debug('[event_sents_to_pc_embeds] All done.')


def event_info(num_clusters, event_suffix):
    logging.debug('[event_info] Starts...')

    df_event_sent_to_pcvec = pd.read_pickle(global_settings.g_events_sent_to_pcvec_file_format.format(event_suffix))
    logging.debug('[event_info] Load g_events_sent_to_pcvec_file with %s recs.' % str(len(df_event_sent_to_pcvec)))

    with open(global_settings.g_tw_narrative_to_code_file, 'r') as in_fd:
        d_nar_to_code = json.load(in_fd)
        in_fd.close()
    logging.debug('[event_info] Load g_tw_narrative_to_code_file with %s recs.' % str(len(d_nar_to_code)))

    d_sent_to_nar = dict()
    with open(global_settings.g_events_sents_nar_csv_file_format.format(event_suffix), 'r') as in_fd:
        csv_reader = csv.reader(in_fd, delimiter=',')
        ln_cnt = 0
        for row in csv_reader:
            if ln_cnt == 0:
                ln_cnt += 1
            else:
                sent_id = row[0].strip()
                nars = row[2].strip()
                l_nars = [item.strip()[1:-1] for item in nars[1:-1].split(',')]
                nar_vec = np.zeros(len(d_nar_to_code), dtype=np.float32)
                for nar in l_nars:
                    if nar == '':
                        continue
                    nar_code = d_nar_to_code[nar]
                    nar_vec[nar_code] += 1.0
                if np.sum(nar_vec) > 0:
                    nar_vec = nar_vec / np.sum(nar_vec)
                d_sent_to_nar[sent_id] = nar_vec
        in_fd.close()
    logging.debug('[event_info] Load %s sent to nar_vec.' % str(len(d_sent_to_nar)))

    d_sent_to_stance = dict()
    with open(global_settings.g_events_sents_stance_csv_file_format.format(event_suffix), 'r') as in_fd:
        csv_reader = csv.reader(in_fd, delimiter=',')
        ln_cnt = 0
        for row in csv_reader:
            if ln_cnt == 0:
                ln_cnt += 1
            else:
                sent_id = row[0].strip()
                un_score = float(row[2].strip())
                am_score = float(row[3].strip())
                pm_score = float(row[4].strip())
                stance_vec = np.asarray([am_score, pm_score, un_score], dtype=np.float32)
                d_sent_to_stance[sent_id] = stance_vec
    logging.debug('[event_info] Load %s sent to stance_vec.' % str(len(d_sent_to_stance)))

    l_ready_recs = []
    for _, sent_pcvec_rec in df_event_sent_to_pcvec.iterrows():
        event_id = sent_pcvec_rec['event_id']
        d_sent_to_pcvec = sent_pcvec_rec['sent_to_pcvec']
        event_pcvec = np.zeros(num_clusters, dtype=np.float32)
        event_narvec = np.zeros(len(d_nar_to_code), dtype=np.float32)
        event_stancevec = np.zeros(3, dtype=np.float32)
        for sent_id in d_sent_to_pcvec:
            event_pcvec += d_sent_to_pcvec[sent_id]
            event_narvec += d_sent_to_nar[sent_id]
            event_stancevec += d_sent_to_stance[sent_id]
        if np.sum(event_pcvec) > 0:
            event_pcvec = event_pcvec / np.sum(event_pcvec)
        if np.sum(event_narvec) > 0:
            event_narvec = event_narvec / np.sum(event_narvec)
        if len(d_sent_to_pcvec) > 0:
            event_stancevec = event_stancevec / len(d_sent_to_pcvec)
        l_ready_recs.append((event_id, event_pcvec, event_narvec, event_stancevec))
    logging.debug('[event_info] Event info done for %s events.' % str(len(l_ready_recs)))

    df_out = pd.DataFrame(l_ready_recs, columns=['event_id', 'pcvec', 'nar', 'stance'])
    df_out.to_pickle(global_settings.g_events_info_file_format.format(event_suffix))
    logging.debug('[event_info] All done.')


def compare_events():
    logging.debug('[compare_events] Starts...')

    df_event_info = pd.read_pickle(global_settings.g_events_info_file)
    logging.debug('[compare_events] Load g_events_info_file with %s recs.' % str(len(df_event_info)))

    df_event_info = df_event_info.sort_values(by='event_id')
    df_event_info = df_event_info.reset_index(drop=True)
    mat_pc_js = np.zeros((len(df_event_info), len(df_event_info)), dtype=np.float32)
    mat_nar_js = np.zeros((len(df_event_info), len(df_event_info)), dtype=np.float32)
    for i in range(max(df_event_info.index)+1):
        pcvec_i = df_event_info.iloc[i]['pcvec']
        narvec_i = df_event_info.iloc[i]['nar']
        for j in range(i, max(df_event_info.index)+1):
            pcvec_j = df_event_info.iloc[j]['pcvec']
            narvec_j = df_event_info.iloc[j]['nar']
            if i == j:
                pc_js = 0.0
                nar_js = 0.0
            else:
                pc_js = jensenshannon(pcvec_i, pcvec_j, base=2)
                nar_js = jensenshannon(narvec_i, narvec_j, base=2)
            mat_pc_js[i][j] = pc_js
            mat_pc_js[j][i] = pc_js
            mat_nar_js[i][j] = nar_js
            mat_nar_js[j][i] = nar_js
    # plt.imshow(mat_nar_js, cmap='hot')
    # plt.colorbar()
    # plt.show()

    l_stances = []
    for i in df_event_info.index:
        stance = df_event_info.iloc[i]['stance']
        l_stances.append(stance)
    l_event_ids = list(df_event_info['event_id'])
    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.grid(True)
    plt.title('am')
    plt.xticks(rotation=90)
    plt.ylim(0.0, 1.0)
    plt.stem(l_event_ids, [stance[0] for stance in l_stances])

    plt.subplot(3, 1, 2)
    plt.grid(True)
    plt.title('pm')
    plt.xticks(rotation=90)
    plt.ylim(0.0, 1.0)
    plt.stem(l_event_ids, [stance[1] for stance in l_stances])

    plt.subplot(3, 1, 3)
    plt.grid(True)
    plt.title('?')
    plt.xticks(rotation=90)
    plt.ylim(0.0, 1.0)
    plt.stem(l_event_ids, [stance[2] for stance in l_stances])
    plt.show()



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'preprocess':
        event_suffix = '20190201#20190207'
        # event_texts_clean(event_suffix)
        # event_sem_units(event_suffix)
        # event_su_to_phs(event_suffix)
        # event_ph_to_embed(event_suffix)
        # update_event_sents(event_suffix)
        # event_ph_to_phrase_cluster(global_settings.g_num_phrase_clusters, event_suffix)
        event_sents_to_pc_embeds(global_settings.g_num_phrase_clusters, event_suffix)

    elif cmd == 'nar_stance':
        event_suffix = '20190201#20190207'
        # event_sents_to_csv(event_suffix)
        event_info(global_settings.g_num_phrase_clusters, event_suffix)
        # compare_events()