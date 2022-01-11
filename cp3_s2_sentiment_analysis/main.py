import json
import logging
from datetime import datetime, timedelta
import os
import time
import sqlite3
import multiprocessing
import threading
import math
# from gensim.models import KeyedVectors
# import numpy as np
import sys
sys.path.insert(0, '/home/mf3jh/workspace/cp3_s2_data_preprocessing')
import data_preprocessing_utils
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sent_tokenize
from gensim.parsing import preprocessing
import re


g_tw_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
# g_sen_int_folder_path = g_path_prefix + 'sentiments_twitter/'
# g_sen_int_file_path = g_sen_int_folder_path + '{0}.json'
g_raw_tw_data_path = g_tw_path_prefix + 'Tng_an_WH_Twitter_v3.json'
g_tw_sen_db_path = g_tw_path_prefix + 'wh_tw_v3_sen.db'

g_yt_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/YouTube/'
g_raw_yt_videos_data_path = g_yt_path_prefix + 'Tng_an_Videos.v2.json'
g_raw_yt_comments_data_path = g_yt_path_prefix + 'Tng_an_Comments.v2.json'
g_yt_sen_db_path = g_yt_path_prefix + 'wh_yt_v2_sen.db'


def yt_text_clean(dirty_text):
    if dirty_text is None or dirty_text == '':
        return []
    l_clean_texts = []
    l_dirty_sents = dirty_text.split('\n')
    for raw_dirt_sent in l_dirty_sents:
        # remove url
        clean_text = re.sub(r'url: [\S]*', '', raw_dirt_sent)
        clean_text = re.sub(r'http[\S]*', '', clean_text)
        # sentence split
        l_sents = sent_tokenize(clean_text)
        # l_clean_texts = []
        for dirty_sent in l_sents:
            clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
            clean_text = [word.strip() for word in clean_text.split(' ')]
            # put words into one string
            clean_text = ' '.join(clean_text)
            if len(clean_text) > 0:
                l_clean_texts.append(clean_text)
    return l_clean_texts


def read_raw_yt_data_to_db(sentiment_analyzer):
    d_raw_texts = dict()
    with open(g_raw_yt_videos_data_path, 'r') as in_videos_fd:
        raw_video_line = in_videos_fd.readline()
        count = 0
        while raw_video_line:
            raw_video_json = json.loads(raw_video_line)
            vid = raw_video_json['id_h']
            if raw_video_json['extension']['detected_language_description'] != 'en':
                if 'google_translation_description' in raw_video_json['extension']:
                    d_raw_texts[vid] = yt_text_clean(raw_video_json['extension']['google_translation_description'])
                else:
                    d_raw_texts[vid] = []
            else:
                d_raw_texts[vid] = yt_text_clean(raw_video_json['extension']['cleaned_description'])
            count += 1
            raw_video_line = in_videos_fd.readline()
        in_videos_fd.close()
    logging.debug('%s videos have been scanned.' % count)

    with open(g_raw_yt_comments_data_path, 'r') as in_comments_fd:
        raw_comment_line = in_comments_fd.readline()
        count = 0
        while raw_comment_line:
            raw_comment_json = json.loads(raw_comment_line)
            vid = raw_comment_json['snippet']['videoId_h']
            if 'detected_language' not in raw_comment_json['extension']:
                count += 1
                raw_comment_line = in_comments_fd.readline()
                continue
            if raw_comment_json['extension']['detected_language'] != 'en':
                if 'google_translation' in raw_comment_json['extension']:
                    l_raw_comment_texts = yt_text_clean(raw_comment_json['extension']['google_translation'])
                else:
                    l_raw_comment_texts = []
            else:
                l_raw_comment_texts = yt_text_clean(raw_comment_json['extension']['cleaned_text'])
            if vid in d_raw_texts:
                d_raw_texts[vid] += l_raw_comment_texts
            else:
                d_raw_texts[vid] = l_raw_comment_texts
            count += 1
            raw_comment_line = in_comments_fd.readline()
        in_comments_fd.close()
    logging.debug('%s comments have been scanned.' % count)

    db_conn = sqlite3.connect(g_yt_sen_db_path)
    db_cur = db_conn.cursor()
    sql_str = '''INSERT INTO wh_yt_sen VALUES (?, ?, ?)'''
    count = 0
    timer_start = time.time()
    for vid in d_raw_texts:
        d_raw_texts[vid] = '\n'.join(d_raw_texts[vid])
        sentiments = sentiment_analyzer.polarity_scores(d_raw_texts[vid])
        sentiment_score = sentiments['compound']
        db_cur.execute(sql_str, (vid, d_raw_texts[vid], str(sentiment_score)))
        if count % 1000 and count >= 1000:
            db_conn.commit()
            db_cur.execute('''DROP INDEX idx_vid''')
            db_cur.execute('''CREATE INDEX idx_vid on wh_yt_sen (vid)''')
            db_conn.commit()
            logging.debug('%s video records have been written in %s seconds.' % (count, str(time.time()-timer_start)))
        count += 1
    db_conn.commit()
    db_cur.execute('''DROP INDEX idx_vid''')
    db_cur.execute('''CREATE INDEX idx_vid on wh_yt_sen (vid)''')
    db_conn.commit()
    logging.debug('%s video records have been written in %s seconds.' % (count, str(time.time()-timer_start)))
    logging.debug('All done.')
    db_conn.close()


def tw_text_clean(dirty_text):
    if dirty_text is None or dirty_text == '':
        return []
    l_clean_texts = []
    l_dirty_sents = dirty_text.split('\n')
    for raw_dirt_sent in l_dirty_sents:
        # remove url
        clean_text = re.sub(r'url: [\S]*', '', raw_dirt_sent)
        clean_text = re.sub(r'http[\S]*', '', clean_text)
        # remove hashed ids
        clean_text = re.sub(r'@un:\s[\S]{22}\s', ' ', clean_text)
        clean_text = re.sub(r'\s[\S]{22}\s', ' ', clean_text)
        # sentence split
        l_sents = sent_tokenize(clean_text)
        # l_clean_texts = []
        for dirty_sent in l_sents:
            clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
            clean_text = [word.strip() for word in clean_text.split(' ')]
            # put words into one string
            clean_text = ' '.join(clean_text)
            if len(clean_text) > 0:
                l_clean_texts.append(clean_text)
    return l_clean_texts


def read_raw_tw_data_to_db(sentiment_analyzer):
    with open(g_raw_tw_data_path, 'r') as in_fd:
        raw_line = in_fd.readline()
        count = 0
        sql_str = '''INSERT INTO wh_tw_sen VALUES (?, ?, ?)'''
        db_conn = sqlite3.connect(g_tw_sen_db_path)
        db_cur = db_conn.cursor()
        timer_start = time.time()
        while raw_line:
            raw_json = json.loads(raw_line)
            tid = raw_json['id_str_h']
            l_raw_texts = data_preprocessing_utils.get_raw_text(raw_json, data_preprocessing_utils.get_tweet_type(raw_json))
            l_clean_texts = []
            for raw_text in l_raw_texts:
                l_clean_sents = tw_text_clean(raw_text)
                l_clean_texts += l_clean_sents
            cat_text = '\n'.join(l_clean_texts)
            sentiments = sentiment_analyzer.polarity_scores(cat_text)
            sentiment_score = sentiments['compound']
            db_cur.execute(sql_str, (tid, cat_text, str(sentiment_score)))
            count += 1
            if count % 10000 == 0 and count >= 10000:
                db_conn.commit()
                db_cur.execute('''DROP INDEX idx_tid''')
                db_cur.execute('''CREATE INDEX idx_tid on wh_tw_sen (tid)''')
                db_conn.commit()
                logging.debug('%s count recs have been written in %s secs.' % (count, str(time.time()-timer_start)))
            raw_line = in_fd.readline()
        db_conn.commit()
        db_cur.execute('''DROP INDEX idx_tid''')
        db_cur.execute('''CREATE INDEX idx_tid on wh_tw_sen (tid)''')
        db_conn.commit()
        logging.debug('All done in %s secs.' % str(time.time()-timer_start))
        db_conn.close()
        in_fd.close()


# def operate_db_multithreads(op_func, add_args=None):
#     with open(g_raw_data_path, 'r') as in_fd:
#         l_raw_lines = in_fd.readlines()
#         line_count = len(l_raw_lines)
#
#     batch_size = math.ceil(line_count / multiprocessing.cpu_count())
#     l_l_raw_lines = []
#     for i in range(0, line_count, batch_size):
#         if i + batch_size < line_count:
#             l_l_raw_lines.append(l_raw_lines[i:i + batch_size])
#         else:
#             l_l_raw_lines.append(l_raw_lines[i:])
#
#     l_threads = []
#     t_id = 0
#     for l_each_batch in l_l_raw_lines:
#         if add_args is not None:
#             t = threading.Thread(target=op_func, args=(l_each_batch,) + add_args)
#         else:
#             t = threading.Thread(target=op_func, args=(l_each_batch,))
#         t.setName('nec_to_db_t' + str(t_id))
#         t.start()
#         l_threads.append(t)
#         t_id += 1
#
#     while len(l_threads) > 0:
#         for t in l_threads:
#             if t.is_alive():
#                 t.join(1)
#             else:
#                 l_threads.remove(t)
#                 logging.debug('Thread %s is finished.' % t.getName())
#
#     logging.debug('All nec data has been written to DB.')


def main():
    sentiment_analyzer = SentimentIntensityAnalyzer()
    # read_raw_tw_data_to_db(sentiment_analyzer)
    read_raw_yt_data_to_db(sentiment_analyzer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()