import logging
import re
from nltk import sent_tokenize
from gensim.parsing import preprocessing
import spacy
from spacy.lang.en import English
from datetime import datetime
import sqlite3


g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_time_int_path = g_time_series_data_path_prefix + 'time_intervals.txt'
g_stopword_path = g_time_series_data_path_prefix + 'stopwords.txt'
g_retweet_db_path = g_time_series_data_path_prefix + 'wh_tw_ret.db'

g_time_format = '%Y%m%d'
g_user_time_type = 'MOMENT'
g_spacy_model = None
g_spacy_nlp = None
g_use_lemmatization = True
g_use_spacy_sent_split = True
g_use_custom_stopwords = True
g_d_stopword = dict()


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


def get_user_time(user_time):
    if isinstance(user_time, str):
        user_time = user_time.split(' ')
    if g_user_time_type == 'MONTH':
        month = translate_month(user_time[1])
        year = user_time[5]
        return year + month
    elif g_user_time_type == 'MOMENT':
        month = translate_month(user_time[1])
        year = user_time[5]
        day = user_time[2]
        time_str = user_time[3].split(':')
        time_str = ''.join([t_field.strip() for t_field in time_str])
        return year + month + day + time_str


def get_tweet_type(t_json):
    t_type = None
    if 'in_reply_to_status_id_h' in t_json and t_json['in_reply_to_status_id_h'] != '':
        t_type = 'r'
    elif 'retweeted_status' in t_json:
        t_type = 't'
    elif 'quoted_status' in t_json:
        t_type = 'q'
    else:
        t_type = 'n'
    return t_type


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


def get_raw_text(t_json, t_type):
    l_user_text = []
    if t_type == 't':
        if t_json['lang'] != 'en' and 'google_translation' in t_json['retweeted_status']['extension']:
            l_user_text = t_json['retweeted_status']['extension']['google_translation'].replace('\r', '').split('\n')
        elif t_json['lang'] == 'en':
            l_user_text = t_json['retweeted_status']['text_m'].replace('\r', '').split('\n')
        if 'quoted_status' in t_json:
            if t_json['quoted_status']['lang'] != 'en' and 'google_translation' in t_json['quoted_status']['extension']:
                l_user_text += t_json['quoted_status']['extension']['google_translation'].replace('\r', '').split('\n')
            elif t_json['quoted_status']['lang'] == 'en':
                l_user_text += t_json['quoted_status']['text_m'].replace('\r', '').split('\n')
    elif t_type == 'q':
        if t_json['lang'] != 'en' and 'google_translation' in t_json['extension']:
            l_user_text = t_json['extension']['google_translation'].replace('\r', '').split('\n')
        elif t_json['lang'] == 'en':
            l_user_text = t_json['text_m'].replace('\r', '').split('\n')
        if t_json['quoted_status']['lang'] != 'en' and 'google_translation' in t_json['quoted_status']['extension']:
            l_user_text += t_json['quoted_status']['extension']['google_translation'].replace('\r','').split('\n')
        elif t_json['quoted_status']['lang'] == 'en':
            l_user_text += t_json['quoted_status']['text_m'].replace('\r', '').split('\n')
    else:
        if t_json['lang'] != 'en' and 'google_translation' in t_json['extension']:
            l_user_text = t_json['extension']['google_translation'].replace('\r', '').split('\n')
        elif t_json['lang'] == 'en':
            l_user_text = t_json['text_m'].replace('\r', '').split('\n')
    return l_user_text


def get_org_text(t_json, t_type):
    l_user_text = []
    if t_type != 't':
        if t_json['lang'] != 'en' and 'google_translation' in t_json['extension']:
            l_user_text = t_json['extension']['google_translation'].replace('\r', '').split('\n')
        elif t_json['lang'] == 'en':
            l_user_text = t_json['text_m'].replace('\r', '').split('\n')
    return l_user_text


def text_clean_init():
    global g_spacy_model, g_spacy_nlp, g_d_stopword
    if g_use_spacy_sent_split:
        g_spacy_nlp = English()
        g_spacy_nlp.add_pipe(g_spacy_nlp.create_pipe('sentencizer'))
    if g_use_lemmatization:
        g_spacy_model = spacy.load("en_core_web_lg")
    if g_use_custom_stopwords:
        with open(g_stopword_path, 'r') as in_fd:
            sw = in_fd.readline()
            while sw:
                sw = sw.strip()
                if sw not in g_d_stopword:
                    g_d_stopword[sw] = None
                sw = in_fd.readline()
            in_fd.close()


def text_clean(dirty_text):
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
        if g_use_spacy_sent_split:
            spacy_doc = g_spacy_nlp(clean_text)
            l_sents = [sent.text for sent in spacy_doc.sents]
        else:
            # l_sents = list(get_sentences(dirty_text))
            l_sents = sent_tokenize(clean_text)
        # l_clean_texts = []
        for dirty_sent in l_sents:
            # remove non-word char
                clean_text = re.sub(r'[^\w\s]', ' ', dirty_sent)
            # remove all other dirty stuffs
            clean_text = preprocessing.strip_tags(clean_text)
            clean_text = preprocessing.strip_punctuation(clean_text)
            clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
            clean_text = preprocessing.strip_numeric(clean_text)
            clean_text = preprocessing.strip_short(clean_text)
            if g_use_lemmatization:
                # lemmatization & stopwords
                spacy_clean_text = g_spacy_model(clean_text)
                clean_text = [token.lemma_ for token in spacy_clean_text if not token.is_stop and not is_stop_word(token)]
            else:
                clean_text = [word.strip() for word in clean_text.split(' ')]
            # remove non-english words
            clean_text = [word for word in clean_text if re.match(r'[^a-zA-Z]+', word) is None]
            # put words into one string
            clean_text = ' '.join(clean_text)
            clean_text = additional_clean(clean_text)
            if len(clean_text) > 0:
                l_clean_texts.append(clean_text)
    return l_clean_texts


def is_stop_word(word):
    if g_use_custom_stopwords:
        if word in g_d_stopword:
            return True
        else:
            return False
    else:
        return False


def additional_clean(dirty_text):
    clean_text = re.sub(r'White\sHelmets', 'whitehelmets', dirty_text, 0, re.I)
    clean_text = re.sub(r'amp', '', clean_text, 0, re.I)
    return clean_text


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
    # logging.debug('%s time intervals are read.' % len(l_time_ints))
    return l_time_ints


def time_int_to_time_int_str(time_int):
    return time_int[0].strftime(g_time_format) + '_' + time_int[1].strftime(g_time_format)