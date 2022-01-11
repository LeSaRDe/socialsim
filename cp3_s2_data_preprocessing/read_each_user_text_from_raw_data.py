import json
import logging
import time
import re
from gensim.parsing import preprocessing
from nltk import sent_tokenize
from gensim.summarization.textcleaner import get_sentences
import spacy
from spacy.lang.en import English



g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
             '20180527_20180603', '20180610_20180617', '20180624_20180701',
             '20180708_20180715', '20180722_20180729', '20180805_20180812',
             '20180819_20180826']
# g_task_name = '20180819_20180826'
g_path_prefix_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/'
# g_raw_data_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2.json'
g_raw_data_path_format = g_path_prefix_format + 'tw_wh_data_{1}_timesorted_dict.json'
g_text_out_path_format = g_path_prefix_format + 'tw_wh_data_{1}_full_user_text_clean_by_moment.json'
# g_user_time_type = 'MONTH'
g_user_time_type = 'MOMENT'
g_spacy_model = None
g_spacy_nlp = None
g_use_lemmatization = True
g_user_spacy_sent_split = True


def text_clean(dirty_text):
    # remove url
    clean_text = re.sub(r'url: [\S]*', '', dirty_text)
    clean_text = re.sub(r'http[\S]*', '', clean_text)
    # remove hashed ids
    clean_text = re.sub(r'@un: [\S]*', '', clean_text)
    clean_text = re.sub(r'[\S]{22}', '', clean_text)
    # sentence split
    if g_user_spacy_sent_split:
        spacy_doc = g_spacy_nlp(clean_text)
        l_sents = [sent.text for sent in spacy_doc.sents]
    else:
        # l_sents = list(get_sentences(dirty_text))
        l_sents = sent_tokenize(clean_text)
    l_clean_texts = []
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
            clean_text = [token.lemma_ for token in spacy_clean_text if not token.is_stop]
        else:
            clean_text = [word.strip() for word in clean_text.split(' ')]
        # remove non-english words
        clean_text = [word for word in clean_text if re.match(r'[^a-zA-Z]+', word) is None]
        # put words into one string
        clean_text = ' '.join(clean_text)
        if len(clean_text) > 0:
            l_clean_texts.append(clean_text)
    return l_clean_texts


# TODO
# we have to split a text into sentences by '\n' since there may not be any punctuation for sentences.
# we then need to remove '\r' and perhaps tokens leading with '\u' as well.
def extract_each_user_text(json_data):
    try:
        user_text = None
        user_id = None
        user_time = None
        if 'retweeted_status' in json_data:
            if json_data['lang'] != 'en' and 'google_translation' in json_data['retweeted_status']['extension']:
                user_text = json_data['retweeted_status']['extension']['google_translation'].replace('\r', '').replace('\n', '')
            else:
                user_text = json_data['retweeted_status']['text_m'].replace('\r', '').replace('\n', '')
            if 'quoted_status' in json_data:
                if json_data['quoted_status']['lang'] != 'en' and 'google_translation' in json_data['quoted_status']['extension']:
                    user_text = user_text + '\r\n' + json_data['quoted_status']['extension']['google_translation'].replace('\r', '').replace('\n', '')
                else:
                    user_text = user_text + '\r\n' + json_data['quoted_status']['text_m'].replace('\r', '').replace('\n', '')
        elif 'quoted_status' in json_data:
            if json_data['lang'] != 'en' and 'google_translation' in json_data['extension']:
                user_text = json_data['extension']['google_translation'].replace('\r', '').replace('\n', '')
            else:
                user_text = json_data['text_m'].replace('\r', '').replace('\n', '')
            if json_data['quoted_status']['lang'] != 'en' and 'google_translation' in json_data['quoted_status']['extension']:
                user_text = user_text + '\r\n' + json_data['quoted_status']['extension']['google_translation'].replace('\r', '').replace('\n', '')
            else:
                user_text  = user_text + '\r\n' + json_data['quoted_status']['text_m'].replace('\r', '').replace('\n', '')
        else:
            if json_data['lang'] != 'en' and 'google_translation' in json_data['extension']:
                user_text = json_data['extension']['google_translation'].replace('\r', '').replace('\n', '')
            else:
                user_text = json_data['text_m'].replace('\r', '').replace('\n', '')
        if 'id_str_h' in json_data['user']:
            user_id = json_data['user']['id_str_h']
        if 'created_at' in json_data:
            user_time = json_data['created_at'].split(' ')
    except:
        print(json_data)
    if user_id is None:
        logging.error('None user_id happens %s' % json_data)
    if user_text is None:
        logging.error('None user_text happens %s' % json_data)
    if user_time is None:
        logging.error('None user_time happens %s' % json_data)
    return user_id, text_clean(user_text), user_time


def init_user_data():
    user_data = dict()
    if g_user_time_type == 'MONTH':
        user_data['Jan'] = []
        user_data['Feb'] = []
        user_data['Mar'] = []
        user_data['Apr'] = []
        user_data['May'] = []
        user_data['Jun'] = []
        user_data['Jul'] = []
        user_data['Aug'] = []
        user_data['Sep'] = []
        user_data['Oct'] = []
        user_data['Nov'] = []
        user_data['Dec'] = []
    return user_data


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


def main():
    global g_spacy_model, g_spacy_nlp
    if g_user_spacy_sent_split:
        g_spacy_nlp = English()
        g_spacy_nlp.add_pipe(g_spacy_nlp.create_pipe('sentencizer'))
    if g_use_lemmatization:
        g_spacy_model = spacy.load("en_core_web_lg")

    for week in g_l_weeks:
        print('%s:' % week)
        d_each_user_texts = dict()
        with open(g_raw_data_path_format.format(week, week), 'r') as in_fd:
            tw_dict = json.load(in_fd)
            # l_lines = in_fd.readlines()
            start = time.time()
            # for i in range(0, len(l_lines)):
            i = 0
            for tw_id in tw_dict:
                i += 1
                # json_data = json.loads(l_lines[i])
                json_data = tw_dict[tw_id]
                user_id, l_user_texts, user_time = extract_each_user_text(json_data)
                if user_id is None or l_user_texts is None or len(l_user_texts) == 0:
                    continue
                if user_id in d_each_user_texts:
                    time_key = get_user_time(user_time)
                    if time_key not in d_each_user_texts[user_id]:
                        d_each_user_texts[user_id][time_key] = []
                    d_each_user_texts[user_id][time_key] += l_user_texts
                else:
                    d_each_user_texts[user_id] = dict()
                    d_each_user_texts[user_id][get_user_time(user_time)] = []
                    d_each_user_texts[user_id][get_user_time(user_time)] += l_user_texts
                if i % 5000 == 0:
                    logging.debug('%s lines done in %s sec.' % (str(i+1), time.time()-start))
            logging.debug('All done in %s sec.' % str(time.time() - start))
        in_fd.close()

        with open(g_text_out_path_format.format(week, week), 'w+') as out_fd:
            json.dump(d_each_user_texts, out_fd)
        out_fd.close()
        print()


def test_main():
    sample_tweet_id = 'zx1Knb5xbJM4Y2ZGNAF6UQ'
    sample_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/20180513_20180520/tw_wh_data_20180513_20180520_timesorted_dict.json'
    with open(sample_path, 'r') as in_fd:
        json_data = json.load(in_fd)
        sample_tweet_json = json_data[sample_tweet_id]
        in_fd.close()
    clean_text = extract_each_user_text(sample_tweet_json)
    print(clean_text)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # main()
    test_main()