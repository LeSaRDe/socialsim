from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import gensim
from time import time
import logging
import json

# '20180415_20180422', '20180429_20180506', '20180513_20180520',
#             '20180527_20180603', '20180610_20180617', '20180624_20180701',
#             '20180708_20180715', , '20180805_20180812',
#             '20180819_20180826'


g_l_weeks = [ '20180722_20180729']
g_l_top_users= ['I_Wry8ROzibHG44UBImpiQ', 'a0fYqjn3qCvgH6MYNqZRew', 'fDk3wnVbzNUk9l-47tt8UQ', 'b0Z0FKyh_ciPQJx0Onzbqg']
g_task_name = '20180819_20180826'
g_in_data = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/'+g_task_name+'/tw_wh_data_'+g_task_name+'_10_sample_user_text_clean_by_moment.json'
g_in_data_format = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/tw_wh_data_{1}_10_sample_user_text_clean_by_moment.json'
# g_in_data = 'Tng_an_WH_Twitter_v2_user_text_clean.json'
g_output = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/'+g_task_name+'/tw_wh_data_'+g_task_name+'_10_sample_topics.txt'
# g_time_type = 'MONTH'
# g_time_type = 'DAY'
# g_time_type = 'WEEK'
g_time_type = 'ALL'

n_samples = 2000
# n_features = 1000
n_features = None
# n_components = 10
n_components_nmf = 10
n_components_lda = 10
n_top_words = 20
learning_method_lda = 'batch'
g_en_lda = True
g_en_nmf = True


def load_data():
    dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    data_samples = dataset.data[:n_samples]
    return data_samples


def load_data_by_time(target_week=None, target_month=None, target_user_id=None):
    user_data_by_time = dict()
    with open(g_in_data_format.format(target_week, target_week), 'r') as in_fd:
        json_data = json.load(in_fd)
        if g_time_type == 'MONTH':
            for user_id in json_data:
                if target_month != None:
                    if target_month in json_data[user_id]:
                        if target_month in user_data_by_time:
                            user_data_by_time[target_month] += json_data[user_id][target_month]
                        else:
                            user_data_by_time[target_month] = json_data[user_id][target_month]
                else:
                    for month in json_data[user_id]:
                        if month in user_data_by_time:
                            user_data_by_time[month] += json_data[user_id][month]
                        else:
                            user_data_by_time[month] = json_data[user_id][month]
        elif g_time_type == 'DAY':
            for user_id in json_data:
                for moment in json_data[user_id]:
                    day = moment[:8]
                    if day in user_data_by_time:
                        user_data_by_time[day] += json_data[user_id][moment]
                    else:
                        user_data_by_time[day] = json_data[user_id][moment]
        elif g_time_type == 'ALL':
            user_data_by_time = []
            for user_id in json_data:
                if target_user_id is not None and user_id != target_user_id:
                    continue
                for moment in json_data[user_id]:
                    user_data_by_time += json_data[user_id][moment]
    in_fd.close()
    return user_data_by_time


def print_top_words(model, feature_names, n_top_words):
    # with open(g_output, 'w+') as out_fd:
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        # out_fd.write(message + '\n')
        print(message)
    # out_fd.close()
    print()


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])


def compute_topics(text_data):
    if g_en_nmf:
        tfidf_vecterizer = TfidfVectorizer(max_df=0.70,
                                           min_df=2,
                                           max_features=n_features)
        tfidf = tfidf_vecterizer.fit_transform(text_data)
        nmf = NMF(n_components=n_components_nmf, random_state=1, alpha=0.1, l1_ratio=0.5, solver='cd', beta_loss='frobenius', max_iter=1000).fit(tfidf)
        tfidf_feature_names = tfidf_vecterizer.get_feature_names()
        print('NMF topics:')
        # print_top_words(nmf, tfidf_feature_names, n_top_words)
        print_topics(nmf, tfidf_vecterizer, n_top_words)
        test_text = "douma tara"
        x = nmf.transform(tfidf_vecterizer.transform([test_text]))[0]
        print(x)

    if g_en_lda:
        tf_vectorizer = CountVectorizer(max_df=0.70,
                                        min_df=2,
                                        max_features=n_features)
        tf = tf_vectorizer.fit_transform(text_data)
        lda = LatentDirichletAllocation(n_components=n_components_lda, max_iter=5,
                                        learning_method=learning_method_lda,
                                        learning_offset=50.,
                                        random_state=0, n_jobs=-1)
        lda.fit(tf)
        tf_feature_names = tf_vectorizer.get_feature_names()
        print('LDA topics:')
        # print_top_words(lda, tf_feature_names, n_top_words)
        print_topics(lda, tf_vectorizer, n_top_words)
        test_text = "douma tara"
        x = lda.transform(tf_vectorizer.transform([test_text]))[0]
        print(x)


def main():
    start = time()
    user_data_by_time = load_data_by_time()
    for time_point in sorted(user_data_by_time):
        print('Topics for %s:' % time_point)
        text_data = user_data_by_time[time_point]
        compute_topics(text_data)
        # print()
    # logging.debug('All done in % seconds.' % str(time() - start))


def sample_main():
    global g_task_name
    # d_user_data_by_time = dict()
    for week in g_l_weeks:
        g_task_name = week
        user_data_week = load_data_by_time(week, None, 'I_Wry8ROzibHG44UBImpiQ')
        # d_user_data_by_time[week] = user_data_week
        print('Topics for %s:' % week)
        compute_topics(user_data_week)


def test_main():
    text_data = load_data()
    compute_topics(text_data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # main()
    # sample_main()
    test_main()
