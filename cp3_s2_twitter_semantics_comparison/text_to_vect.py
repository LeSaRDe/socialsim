import json
import logging
import sent2vec
import scipy.spatial.distance as scipyd
import multiprocessing as mp
import threading
import time


g_l_weeks = ['20180415_20180422', '20180429_20180506', '20180513_20180520',
            '20180527_20180603', '20180610_20180617', '20180624_20180701',
            '20180708_20180715', '20180722_20180729', '20180805_20180812',
            '20180819_20180826']
g_l_top_users= ['I_Wry8ROzibHG44UBImpiQ', 'a0fYqjn3qCvgH6MYNqZRew', 'fDk3wnVbzNUk9l-47tt8UQ', 'b0Z0FKyh_ciPQJx0Onzbqg']
g_l_tw_data_file_paths = ['/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/{0}/tw_wh_data_{1}_10_sample_dict.json'.format(item, item) for item in g_l_weeks]
g_clean_text_file_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_user_text_clean_by_moment.json'
g_sent2vec_model_file_path = 'sent2vec/twitter_unigrams.bin'
g_output_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/'


def compute_text_vect_thread_func(tw_data_file_path, d_tw_clean_text, sent2vec_model, t_name):
    fields = t_name.split('_')
    start = fields[0] + '000000'
    end = fields[1] + '000000'
    l_overall_text = []
    d_top_user_text = dict()
    timer_s = time.time()
    with open(tw_data_file_path, 'r') as in_fd:
        tw_data = json.load(in_fd)
        l_users = []
        for tw_id in tw_data:
            user_id = tw_data[tw_id]['user']['id_str_h']
            l_users.append(user_id)
        del tw_data
        for user_idx, user in enumerate(l_users):
            if user not in d_tw_clean_text:
                continue
            d_user_text = d_tw_clean_text[user]
            for text_time in d_user_text:
                if start <= text_time < end:
                    l_overall_text += d_user_text[text_time]
                    if user in g_l_top_users:
                        if user not in d_top_user_text:
                            d_top_user_text[user] = d_user_text[text_time]
                        else:
                            d_top_user_text[user] += d_user_text[text_time]
            if user_idx % 100 == 0 and user_idx > 100:
                logging.debug('%s users have been scanned in %s seconds.' % (user_idx, time.time()-timer_s))
        logging.debug('All users have been scanned in %s second.' % str(time.time()-timer_s))
    in_fd.close()

    embs = sent2vec_model.embed_sentences(l_overall_text, mp.cpu_count())
    overall_text_vect = [sum(x) / len(l_overall_text) for x in zip(*embs)]
    for user in d_top_user_text:
        embs = sent2vec_model.embed_sentences(d_top_user_text[user], mp.cpu_count())
        user_text_vect = [sum(x) / len(d_top_user_text[user]) for x in zip(*embs)]
        d_user_text[user] = user_text_vect
    with open(g_output_prefix+t_name+'/text_vect.txt', 'w+') as out_fd:
        d_output = dict()
        d_output[t_name] = overall_text_vect
        json.dump(d_output, out_fd, indent=4)
    out_fd.close()
    with open(g_output_prefix+t_name+'/top_user_text_vect.txt', 'w+') as out_fd:
        json.dump(d_top_user_text, out_fd, indent=4)
    out_fd.close()


def compute_text_vects_from_sample(sent2vec_model):
    d_week_text = dict()
    with open(g_clean_text_file_path, 'r') as in_clean_text_fd:
        d_tw_clean_text = json.load(in_clean_text_fd)
        l_jobs = []
        for week_idx, tw_data_file_path in enumerate(g_l_tw_data_file_paths):
            job_t_name = g_l_weeks[week_idx]
            job_t = threading.Thread(target=compute_text_vect_thread_func, args=(tw_data_file_path, d_tw_clean_text, sent2vec_model, job_t_name))
            job_t.setName(job_t_name)
            job_t.start()
            l_jobs.append(job_t)
        while len(l_jobs) > 0:
            for t in l_jobs:
                t.join(2)
                if not t.is_alive():
                    logging.debug('%s is done.' % t.getName())
                    l_jobs.remove(t)
        del d_tw_clean_text
    in_clean_text_fd.close()



def main():
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(g_sent2vec_model_file_path)
    compute_text_vects_from_sample(sent2vec_model)


if __name__ == '__main__':
    main()