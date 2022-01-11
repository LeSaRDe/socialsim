import logging
import sent2vec
import scipy.spatial.distance as scipyd
import multiprocessing as mp


g_sent2vec_model_file_path = 'sent2vec/twitter_unigrams.bin'

g_topic_1 = ['advisor','alqaeda','amp','assad','atma','attack','auxilliary','barrel','block','bolton','bomb','border','chemical','chemicalattack','chlorine','collusion','container','continue','coordinate','crime','deliver','encourage','euro','facebook','factory','falseflag','flag','fund','germany','hallouz','helmet','helmets','idleb','idlib','infographic','isis','jerusalem','jisr','john','million','month','nat','near','need','news','nusra','old','openly','organisation','perpetrate','plan','post','pretext','propaganda','provide','province','qaeda','remain','sarcastic','security','shughur','soon','specialize','stage','support','syria','syriaus','terrorist','threaten','today','transport','trigger','turkish','use','utmost','van','video','weapons','west','white','whitehelmets','woman','wonder','working','false']
g_topic_2 = ['helmets','hrw','human','israel','paint','recognize','rights','syria','watch','white','whitehelmets']


def main():
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(g_sent2vec_model_file_path)

    embs = sent2vec_model.embed_sentences(g_topic_1, mp.cpu_count())
    topic_1_vect = [sum(x) / len(g_topic_1) for x in zip(*embs)]

    embs = sent2vec_model.embed_sentences(g_topic_2, mp.cpu_count())
    topic_2_vect = [sum(x) / len(g_topic_2) for x in zip(*embs)]

    sim = 1.0 - scipyd.cosine(topic_1_vect, topic_2_vect)
    print('sim = %s' % sim)


if __name__ == '__main__':
    main()