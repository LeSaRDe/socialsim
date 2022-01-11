import networkx as nx
import logging
import sent2vec
import scipy.spatial.distance as scipyd
import json
import multiprocessing as mp
from sklearn import metrics


g_sent2vec_model_file_path = 'sent2vec/twitter_unigrams.bin'
g_path_prefix = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/'
g_time_series_data_path_prefix = g_path_prefix + 'time_series_data/'
g_time_series_data_format = g_time_series_data_path_prefix + '{0}/{1}_nec_data'
g_fsgraph_full_gml_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph.gml'
g_fsgraph_top_gml_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph_top50.gml'
g_tsgraph_full_gml_format = g_time_series_data_path_prefix + '{0}/{1}_tsgraph_sim50.gml'
g_tsgraph_top_gml_format = g_time_series_data_path_prefix + '{0}/{1}_tsgraph_top50.gml'
g_tsgraph_top_pure_gml_format = g_time_series_data_path_prefix + '{0}/{1}_tsgraph_top50_pure.gml'
time_int_str = '20180718_20180724'
g_sim_threshold = 0.9


g_fsgraph_top_sample_format = g_time_series_data_path_prefix + '{0}/{1}_fsgraph_sample.gml'
g_tsgraph_top_sample_gml_format = g_time_series_data_path_prefix + '{0}/{1}_tsgraph_sample_pure.gml'
g_sample_nodes = ['iGs9GbCjDHYxyCA5GVAvCw', 'e2YB6r-vi0hHh7-SFf9EwQ', 'IkJDxkMuFRFVttUiVnDXmQ', 'svLFY6VqyxhE176e9bs7_Q', '-w9PuVSzBpvr9686KCRBmA', 'kIblee8PR-r8RIGlkuf5_A', 'BNC9P3QZ9zsHu6awAryesQ', '_16ryuaGVkBuZuIs0xP9tw', 'evFWy5O3aG1uNmionpgG5A', 'QwW7loomahEEAp-w8scOVQ', 'YGH0AiIQS_LD2vzYw3doqA', 'p30h_D9S8UOAg6kGQW9gjg', 'O_-cRxYptp4H0Fji6_ytgA', 'YUNReDWIxgpdgyU9Dmt-oA', 'lXAHLXGJGSnBXMvDnMag-A', 'YYlIgQM9W7jz5PDXgjy9dA', '6tB0N_eBYisWtbLOWUM5Pw', 'b7D7LlvE4tgg-EXCrDi7gg', 'KMxlLltZmoTabwzsjRRbYA', 'FKSvhqps-IKP495Y16aCTw', '2HDpZkWfwWsJc3DlbSKMVA', 'HtN8R31A-CHHb_GuaG201A', '0yp3Zq7DRgmgA3Z3KKFLXw', 'MpfSfj93ezyUfGzMGnP9ZQ', 'HjrWL4_FoevD0yHWQdeyXQ', 'lv2KOgLON5D8j7YWOs8CFw', 'eKP6iyAFH0ksl9qo8tthjg', 'KYL3LpicQM9XVTn1re-5VQ', 'M6X9EWc1NC4YvmXMr01ndw', 'uHh7ciaH24btVsCMbMHesQ', 'vKZBLoTff5gOxXf0LpWfLw', 'GYm1bPsZ-kpSo3A7jo4bhQ', 'QEkBQBiZjBoOj1v4-G84mQ', 'rAVqBqcQGFH0gZQImqcLKw', '71_UIWSTmIRdk7mga8c4KA', '7n7HIzz4mT9WgwB-wZuUzA', 'JLAIiOswFKqXuNrnS2S50Q', 'rEahV1J0V1gilT4FjU7fHg', 'v7nbdtw3df0mjL7U1qdeQg', 'fApGpLljXfEb0QU1FEI-ow', 'Kziof1CjHQDcdhW0Mqaiuw', '35UF6qRu38bx3I16DAHfVw', 'l6Z9HSlEKvdGO1t1w-yoKQ', 'zNlne25imF1iE4FlsBhliA', 'CTy3L7ahKPbxn7o--snc7g', 'LdxSoc2sSU99tk1knz_KjQ', 'xFwWb9Mw5CgXKivPB69ZoA', 'QDLVx52O1dttqyoL2-3Rhg', 'mllbZfJUUpbl9T8ydgWVOw', 'Xs3NRLPxNsFqJ_o9KxxeQA', 'UVukxGAGLdMXUJbGjhRKKg', 'MDeJ3nXol5fimFTtYApGXA', 'gODwieyUjEQcZHiiiWDCUQ', 'NEiwNYwjZqJ0_M2biJRsyQ', '720wi5oglxFzf13OG8n54g', 'gaJSZC8WqcqOaDt1Q8cOvA', '0nu1ia43gWDRNJ3AHkapVg', 'f3fmCT-vpPvLM_CezJlbEg', 'eaUT_iiQ2e7hdyws6YAxEw', 'soW6EIyebWB-hFkDh4Q0Ew', 'w5NOTxFafQBq9wjUxoCyVw', 'ziXVk3-ekgTdWLvMLHsghQ', '9wRCzzewwRVZuCdzlUj48g', 'i5CHCy2z3gNoQC6EaoOmfQ', 'qOm6LGtSRQMUEDKwH8DcxQ', 'qTS1a6RVH-3pr5tVMoXfzw', 'Fw5b2N4rky8X0W-MsVolCQ', 'Rs0is9mQQYgoXCj2ODgwFw', 'LDHVh4spgrrA9afkzI0iTg', 'p_o-cYnr2mgLpNSn__fcww', 'yAySW8kY02xxORU6JgPvew', 'TdMzPadwk5rsHNVUV1fHAQ', 'ciQKlJ435Fl9e6C35uZnkg', 'jQzHlwDqJQdk8OBDWfOf2g', 'ciZ_Oqe3OualNiJT56FPoQ', '0SzxfS3xy_-gFp7kIohzpg', 'ib0XnNmHBtf8c5sFiydXxA', 'GR5sqfsQW7Xb-muiFe5bWg', 'wmN3Nz-plsZ_mRhRhgqKNg', 'aXLy9O5fLV3LvqdBinKWmQ', 'tTs7XMw6NyJRxQb2E95VHg', 'c8z-H0VRYK4q02Au5HJThw', 'dE9ZRsxSjbIECJ1NyalLgQ', 'CIBz0WQIJCCa59B-54c-rA', 'qEM5O1ToJCwZLCACgKgsHQ', 'sv9Dblc50z7h1anc_tIkfg', 'DidrTfuOcsD5zua88ApqSw', 'bhDgLU2sgUXPkLWeN603iQ', 'hTuyyrATOKzB8iYYpCalZg', '6m0MNEJCMxPIYEPJJs4vcA', 'm_YXf6qeF49r4p9RahcE5g', 'bpuRkrQPbltotnnp5eNHPQ', 'XdDJ156CJMS5Tg1zdgB0HA', 'YYBAHJuQLl4wTNIW1GlnDA', 'Kqfbdk9azTHpiLJfeGUg-w', '1_8zm3DVhYN9q39DxNpV9w', 'hENmlHe6ak4QDDF1pR8bdg', 'QEzQida-_XILQMLHXtfZhw', 'IkC3HKhwTXpwXrYgB5XXvQ', 'iYy_SHjYNYuDkaEho7hE1A', '-P4GiKPqqvJ1W2EFKyZ62g', 'AnUnGNqvUUbxSIStoVG47A', 'I-cd-erq7Si7AU1jMvHf2Q', 'qVsIVg5jWR95CFtKu31LSw', 'PgZqHcOqiglWNwtEQnUsoQ', 'fyFrp0i9aaZ5Yk6_q66WYg', 'X8t60gunxi1hITAzcZJALQ', 'bKAuipSnS1P4R9_6q-srOQ', 'c5syeYy9kOufcXflybA0PQ', 'ykNUyxrZ1vtZantWOIwXBw', 'bjC4hjeh6WM7FQzDgurtVw', 'LQpHhlT8ysBbhaWHmrDFrg', '5S0k-fjh0D0OL8h10BNsiA', 'EOhxNvLzkpITKIAufo042Q', '5V7rNoNIF6w8ti_utUQIxA', '9sYD7W06JHVztcky2hnPMg', 'KSv7n6qROPf1CI8C26DpUg', 'IzVsYcljRhmr4_ZZJQM1vA', 'H94wWVfATBAOfAHezRv8zw', 'VcrNdZ1Kh6-YTMi8Vac-KA', 'Zl63Mpnhqg1qbFq58lJnTg', 'eqDxnh--JA8GG-arQQ8p0Q', '6k8qh7dLsPsrAt88ssZopw', 'C7A_tolSQIM8-6CzQVutwA', 'IWsCPsxhSg_rjn3Em6_FkQ', 'uUpy7A6eqO_sSasWAQmCdw', 'KQRHWMq3P0BYhFEvWIXCWw', 'O66VdCaZzzsufWUHjVL4lQ', 'fC9vvp4iaexVXpD-6nSHpg', 'bqJ57uAdB5hrZGnONN8I5g', 'NjO1IkogsNqrvI3FXULjnQ', 'yXB3-l0MQ3qPlPmE3XJyeA', 'xt49GdpKS5900I9QvYEiKw', 'e_Il_o2HSw_5DYMTC3JcOQ', 'j6W8HFk1KsLoJ8r7mpXgrg']


def sample_function():
    full_fsgraph = nx.read_gml(g_fsgraph_full_gml_format.format(time_int_str, time_int_str))
    sample_fsgraph = nx.subgraph(full_fsgraph, g_sample_nodes)
    # nx.write_gml(sample_fsgraph, g_fsgraph_top_sample_format.format(time_int_str, time_int_str))

    d_user_text = dict()
    with open(g_time_series_data_format.format(time_int_str, time_int_str), 'r') as in_fd:
        t_nec_str = in_fd.readline()
        while t_nec_str:
            t_nec_json = json.loads(t_nec_str)
            uid = t_nec_json['uid']
            l_text = t_nec_json['raw_cln']
            if uid not in g_sample_nodes:
                t_nec_str = in_fd.readline()
                continue
            if uid not in d_user_text:
                d_user_text[uid] = l_text
            else:
                d_user_text[uid] += l_text
            t_nec_str = in_fd.readline()
        in_fd.close()

    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(g_sent2vec_model_file_path)

    d_user_text_vec = dict()
    for uid in d_user_text:
        text_vec = get_one_doc_vect(sent2vec_model, d_user_text[uid])
        d_user_text_vec[uid] = text_vec

    sample_pure_tsgraph = nx.Graph()
    for node_1 in g_sample_nodes:
        for node_2 in g_sample_nodes:
            if node_2 != node_1 and not sample_pure_tsgraph.has_edge(node_1, node_2):
                if not node_1 in d_user_text_vec or not node_2 in d_user_text_vec:
                    sim = 0.0
                else:
                    sim = 1.0 - scipyd.cosine(d_user_text_vec[node_1], d_user_text_vec[node_2])
                if sim >= g_sim_threshold:
                    sample_pure_tsgraph.add_edge(node_1, node_2, weight=(sim + 1.0) / 2.0)
                # pure_tsgraph.add_edge(node_1, node_2, weight=(sim + 1.0) / 2.0)
    # nx.write_gml(sample_pure_tsgraph, g_tsgraph_top_sample_gml_format.format(time_int_str, time_int_str))

    mi = compute_mutual_info(sample_fsgraph, sample_pure_tsgraph)
    print(mi)


def get_one_doc_vect(sent2vec_model, l_sents):
    if len(l_sents) == 0:
        return [0.0 for i in range(0, 700)]
    l_sents = [sent.lower() for sent in l_sents]
    embs = sent2vec_model.embed_sentences(l_sents, mp.cpu_count())
    doc_vect = [sum(x)/len(l_sents) for x in zip(*embs)]
    return doc_vect


def compute_mutual_info(normfsgraph, normtsgraph):
    d_edges = dict()
    for edge in normtsgraph.edges.data('weight'):
        none_mark_1 = False
        none_mark_2 = False
        try:
            if normfsgraph.has_edge(edge[0], edge[1]):
                fs_weight = normfsgraph.edges[edge[0], edge[1]]['weight']
            else:
                none_mark_1 = True
            if normfsgraph.has_edge(edge[1], edge[0]):
                if none_mark_1:
                    fs_weight = normfsgraph.edges[edge[0], edge[1]]['weight']
                else:
                    fs_weight += normfsgraph.edges[edge[1], edge[0]]['weight']
            else:
                none_mark_2 = True
        except:
            none_mark_1 = True
            none_mark_2 = True
        if not none_mark_1 or not none_mark_2:
            d_edges[(edge[0], edge[1])] = [edge[2], fs_weight]
    l_fs_weights = []
    l_ts_weights = []
    for item in d_edges.values():
        l_fs_weights.append(item[0])
        l_ts_weights.append(item[1])
    # l_fs_weights = [item[0] for item in list(d_edges.values())]
    # l_ts_weights = [item[1] for item in list(d_edges.values())]
    mi = metrics.normalized_mutual_info_score(l_fs_weights, l_ts_weights, average_method='arithmetic')
    return mi


def main():
    d_user_text = dict()
    fsgraph = nx.read_gml(g_fsgraph_top_gml_format.format(time_int_str, time_int_str))
    with open(g_time_series_data_format.format(time_int_str, time_int_str), 'r') as in_fd:
        t_nec_str = in_fd.readline()
        while t_nec_str:
            t_nec_json = json.loads(t_nec_str)
            uid = t_nec_json['uid']
            l_text = t_nec_json['raw_cln']
            if uid not in fsgraph.nodes:
                t_nec_str = in_fd.readline()
                continue
            if uid not in d_user_text:
                d_user_text[uid] = l_text
            else:
                d_user_text[uid] += l_text
            t_nec_str = in_fd.readline()
        in_fd.close()

    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(g_sent2vec_model_file_path)

    d_user_text_vec = dict()
    for uid in d_user_text:
        text_vec = get_one_doc_vect(sent2vec_model, d_user_text[uid])
        d_user_text_vec[uid] = text_vec

    tsgraph = nx.Graph()
    for edge in fsgraph.edges:
        if not tsgraph.has_edge(edge[0], edge[1]):
            if not edge[0] in d_user_text_vec or not edge[1] in d_user_text_vec:
                sim = 0.0
            else:
                sim = 1.0 - scipyd.cosine(d_user_text_vec[edge[0]], d_user_text_vec[edge[1]])
            if sim >= g_sim_threshold:
                tsgraph.add_edge(edge[0], edge[1], weight=(sim+1.0)/2.0)
            # tsgraph.add_edge(edge[0], edge[1], weight=(sim + 1.0) / 2.0)
    # nx.write_gml(tsgraph, g_tsgraph_full_gml_format.format(time_int_str, time_int_str))

    mi = compute_mutual_info(fsgraph, tsgraph)
    print(mi)

    pure_tsgraph = nx.Graph()
    for node_1 in fsgraph.nodes:
        for node_2 in fsgraph.nodes:
            if node_2 != node_1 and not pure_tsgraph.has_edge(node_1, node_2):
                if not node_1 in d_user_text_vec or not node_2 in d_user_text_vec:
                    sim = 0.0
                else:
                    sim = 1.0 - scipyd.cosine(d_user_text_vec[node_1], d_user_text_vec[node_2])
                if sim >= g_sim_threshold:
                    pure_tsgraph.add_edge(node_1, node_2, weight=(sim + 1.0) / 2.0)
                # pure_tsgraph.add_edge(node_1, node_2, weight=(sim + 1.0) / 2.0)
    nx.write_gml(pure_tsgraph, g_tsgraph_top_pure_gml_format.format(time_int_str, time_int_str))

    mi = compute_mutual_info(fsgraph, pure_tsgraph)
    print(mi)

if __name__ == '__main__':
    # main()
    sample_function()