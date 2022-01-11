'''
OBJECTIVES:
    In this file, all necessary global variables are defined. All modules should import this module to access global
    settings. Try to avoid to define any unnecessary global variables in other modules.
'''


'''INPUTS'''
# g_tw_version = 'v0221'
# g_tw_version = 'v0307_0724'
# g_tw_version = 'v2-1'
# g_tw_version = 'v3'
# g_tw_version = 'v0729'
# g_tw_version = 'v0214'
g_tw_version = 'v0404'

# TODO
# Modify 'g_tw_work_folder'
# g_tw_work_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
g_tw_work_folder = '/scratch/mf3jh/data/cp4_ven_tw/'
g_tw_raw_data_folder = g_tw_work_folder + 'tw_raw_data/'
g_tw_raw_data_file_list = [
                           'collection1_2019-02-08_2019-02-14_twitter_raw_data.json',
                           'collection1_2019-02-01_2019-02-07_twitter_raw_data.json',
                           'collection1_2019-02-15_2019-02-21_twitter_raw_data.json',
                           'cp4.ven.ci2.twitter.v2.2019-02-15_2019-02-21.json',
                           'cp4.ven.ci2.twitter.v2.2019-02-22_2019-02-28.json',
                           'cp4.ven.ci2.twitter.v2.2019-03-01_2019-03-07.json',
                           'cp4.ven.ci2.twitter.v2.2019-03-08_2019-03-14.json',
                           'cp4.ven.ci2.twitter.v2.2019-03-15_2019-03-21.json',
                           'cp4.ven.ci2.twitter.v2.2019-03-22_2019-04-04.json',
                           # 'cp4.venezuela.twitter.training.anon.v3.2018-12-24_2018-12-31.json',
                           # 'cp4.venezuela.twitter.training.anon.v3.2019-01-01_2019-01-14.json',
                           # 'cp4.venezuela.twitter.training.anon.v3.2019-01-15_2019-01-20.json',
                           # 'cp4.venezuela.twitter.training.anon.v3.2019-01-21_2019-01-24.json',
                           # 'cp4.venezuela.twitter.training.anon.v3.2019-01-25_2019-01-31.json',
                           # 'cp4.venezuela.twitter.training.anon.v3.reply-cascades.json'
                          ]

# g_tw_raw_data_folder = g_tw_work_folder + '714new/'
# g_tw_raw_data_file_list = ['collection1_2019-02-01_2019-02-07_twitter_raw_data.json']

g_events_folder = g_tw_work_folder + 'ven_events/'
g_events_raw_texts_folder = g_events_folder + 'raw_texts/'


'''GENERATED'''
g_tw_en_db = g_tw_work_folder + 'tw_en_{0}.db'.format(g_tw_version)
g_tw_raw_data_int_folder = g_tw_raw_data_folder + 'ven_tw_en_int/'
g_tw_raw_data_int_file_format = g_tw_raw_data_int_folder + 'ven_tw_en_int_{0}.pickle'
g_tw_raw_data_int_tw_ids_file_format = g_tw_raw_data_int_folder + 'tw_ids_{0}.txt'
g_tw_raw_data_clean_txt_int_file_format = g_tw_raw_data_int_folder + 'clean_txt_int_{0}.pickle'

g_udt_folder = g_tw_work_folder + 'udt/'
g_udt_int_folder = g_udt_folder + 'udt_int/'
g_udt_community_folder = g_udt_folder + 'communities/'
g_udt_tw_raw_data_int_file_format = g_udt_int_folder + 'udt_ven_tw_int_{0}.pickle'
g_udt_tw_data_file = g_udt_folder + 'udt_tw_data_{0}.pickle'.format(g_tw_version)
g_udt_usr_data_task_file_format = g_udt_int_folder + 'udt_usr_data_task_int_' + g_tw_version + '_{0}.pickle'
g_udt_usr_data_int_file_format = g_udt_int_folder + 'udt_usr_data_int_' + g_tw_version + '_{0}.pickle'
g_udt_usr_data_file = g_udt_folder + 'udt_usr_data_{0}.pickle'.format(g_tw_version)
g_udt_community_file_list = ['Community_1404_Nodes.csv', 'Community_151_Nodes.csv']
g_udt_com_data_file_format = g_udt_folder + 'udt_com_{0}_' + g_tw_version + '.pickle'
g_udt_tw_srg_trg_data_int_folder = g_udt_folder + 'udt_tw_src_trg_int/'
g_udt_tw_srg_trg_data_int_file_format = g_udt_tw_srg_trg_data_int_folder + 'udt_tw_src_trg_data_int_' + g_tw_version + '_{0}.pickle'
g_udt_tw_src_trg_data_file = g_udt_folder + 'udt_tw_src_trg_data_' + g_tw_version + '.pickle'
g_udt_tw_src_trg_sna_com_id_file = g_udt_folder + 'udt_tw_src_trg_sna_id_' + g_tw_version + '.pickle'
g_udt_tw_src_trg_sna_data_file_format = g_udt_folder + 'udt_tw_src_trg_sna_data_{0}#{1}.pickle'

g_tw_sem_units_folder = g_tw_work_folder + 'sem_units/'
g_tw_sem_units_int_folder = g_tw_sem_units_folder + 'sem_units_int/'
g_tw_sem_units_db = g_tw_sem_units_folder + 'tw_en_sem_units_{0}.db'.format(g_tw_version)
g_tw_sem_unit_int_file_format = g_tw_sem_units_int_folder + 'sem_units_int_{0}.pickle'

g_tw_embed_folder = g_tw_work_folder + 'tw_embeds/'
g_tw_embed_task_file_format = g_tw_work_folder + 'tw_embed_job_{0}.txt'
g_tw_embed_db = g_tw_embed_folder + 'tw_en_{0}_embeds.db'.format(g_tw_version)
g_tw_embed_usr_avg_embeds_int_folder = g_tw_embed_folder + 'usr_avg_embeds/'
g_tw_embed_usr_avg_embeds_alt_task_file_format = g_tw_embed_usr_avg_embeds_int_folder + 'usr_avg_embed_alt_task_{0}'
g_tw_embed_usr_avg_embeds_int_file_format = g_tw_embed_usr_avg_embeds_int_folder + 'usr_avg_embeds_int_{0}.json'
g_tw_embed_usr_avg_embeds_output = g_tw_embed_usr_avg_embeds_int_folder + 'usr_avg_embeds_all.json'
g_tw_embed_usr_avg_embeds_activated_usrs_only_output = g_tw_embed_usr_avg_embeds_int_folder + 'usr_avg_embeds_act.json'

g_tw_phrase_graph_folder = g_tw_work_folder + 'tw_phrase_graph/'
g_tw_phrase_extraction_int_folder = g_tw_phrase_graph_folder + 'raw_phrases_int/'
g_tw_phrase_extraction_tw_to_phrases_int_file_format = g_tw_phrase_extraction_int_folder + 'tw_to_phrases_int_{0}.pickle'
g_tw_phrase_extraction_tw_to_phrases_file = g_tw_phrase_graph_folder + 'tw_to_phrases_{0}.pickle'.format(g_tw_version)
g_tw_phrase_extraction_tw_to_phids_file = g_tw_phrase_graph_folder + 'tw_to_phids_{0}.pickle'.format(g_tw_version)
g_tw_raw_phrases_phrase_to_id = g_tw_phrase_graph_folder + 'phrase_to_id_{0}.json'.format(g_tw_version)
g_tw_raw_phrases_id_to_phrase = g_tw_phrase_graph_folder + 'id_to_phrase_{0}.json'.format(g_tw_version)

g_tw_phrase_job_file_format = g_tw_phrase_graph_folder + 'tw_raw_phrase_job_{0}.txt'
g_tw_raw_phrases_file_format = g_tw_phrase_graph_folder + 'tw_raw_phrases_{0}.txt'
g_tw_raw_phrases_output = g_tw_phrase_graph_folder + 'tw_raw_phrases_all.txt'
g_tw_raw_phrases_unique = g_tw_phrase_graph_folder + 'tw_raw_phrases_unique_all.json'

g_tw_raw_phrases_pos_stat = g_tw_phrase_graph_folder + 'tw_raw_phrases_pos_stat.txt'
g_tw_raw_phrases_pos_mapping = g_tw_phrase_graph_folder + 'tw_raw_phrases_pos_mapping.txt'
g_tw_raw_phrases_output_by_pos = g_tw_phrase_graph_folder + 'tw_raw_phrases_all_by_pos.json'
g_tw_raw_phrases_vocab_file = g_tw_phrase_graph_folder + 'tw_raw_phrases_vocab.txt'
g_tw_raw_phrases_ae_training_sets_folder = g_tw_phrase_graph_folder + 'phrase_ae_train_sets/'
g_tw_raw_phrases_ae_training_sets_file_format = g_tw_raw_phrases_ae_training_sets_folder + 'ae_train_set_{0}.txt'
g_tw_raw_phrases_token_ae_training_sets_folder = g_tw_phrase_graph_folder + 'token_ae_train_sets/'
g_tw_raw_phrases_token_ae_training_sets_file_format = g_tw_raw_phrases_token_ae_training_sets_folder + 'token_ae_train_set_{0}.txt'

g_tw_raw_phrases_embeds_int_folder = g_tw_phrase_graph_folder + 'phrase_embeds_int/'
g_tw_raw_phrases_embeds_int_format = g_tw_raw_phrases_embeds_int_folder + 'phrase_embeds_int_{0}.pickle'
g_tw_raw_phrases_embeds = g_tw_phrase_graph_folder + 'phrase_embeds_all_{0}.pickle'.format(g_tw_version)

g_tw_raw_phrases_embeds_int_npy_format = g_tw_raw_phrases_embeds_int_folder + 'raw_phrase_embeds_{0}.npy'
g_tw_raw_phrases_embeds_npy_rev_phrase_idx = g_tw_raw_phrases_embeds_int_folder + 'raw_phrase_embeds_npy_rev_phrase_idx.json'
# g_tw_raw_phrases_embeds_npy = g_tw_raw_phrases_embeds_int_folder + 'raw_phrase_embeds_all.npy'

g_tw_raw_phrases_phrase_cluster_folder = g_tw_phrase_graph_folder + 'phrase_cluster/'
g_tw_raw_phrases_embeds_for_clustering = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_embeds_for_clustering_{0}.npy'.format(g_tw_version)
g_tw_raw_phrases_clustering_labels_format = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_cluster_labels_' + g_tw_version + '_{0}.txt'
g_tw_raw_phrases_clustering_info_format = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_cluster_info_' + g_tw_version + '_{0}.pickle'
g_tw_raw_phrases_clustering_center_sim_heatmap_format = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_cluster_center_sim_heatmap_' + g_tw_version + '_{0}.png'
g_tw_raw_phrases_clustering_member_to_center_sim_fig_format = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_cluster_member_to_center_sim_' + g_tw_version + '_{0}.png'

g_tw_phrase_cluster_embeds_int_folder = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_cluster_embeds_int/'
g_tw_phrase_cluster_embeds_tasks_file_format = g_tw_phrase_cluster_embeds_int_folder + 'phrase_cluster_embeds_tasks_' + g_tw_version + '_{0}.pickle'
g_tw_phrase_cluster_embeds_int_file_format = g_tw_phrase_cluster_embeds_int_folder + 'phrase_cluster_embeds_int_' + g_tw_version + '_{0}.pickle'
g_tw_phrase_cluster_embeds_all_file_format = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_cluster_embeds_all_' + g_tw_version + '_{0}.pickle'
g_tw_phrase_cluster_onehot_embeds_all_file_format = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_cluster_onehot_embeds_all_' + g_tw_version + '_{0}.pickle'

g_tw_pc_onehot_folder = g_tw_phrase_graph_folder + 'tw_pc_onehot/'
g_tw_pc_onehot_int_folder = g_tw_pc_onehot_folder + 'tw_pc_onehot_int/'
g_tw_pc_onehot_task_file_format = g_tw_pc_onehot_int_folder + 'tw_pc_onehot_task_' + g_tw_version + '_{0}.pickle'
g_tw_pc_onehot_int_file_format = g_tw_pc_onehot_int_folder + 'tw_pc_onehot_int_' + g_tw_version + '_{0}.pickle'
g_tw_pc_onehot_file_format = g_tw_pc_onehot_folder + 'tw_pc_onehot_' + g_tw_version + '_{0}.pickle'

g_tw_response_graph_folder = g_tw_phrase_graph_folder + 'resp_graph/'
g_tw_src_trg_data_file_format = g_tw_response_graph_folder + 'src_trg_tw_data_' + g_tw_version + '_{0}.pickle'
g_tw_response_graph_int_folder = g_tw_response_graph_folder + 'resp_graph_int/'
g_tw_response_graph_task_file_format = g_tw_response_graph_int_folder + 'resp_graph_task_' + g_tw_version + '_{0}.pickle'
g_tw_response_graph_int_file_format = g_tw_response_graph_int_folder + 'resp_graph_int_' + g_tw_version + '_{0}.npy'
g_tw_response_graph_transition_matrix_format = g_tw_response_graph_folder + 'resp_graph_trans_mat_' + g_tw_version + '_{0}.npy'
g_tw_response_graph_transition_matrix_fig_format = g_tw_response_graph_folder + 'resp_graph_trans_mat_' + g_tw_version + '_{0}.png'
g_tw_src_trg_comprehensive_data_file_format = g_tw_response_graph_folder + 'src_trg_combo_tw_data_' + g_tw_version + '_{0}.pickle'

g_tw_usr_beliefs_folder = g_tw_phrase_graph_folder + 'usr_beliefs/'
g_tw_usr_beliefs_train_set_folder = g_tw_usr_beliefs_folder + 'train_sets/'
g_tw_usr_beliefs_train_set_int_folder = g_tw_usr_beliefs_train_set_folder + 'train_set_int/'
g_tw_usr_beliefs_train_set_int_file_format = g_tw_usr_beliefs_train_set_int_folder + 'usr_beliefs_train_set_int_' + g_tw_version + '_{0}.pickle'
g_tw_usr_beliefs_train_set_file_format = g_tw_usr_beliefs_train_set_folder + 'usr_beliefs_train_set_' + g_tw_version + '_{0}.pickle'
g_tw_usr_beliefs_saved_models_folder = g_tw_usr_beliefs_folder + 'saved_models/'
g_tw_usr_beliefs_model_checkpoint_format = g_tw_usr_beliefs_saved_models_folder + 'usr_beliefs_model_{0}.ckpt'

g_tw_new_tw_folder = g_tw_work_folder + 'new_tw/'
g_tw_new_tw_data = g_tw_new_tw_folder + 'new_tw_data_' + g_tw_version + '.pickle'
g_tw_new_tw_data_time_interval_folder = g_tw_new_tw_folder + 'time_ints/'
g_tw_new_tw_data_time_int_file_format = g_tw_new_tw_data_time_interval_folder + 'com_{0}_new_tw_data_' + g_tw_version + '_{1}.pickle'
g_tw_new_tw_time_int_pc_dists_int_folder = g_tw_new_tw_folder + 'time_int_pc_dists_ints/'
g_tw_new_tw_time_int_pc_dists_int_file_format = g_tw_new_tw_time_int_pc_dists_int_folder + 'new_tw_time_int_pc_dists_int_' + g_tw_version + '_{0}.pickle'
g_tw_new_tw_time_int_embeds_int_file_format = g_tw_new_tw_time_int_pc_dists_int_folder + 'new_tw_time_int_embeds_int_' + g_tw_version + '_{0}.pickle'
g_tw_new_tw_time_int_pc_dists_and_embeds_int_file_format = g_tw_new_tw_time_int_pc_dists_int_folder + 'new_tw_time_int_pc_dists_and_embeds_int_' + g_tw_version + '_{0}.json'
g_tw_new_tw_time_int_pc_dists_and_embeds_file_format = g_tw_new_tw_folder + 'new_tw_time_int_pc_dists_and_embeds_' + g_tw_version + '_{0}.pickle'
g_tw_new_tw_time_int_embeds_file_format = g_tw_new_tw_folder + 'new_tw_time_int_embeds_' + g_tw_version + '_{0}.pickle'
g_tw_new_tw_by_com_file = g_tw_new_tw_folder + 'new_tw_by_com_' + g_tw_version + '.pickle'
g_tw_new_tw_by_com_stats_file = g_tw_new_tw_folder + 'new_tw_by_com_stats_' + g_tw_version + '.json'
g_tw_community_usr_cnts_file = g_tw_new_tw_folder + 'sna_com_usr_cnts.json'
g_tw_new_tw_time_series_data_by_com_file_format = g_tw_new_tw_folder + 'new_tw_ts_data_by_com_{0}.pickle'
g_tw_new_tw_time_series_pconehot_nar_stance_by_com_file_format = g_tw_new_tw_folder + 'new_tw_ts_pconehot_nar_stance_by_com_{0}.pickle'
g_tw_new_tw_model_folder = g_tw_new_tw_folder + 'new_tw_models/'
g_tw_new_tw_model_train_sets = g_tw_new_tw_model_folder + 'new_tw_model_train_sets.pickle'

g_events_clean_texts_file_format = g_events_folder + 'ven_events_clean_texts_{0}.pickle'
g_events_sents_file_format = g_events_folder + 'ven_events_sents_{0}.pickle'
g_events_sent_to_phs_file_format = g_events_folder + 'ven_events_sent_to_phs_{0}.pickle'
g_events_sent_to_phids_file_format = g_events_folder + 'ven_events_sent_to_phids_{0}.pickle'
g_events_sent_to_pcvec_file_format = g_events_folder + 'ven_events_sent_to_pcvec_{0}.pickle'
g_events_sem_units_folder = g_events_folder + 'sem_units/'
g_events_sem_units_file_format = g_events_sem_units_folder + 'su_{0}.pickle'
g_events_ph_to_id_file_format = g_events_folder + 'ven_events_ph_to_id_{0}.json'
g_events_id_to_ph_file_format = g_events_folder + 'ven_events_id_to_ph_{0}.json'
g_events_phid_to_embed_file_format = g_events_folder + 'ven_events_phid_to_embed_{0}.pickle'
g_events_phid_to_pcid_file_format = g_events_folder + 'ven_events_phid_to_pcid_{0}.pickle'
g_events_sents_csv_file_format = g_events_folder + 'ven_events_sents_{0}.csv'
g_events_sents_nar_csv_file_format = g_events_folder + 'ven_events_sents_nar_{0}.csv'
g_events_sents_stance_csv_file_format = g_events_folder + 'ven_events_sents_stance_{0}.csv'
g_events_info_file_format = g_events_folder + 'ven_events_info_{0}.pickle'

g_community_analysis_folder = g_tw_work_folder + 'community_analysis/'
g_new_tw_ts_pw_com_comparison_data_format = g_community_analysis_folder + 'new_tw_ts_com_pw_comparisons_{0}.pickle'

g_microsim_folder = g_tw_work_folder + 'microsim/'
g_microsim_msg_prop_graphs_folder = g_microsim_folder + 'msg_prop_graphs/'
g_microsim_msg_prop_graphs_file_format = g_microsim_msg_prop_graphs_folder + 'msg_prop_graph_com_{0}_src_{1}.json'
g_microsim_msg_prop_graph_data_folder = g_microsim_folder + 'msg_prop_graph_data/'
g_microsim_msg_prop_graph_data_by_com_file_format = g_microsim_msg_prop_graph_data_folder + 'msg_prop_graph_data_com_{0}.pickle'
g_microsim_sim_data_folder = g_microsim_folder + 'sim_data/'
g_microsim_results_folder = g_microsim_folder + 'sim_rets/'
g_microsim_results_by_com_folder_format = g_microsim_results_folder + 'sim_rets_{0}/'
g_microsim_seeds_by_com_folder_format = g_microsim_results_folder + 'seeds_{0}/'
g_microsim_sim_graphs_folder = g_microsim_folder + 'sim_graphs/'
g_microsim_sim_graphs_file_format = g_microsim_sim_graphs_folder + 'sim_graph_com_{0}_src_{1}_seed_{2}.json'
g_microsim_sim_graph_data_folder = g_microsim_folder + 'sim_graph_data/'
g_microsim_sim_graph_data_by_com_file_format = g_microsim_sim_graph_data_folder + 'sim_graph_data_com_{0}.pickle'
g_microsim_analysis_results_folder = g_microsim_folder + 'ana_rets/'
g_microsim_analysis_sim_vs_gt_by_com_file_format = g_microsim_analysis_results_folder + 'sim_vs_gt_com_{0}.pickle'
g_microsim_ana_sum_sim_vs_gt_by_com_folder_format = g_microsim_analysis_results_folder + 'sim_vs_gt_{0}/'
g_microsim_ana_sum_sim_alone_by_com_folder_format = g_microsim_analysis_results_folder + 'sim_{0}/'
g_microsim_ana_sum_gt_alone_by_com_folder_format = g_microsim_analysis_results_folder + 'gt_{0}/'
g_microsim_ana_sum_gt_sim_alone_table_file_by_com_file_format = g_microsim_analysis_results_folder + 'gt_sim_alone_{0}.csv'
g_microsim_ana_sum_sim_vs_gt_table_file_by_com_file_format = g_microsim_analysis_results_folder + 'sim_vs_gt_{0}.csv'
g_microsim_ana_simperiod_folder = g_microsim_folder + 'simperiod_rets/'
g_microsim_ana_simperiod_sum_file = g_microsim_ana_simperiod_folder + 'simperiod_sum.pickle'

######################################################################
g_tw_raw_phrases_clustering_member_to_center_sims_format = g_tw_raw_phrases_phrase_cluster_folder + 'phrase_cluster_member_to_center_sims_' + g_tw_version + '_{0}.json'
g_tw_raw_phrases_clustering_groups_format = g_tw_phrase_graph_folder + 'phrase_cluster_groups_' + g_tw_version + '_{0}.json'
g_tw_raw_phrases_weid2phid = g_tw_phrase_graph_folder + 'raw_phrases_weid2phid.txt'

g_tw_raw_phrases_cluster_space_embeds_int_format = g_tw_raw_phrases_embeds_int_folder + 'raw_phrase_cluster_space_embeds_{0}_{1}.npy'
g_tw_raw_phrases_cluster_space_embeds_format = g_tw_phrase_graph_folder + 'raw_phrase_cluster_space_embeds_{0}.npy'
g_tw_raw_phrases_cluster_space_embeds_dim_format = g_tw_phrase_graph_folder + 'raw_phrase_cluster_space_embeds_dim_{0}.txt'

g_tw_src_trg_tw_id_pairs = g_tw_phrase_graph_folder + 'tw_src_to_trg_tw_id_pairs.json'
g_tw_trg_src_tw_id_pairs = g_tw_phrase_graph_folder + 'tw_trg_to_src_tw_id_pairs.json'
g_tw_replies_quotes_per_user = g_tw_phrase_graph_folder + 'tw_replies_quotes_per_usr.json'
g_tw_phrases_by_idx = g_tw_phrase_graph_folder + 'tw_phrases_by_idx.json'
g_tw_phrases_by_clusters_format = g_tw_phrase_graph_folder + 'tw_phrases_by_clusters_{0}.json'
# g_tw_phrases_response_graph_task_format = g_tw_phrases_response_graph_int_folder + 'resp_graph_task_{0}.txt'
# g_tw_phrases_response_graph_int_format = g_tw_phrases_response_graph_int_folder + 'resp_graph_int_{0}.npy'
g_tw_phrases_response_graph_format = g_tw_phrase_graph_folder + 'tw_phrases_resp_graph_{0}.json'
g_tw_phrases_response_graph_transition_mat_format = g_tw_phrase_graph_folder + 'tw_phrases_resp_graph_trans_mat_{0}.npy'


# g_tw_usr_beliefs_train_set_format = g_tw_usr_beliefs_folder + 'usr_beliefs_train_set_{0}.pickle'
# g_tw_usr_beliefs_train_set_format = g_tw_usr_beliefs_folder + 'usr_beliefs_train_set_by_usr_{0}.json'
g_tw_usr_beliefs_usr_stats = g_tw_usr_beliefs_folder + 'usr_beliefs_usr_stats.txt'
# g_tw_usr_beliefs_model_checkpoint_format = g_tw_usr_beliefs_folder + 'usr_beliefs_model_{0}.ckpt'

g_tw_all_usr_ids_file = g_tw_work_folder + 'all_usr_ids.txt'
g_tw_ae_narrative_rep_train_sets_folder = g_tw_work_folder + 'ae_nar_train_sets/'
g_tw_activated_usr_ids_file = g_tw_ae_narrative_rep_train_sets_folder + 'activated_usr_ids.txt'
# g_tw_narrative_to_code_file = g_tw_ae_narrative_rep_train_sets_folder + 'tw_nar_to_code_{0}.json'.format(g_tw_version)
# g_tw_code_to_narrative_file = g_tw_ae_narrative_rep_train_sets_folder + 'tw_code_to_nar_{0}.json'.format(g_tw_version)
g_tw_narrative_to_code_file = g_tw_ae_narrative_rep_train_sets_folder + 'tw_nar_to_code_all.json'
g_tw_code_to_narrative_file = g_tw_ae_narrative_rep_train_sets_folder + 'tw_code_to_nar_all.json'
g_tw_ae_narrative_rep_train_sets_file_format = g_tw_ae_narrative_rep_train_sets_folder + 'ae_nar_rep_train_set_{0}.pickle'
g_tw_ae_narrative_rep_train_subset_by_com_id_file_format = g_tw_ae_narrative_rep_train_sets_folder + 'ae_nar_rep_train_subset_{0}_{1}_@com_id_{2}.pickle'


'''CONFIGURATIONS'''
# g_tw_sem_units_extractor_config_file = 'agent_conf.conf'
g_sem_units_extractor_config_file = 'sem_units_ext.conf'
g_stopwords_file = 'stopwords.txt'
g_d_pos_pair_weights = {'VERB_NOUN': 0.648, 'NOUN_VERB': 0.648,
                        'NOUN_NOUN': 0.244,
                        'VERB_VERB': 0.040,
                        'NOUN_ADJ': 0.028, 'ADJ_NOUN': 0.028,
                        'VERB_ADV': 0.010, 'ADV_VERB': 0.010,
                        'NOUN_ADV': 0.010, 'ADV_NOUN': 0.010,
                        'VERB_ADJ': 0.011, 'ADJ_VERB': 0.011,
                        'ADP_NOUN': 0.005, 'NOUN_ADP': 0.005,
                        'PROPN_NOUN': 0.004, 'NOUN_PROPN': 0.004}
g_word_embedding_model = 'lexvec'
# TODO
# Modify 'g_lexvec_model_folder', 'g_lexvec_vect_file_path'
# g_lexvec_model_folder = '/home/mf3jh/lib/lexvec/lexvec/python/lexvec/'
g_lexvec_model_folder = '/home/mf3jh/workspace/lib/lexvec/python/lexvec/'
# g_lexvec_vect_file_path = '/home/mf3jh/lib/lexvec/lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
g_lexvec_vect_file_path = '/home/mf3jh/workspace/lib/lexvec/lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
g_embedding_type = ['avg_nps_vect', 'avg_nps_vect_w_avg_edge_vect', 'avg_nps_vect_sp_w_avg_edge_vect']
g_tw_raw_phrase_job_num = 8
g_postgis1_account_conf = 'postgis1_account.conf'
g_postgis1_username = None
g_postgis1_password = None
g_num_phrase_clusters = 150
g_sim_threshold_for_phrase_cluster_embeds = 0.4


'''ENVIRONMENT CHECKING'''
def env_check():
    import os
    from os import path
    if not path.exists(g_tw_work_folder):
        raise Exception('g_tw_work_folder does not exist!')
    if not path.exists(g_tw_raw_data_folder):
        raise Exception('g_tw_raw_data_folder does not exist!')
    if not path.exists(g_postgis1_account_conf):
        raise Exception('postgis1_account.conf is not setup yet!')
    else:
        global g_postgis1_username, g_postgis1_password
        with open(g_postgis1_account_conf, 'r') as in_fd:
            for idx, ln in enumerate(in_fd):
                if idx == 0:
                    g_postgis1_username = ln.strip()
                elif idx == 1:
                    g_postgis1_password = ln.strip()
                else:
                    break
            in_fd.close()
    if not path.exists(g_tw_raw_data_int_folder):
        os.mkdir(g_tw_raw_data_int_folder)
    if not path.exists(g_tw_sem_units_folder):
        os.mkdir(g_tw_sem_units_folder)
    if not path.exists(g_tw_sem_units_int_folder):
        os.mkdir(g_tw_sem_units_int_folder)
    if not path.exists(g_tw_embed_folder):
        os.mkdir(g_tw_embed_folder)
    if not path.exists(g_tw_embed_usr_avg_embeds_int_folder):
        os.mkdir(g_tw_embed_usr_avg_embeds_int_folder)
    if not path.exists(g_tw_phrase_graph_folder):
        os.mkdir(g_tw_phrase_graph_folder)
    if not path.exists(g_tw_response_graph_int_folder):
        os.mkdir(g_tw_response_graph_int_folder)
    if not path.exists(g_tw_phrase_extraction_int_folder):
        os.mkdir(g_tw_phrase_extraction_int_folder)
    if not path.exists(g_tw_raw_phrases_embeds_int_folder):
        os.mkdir(g_tw_raw_phrases_embeds_int_folder)
    if not path.exists(g_tw_ae_narrative_rep_train_sets_folder):
        os.mkdir(g_tw_ae_narrative_rep_train_sets_folder)
    if not path.exists(g_tw_usr_beliefs_folder):
        os.mkdir(g_tw_usr_beliefs_folder)
    if not path.exists(g_tw_response_graph_folder):
        os.mkdir(g_tw_response_graph_folder)
    if not path.exists(g_tw_raw_phrases_phrase_cluster_folder):
        os.mkdir(g_tw_raw_phrases_phrase_cluster_folder)
    if not path.exists(g_udt_folder):
        os.mkdir(g_udt_folder)
    if not path.exists(g_udt_int_folder):
        os.mkdir(g_udt_int_folder)
    if not path.exists(g_udt_community_folder):
        os.mkdir(g_udt_community_folder)
    if not path.exists(g_tw_phrase_cluster_embeds_int_folder):
        os.mkdir(g_tw_phrase_cluster_embeds_int_folder)
    if not path.exists(g_tw_usr_beliefs_train_set_folder):
        os.mkdir(g_tw_usr_beliefs_train_set_folder)
    if not path.exists(g_tw_usr_beliefs_train_set_int_folder):
        os.mkdir(g_tw_usr_beliefs_train_set_int_folder)
    if not path.exists(g_tw_usr_beliefs_saved_models_folder):
        os.mkdir(g_tw_usr_beliefs_saved_models_folder)
    if not path.exists(g_tw_new_tw_folder):
        os.mkdir(g_tw_new_tw_folder)
    if not path.exists(g_tw_new_tw_data_time_interval_folder):
        os.mkdir(g_tw_new_tw_data_time_interval_folder)
    if not path.exists(g_tw_new_tw_time_int_pc_dists_int_folder):
        os.mkdir(g_tw_new_tw_time_int_pc_dists_int_folder)
    if not path.exists(g_udt_tw_srg_trg_data_int_folder):
        os.mkdir(g_udt_tw_srg_trg_data_int_folder)
    if not path.exists(g_tw_new_tw_model_folder):
        os.mkdir(g_tw_new_tw_model_folder)
    if not path.exists(g_events_sem_units_folder):
        os.mkdir(g_events_sem_units_folder)
    if not path.exists(g_tw_pc_onehot_folder):
        os.mkdir(g_tw_pc_onehot_folder)
    if not path.exists(g_tw_pc_onehot_int_folder):
        os.mkdir(g_tw_pc_onehot_int_folder)
    if not path.exists(g_community_analysis_folder):
        os.mkdir(g_community_analysis_folder)
    if not path.exists(g_microsim_folder):
        os.mkdir(g_microsim_folder)
    if not path.exists(g_microsim_msg_prop_graphs_folder):
        os.mkdir(g_microsim_msg_prop_graphs_folder)
    if not path.exists(g_microsim_msg_prop_graph_data_folder):
        os.mkdir(g_microsim_msg_prop_graph_data_folder)
    if not path.exists(g_microsim_sim_data_folder):
        os.mkdir(g_microsim_sim_data_folder)
    if not path.exists(g_microsim_sim_graphs_folder):
        os.mkdir(g_microsim_sim_graphs_folder)
    if not path.exists(g_microsim_sim_graph_data_folder):
        os.mkdir(g_microsim_sim_graph_data_folder)
    if not path.exists(g_microsim_analysis_results_folder):
        os.mkdir(g_microsim_analysis_results_folder)


env_check()
