import logging
import multiprocessing
import threading

from allennlp import pretrained
import networkx as nx
import spacy
# import neuralcoref
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import softmax
import igraph as ig
import matplotlib.pyplot as plt
import re
from gensim.parsing import preprocessing
import json
from collections import deque, OrderedDict
import sd_2_usd
from sd_2_usd import sd_to_usd
from os import walk, path
from networkx.algorithms.operators.binary import union
import sqlite3
import os
import shutil
import time
import math
import random
import sys
import scipy.stats

sys.path.insert(1, '/home/mf3jh/workspace/lib/lexvec/lexvec/python/lexvec/')
import model as lexvec
import scipy.spatial.distance as scipyd
from scipy.special import comb

# g_spacy_model = None
# g_stopwords_path = 'stopwords.txt'
# g_s_stopwords = None
g_elmo_config_folder = '/home/mf3jh/workspace/lib/elmo/'
g_elmo_weight_file = g_elmo_config_folder + 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
g_elmo_options_file = g_elmo_config_folder + 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
g_elmo_model = None
g_np_graph_path = 'np_graph.gml'
g_np_graph_img_path = 'np_graph.pdf'
# {0} for tw_id, {1} for sentence id.
g_core_cls_graph_format = 'core_cls_graph_{0}_{1}.gml'
g_core_cls_graph_img_format = 'core_cls_graph_{0}_{1}.pdf'
g_tw_cascade_1_file = 'second_order_first_level_wheader1.txt'
g_tw_cascade_2_file = 'second_order_second_level_wheader.txt'
g_tw_cascade_files = [g_tw_cascade_1_file, g_tw_cascade_2_file]

g_tw_cascade_results_folder = 'sample_cascades/'
g_tw_cascade_graph_path = g_tw_cascade_results_folder + 'sample_cas_graph.gml'
g_tw_cascade_graph_img_path = g_tw_cascade_results_folder + 'sample_cas_graph.pdf'
g_tw_cascade_core_cls_graph_format = g_tw_cascade_results_folder + '/{0}_core_cls_graph.gml'
g_tw_cascade_core_cls_graph_img_format = g_tw_cascade_results_folder + '/{0}_core_cls_graph.pdf'
g_tw_cascade_train = g_tw_cascade_results_folder + '/train_labels.json'
g_tw_cascade_nps = g_tw_cascade_results_folder + '/noun_phrases.json'
g_tw_cascade_root = 'YD_eeslkY2ZVeHYy0j8y2Q'

# For COVID-19 Twitter Data
g_covid_tw_salient_cascades_semantic_units_folder = '/home/mf3jh/workspace/cp4_narratives/covid19/sal_cas_sem_units/'

# g_test_text = '''United States. The Coronavirus was leaked on purpose to stop the protests. I find it really, really bizarre that a U.S. media realm that can hype the dangers of everything from sharks to fracking to micro-aggressions to Russian-generated Facebook posts is looking at coronavirus and saying, "ACTUALLY, you're in much more danger from the regular flu."'''
# g_test_text = '''Meanwhile, Trump’s Coronavirus Task Force Leader, Ken Cuccinelli, Asks Twitter Followers --- For HELP getting past the Johns Hopkins Paywall to check on coronavirus status updates, rather than their own CDC (who would normally be on top of this)'''
# g_test_text = '''We are looking for a region of central Italy bordering the Adriatic Sea. The area is mostly mountainous and includes Mt. Corno, the highest peak of the mountain range. It also includes many sheep and an Italian entrepreneur has an idea about how to make a little money of them.'''
# g_test_text = '''Mandy likes pizza. She likes burgers as well.'''
g_test_text = '''I understand the Hopkins info is available on PubMed, a free NIH site that aggregates scientific research. That’s the federal govt. Why doesn’t right hand know what left hand is doing?'''
g_test_tw_1 = '''This morning’s classified coronavirus briefing should have been made fully open to the American people—they would be as appalled & astonished as I am by the inadequacy of preparedness & prevention.'''
g_test_tw_2 = '''Meanwhile, Trump’s Coronavirus Task Force Leader, Ken Cuccinelli, Asks Twitter Followers --- For HELP getting past the Johns Hopkins Paywall to check on coronavirus status updates, rather than their own CDC (who would normally be on top of this)'''
g_test_tw_3 = '''I understand the Hopkins info is available on PubMed, a free NIH site that aggregates scientific research. That’s the federal govt. Why doesn’t right hand know what left hand is doing?'''
g_test_tw_4 = '''“In 2018, the Trump administration fired the government’s entire pandemic response chain of command, including the White House management infrastructure.”'''
g_test_tw_5 = '''Kicking door is a bad behavior.'''
g_test_tw_6 = '''This pizza is neither pretty nor delicious.'''
g_test_tw_7 = '''There is no headache.'''
g_test_tw_8 = '''There is not anyone who can make this pizza.'''
g_l_test_tws = [g_test_tw_1, g_test_tw_2, g_test_tw_3, g_test_tw_4]


# g_s_core_deps = {'nsubj', 'nsubjpass', 'dobj', 'iobj', 'csubj', 'csubjpass', 'ccomp', 'xcomp', 'advcl', 'relcl',
#                  'oprd', 'case'}


def allennlp_coref(doc):
    predictor = pretrained.neural_coreference_resolution_lee_2017()
    ret = predictor.predict(document=doc)
    print(ret)


def allennlp_load_elmo():
    global g_elmo_model
    g_elmo_model = Elmo(g_elmo_options_file, g_elmo_weight_file, 2, dropout=0)


def allennlp_elmo_test(text):
    parsed_text = g_spacy_model(text)
    char_ids = batch_to_ids([['I', 'love', 'cat', 'and', 'dog']])
    embeddings = g_elmo_model(char_ids)
    print()


def enum_verbs_from_spacy_dep_parse_tree(root):
    '''
    Enumerate all verbs in a dep parse tree from root to leaves.
    :param
        root: the root of a dep parse tree.
    :return:
        A list of verbs in the BFS-order stored in l_verbs.
    '''
    if root is None:
        logging.error('Invalid root for dep parse tree.')
        return None
    l_verbs = []
    stack_tokens = [root]
    while len(stack_tokens) > 0:
        cur_root = stack_tokens.pop()
        if cur_root.pos_ == 'VERB':
            l_verbs.append(cur_root)
        stack_tokens += list(cur_root.children)
    return l_verbs


# def spacy_extract_nps_n_vects_from_text(text):
#     if g_spacy_model is None:
#         spacy_init()
#     parsed_text = g_spacy_model(text)
#     d_sal_nps = dict()
#     for noun_phrase in parsed_text.noun_chunks:
#         l_sal_words = []
#         for word in noun_phrase:
#             word_lemma = word.lemma_
#             if not word.is_stop and word_lemma not in g_s_stopwords and word_lemma != '' and word not in l_sal_words:
#                 l_sal_words.append(word)
#         sal_np = ' '.join([word_clean(w.lemma_) for w in l_sal_words])
#         if sal_np != '' and sal_np not in d_sal_nps:
#             sal_np_vect = sum([w.vector for w in l_sal_words]) / len(l_sal_words)
#             d_sal_nps[sal_np] = sal_np_vect
#     return d_sal_nps


################################################################################
#                           Active Code Starts Here
################################################################################

class NarrativeAgent:
    def __init__(self, config_file_path):
        if config_file_path is None or config_file_path == '':
            logging.error('Cannot find configuration file!')
            return
        with open(config_file_path, 'r') as in_fd:
            d_conf = json.load(in_fd)
            in_fd.close()
        try:
            self.m_conf_spacy_model_name = d_conf['spacy_model']
            self.m_conf_stopwords_path = d_conf['stopwords_path']
            self.m_conf_spacy_coref_greedyness = float(d_conf['spacy_coref_greedyness'])
            self.m_s_ner_tags = set(d_conf['ner'])
            self.m_s_core_deps = set(d_conf['core_dep'])
            self.m_s_neg_cues = set(d_conf['neg_cue'])
            self.m_spacy_model_ins = self.spacy_init(self.m_conf_spacy_model_name, self.m_conf_spacy_coref_greedyness)
            self.m_s_stopwords = self.load_stopwords(self.m_conf_stopwords_path)
            self.m_d_sd_2_usd = sd_2_usd.g_d_sd_to_usd
        except Exception as err:
            logging.error(err)
            return

    def spacy_init(self, spacy_model_name, coref_greedyness):
        '''
        Load the spaCy model with a specific name.
        :param coref_greedyness:
        :return:
        '''
        # global g_spacy_model
        spacy_model = spacy.load(spacy_model_name)
        # neuralcoref.add_to_pipe(spacy_model, greedyness=coref_greedyness)
        return spacy_model

    def load_stopwords(self, stopwords_path):
        # global g_s_stopwords
        s_stopwords = set([])
        with open(stopwords_path, 'r') as in_fd:
            ln = in_fd.readline()
            while ln:
                sw = ln.strip()
                s_stopwords.add(sw)
                ln = in_fd.readline()
            in_fd.close()
        return s_stopwords

    def word_clean(self, word):
        '''
        Clean the input token.
        1. Remove all non-word characters
        2. Remove HTML and other tags
        3. Remove punctuations
        4. Remove unnecessary white spaces
        5. Remove numbers
        6. Remove short tokens
        :param
            word: A token.
        :return:
            A cleaned token.
        '''
        clean_text = re.sub(r'[^\w\s]', ' ', word)
        clean_text = preprocessing.strip_tags(clean_text)
        clean_text = preprocessing.strip_punctuation(clean_text)
        clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
        clean_text = preprocessing.strip_numeric(clean_text)
        clean_text = preprocessing.strip_short(clean_text, minsize=3)
        return clean_text

    def is_trivial_token(self, spacy_token):
        '''
        Determine if the input token is trivial.
        NOTE:
        This method is not perfect. Any token in a sentence with a capped leading letter can be recognized as an
        entity. In this case, this method cannot detect if the token is trivial.
        NOTE:
        Before calling this function, the dep of the input token should be changed to neg if it is a negation token.
        TODO:
        Any better method to address the above issue?
        How should we deal with special entities? A capped token can be easily a false special entity.
        :param
            spacy_token: A spaCy annotated token.
        :return:
            True -- trivial
            False -- non-trivial
        '''
        clean_token = self.word_clean(spacy_token.lemma_)
        if clean_token == '':
            return True
        if (spacy_token.is_stop or clean_token.lower() in self.m_s_stopwords) and spacy_token.dep_ != 'neg':
            return True
        # if spacy_token.ent_type_ not in self.m_s_ner_tags and spacy_token.dep_ != 'neg':
        #     return True
        return False

    def spacy_pipeline_parse(self, raw_text):
        '''
        Obtain POS, NER, Dep Parse Trees, Noun Chunks and other linguistic features from the text. The text is firstly
        segmented into sentences, and then parsed. spaCy uses Universal Dependencies and POS tags.
        :param
            text: a raw text that may contains multiple sentences.
        :return:
            A list of tagged sentences, and the parsed doc.
        '''
        if raw_text is None:
            return None, None
        doc = self.m_spacy_model_ins(raw_text)
        return list(doc.sents), doc

    def spacy_extract_nps_from_sent(self, spacy_sent):
        '''
        Extract noun phrases from a spaCy parsed sentence. The noun phrases contain non-stopwords and lemmas. Also, each
        lemma in a noun phrase appears only once.
        :param
            spacy_sent: A spaCy parsed sentence. Should not be None.
        :return:
            A list of noun phrase tuples. Each tuple = (noun phrase string, token index set)
        '''
        l_nps = []
        for noun_phrase in spacy_sent.noun_chunks:
            l_sal_lemmas = []
            s_np_idx = set([])
            for word in noun_phrase:
                word_lemma = self.word_clean(word.lemma_)
                if not self.is_trivial_token(word) \
                        and word_lemma not in l_sal_lemmas:
                    # and word not in self.m_s_stopwords \
                    # and word_lemma not in self.m_s_stopwords \
                    # and word_lemma != '' \
                    l_sal_lemmas.append(word_lemma)
                    s_np_idx.add(word.i)
            sal_np = ' '.join(l_sal_lemmas)
            if sal_np != '' and sal_np != ' ' and sal_np not in [item[1] for item in l_nps]:
                l_nps.append((sal_np, s_np_idx))
        return l_nps

    def spacy_extract_npool_from_sent(self, spacy_sent):
        '''
        Extract non-trivial nouns from a spacy parsed sentence. Stopwords are removed. All resulting nouns are converted to
        lemmas.
        :param
            spacy_sent: A spacy parsed sentence. Should not be None.
        :return:
            A set of non-trivial nouns.
        '''
        s_npool = set([])
        for token in spacy_sent:
            if token.pos_ == 'NOUN' and not token.is_stop and token.lemma_ not in self.m_s_stopwords:
                s_npool.add(token.lemma_)
        return s_npool

    def allennlp_dep_parse(self, raw_sent):
        """
        Parse a sentence into a dependency parse tree. The annotations are Stanford dependencies.
        TODO: We may need to move to Universal dependencies. If AllenNLP does not do this, then we would do it ourselves.
        :param
            raw_sent: A sentence with original text
        :return:
            The dependency parse tree for the sentence
        """
        predictor = pretrained.biaffine_parser_stanford_dependencies_todzat_2017()
        ret = predictor.predict(sentence=raw_sent)
        dep_parse_tree = ret['hierplane_tree']['root']
        return dep_parse_tree

    def spacy_dep_parse_tree_to_nx_graph(self, spacy_sent):
        '''
        Convert a spaCy dependency parse tree to a NetworkX directed graph. It would be more convenient for the tree
        pruning.
        :param
            spacy_sent: A spaCy parsed sentence.
        :return:
            A NetworkX directed graph.
        '''
        nx_dep_parse_tree = nx.DiGraph()
        q_nodes = deque()
        q_nodes.append(spacy_sent.root)
        while len(q_nodes) > 0:
            cur = q_nodes.pop()
            for child in cur.children:
                q_nodes.append(child)
                usd_dep = child.dep_
                if child.dep_ in self.m_d_sd_2_usd:
                    usd_dep = self.m_d_sd_2_usd[child.dep_]
                nx_dep_parse_tree.add_edge(cur, child, type=usd_dep)
        return nx_dep_parse_tree

    # def nx_dep_parse_tree_prune(self, nx_dep_parse_tree):
    #     '''
    #     Remove stopwords, punctuations, and other trivial tokens. Also, this function will convert the 'conj' relation
    #     from a linear relation to a tree relation. For example, {x-->a--conj-->b} => {x-->a, x-->b}. And we will convert
    #     the input DiGraph to a Graph.
    #     :param
    #         nx_dep_parse_tree: A dependency parse tree of the NetworkX DiGraph structure.
    #     :return:
    #         A pruned Graph.
    #     '''
    #     if nx_dep_parse_tree is None or type(nx_dep_parse_tree) != nx.DiGraph:
    #         logging.error('nx_dep_parse_tree is invalid!')
    #         return None
    #     # find the root first. usually, the first node should be the root, so this wouldn't be too slow.
    #     root = None
    #     for node in nx_dep_parse_tree.nodes:
    #         if node.dep_ == 'ROOT':
    #             root = node
    #     nx_und_dep_parse_tree = nx.Graph()
    #     q_nodes = deque()
    #     q_nodes.append(root)
    #     while len(q_nodes) > 0:
    #         cur = q_nodes.pop()
    #         parent = list(nx_dep_parse_tree.predecessors(cur))[0]
    #         if self.is_trivial_token(cur.text):
    #             if cur.dep_ == 'ROOT':
    #                 if len(cur.children) < 1:
    #                     return None
    #                 elif len(cur.children) == 1:
    #                     nx_und_dep_parse_tree.add_node()
    #             else:
    #                 nx_dep_parse_tree
    #
    # def nx_dep_parse_tree_contract_noun_phrases(self, l_nps):
    #     print()

    def find_nearest_ancestor_in_nx_graph(self, nx_dep_tree, sent_id, spacy_token):
        conj_jump = False
        if spacy_token.dep_ == 'conj':
            conj_jump = True
        parent = spacy_token.head
        while parent != spacy_token:
            clean_parent = self.word_clean(parent.lemma_)
            clean_parent_id = self.build_node_id(sent_id, clean_parent)
            if clean_parent_id in nx_dep_tree.nodes:
                if conj_jump:
                    spacy_token = parent
                    parent = spacy_token.head
                    conj_jump = False
                    continue
                return clean_parent
            else:
                spacy_token = parent
                parent = spacy_token.head
        return None

    def token_to_np(self, spacy_token, l_nps):
        '''
        Substitude a token with a noun phrase if any. All inputs should be the same sentence.
        :param
            spacy_token: a spaCy token.
            l_nps: a list of noun phrase tuples.
        :return:
            if the input token in contained in a noun phrase, then return the noun phrase string with 'NOUN' as its
            POS tag.
            otherwise, return the cleaned token with its original POS tag.
        '''
        if spacy_token is None:
            raise Exception('spacy_token is None!')
        if len(l_nps) == 0:
            clean_token = self.word_clean(spacy_token.lemma_)
            return clean_token, spacy_token.pos_
        ret_str = self.word_clean(spacy_token.lemma_)
        token_idx = spacy_token.i
        for np_tpl in l_nps:
            if token_idx in np_tpl[1]:
                ret_str = np_tpl[0]
                return ret_str, 'NOUN'
        return ret_str, spacy_token.pos_

    def neg_tag_substitute(self, spacy_sent):
        '''
        Substitute the dependency tags of negation tokens to 'neg'. Due to the imperfection of dependency parsers,
        some negation tokens, e.g. 'neither', may not be tagged as 'neg'. We fix this. The negation token list comes
        from the paper 'Negation Scope Detection for Twitter Sentiment Analysis'. Also, we add 'noone', 'couldnt',
        'wont' and 'arent' that are not included in this paper but in "Sentiment Symposium Tutorial" by Potts 2011.
        :param
            spacy_sent: A spaCy parsed sentence.
        :return:
            A modified spaCy sentence with all considered negation tokens tagged as 'neg'.
        '''
        for token in spacy_sent:
            if token.text in self.m_s_neg_cues:
                token.dep_ = 'neg'
        return spacy_sent

    def build_node_id(self, sent_id, node):
        return sent_id + '|' + node

    def extract_cls_from_sent(self, sent_id, spacy_sent, l_nps):
        '''
        Extract core clause structures from a spacy parsed sentence. Each structure is represented as a graph (a tree
        without root). Each graph contains salient clause structures led by verbs. These structures are inspired by the
        five clause structures of simple sentences:
          1. <intransitive_verb subj>
          2. <transitive_verb subj obj>
          3. <transitive_verb subj iobj dobj>
          4. <subj subj_comp>
          5. <transitive_verb subj obj obj_comp>
        The dependency relations involved in these clause structures are various, and are (partially) enumerated in
        'g_s_core_deps'. For tokens in a noun phrase, substitute the noun phrase for the tokens. We always add lemmas
        as vertices into the resulting graph.
        NOTE:
        We temporarily use the dependency parser in spaCy to do this. Though, it may not be the best solution. Also,
        the parsing results from the spaCy parser may not the same as the visualization from their online demo.
        TODO:
        We may need to compare between the spaCy parser and the biaffine attention parser in AllenNLP to get the better
        one, or we may design a "ensemble" parser taking advantage of both of them.
        TODO:
        1. Negation: We would like to attach a negation word to its most dependent word. This dependent word can be a
        verb, noun, adjective or something else.
        # 2. conj
        3. More precise non-trivial words
        4. Co-references so far seem rather unstable in performance. Also, since a message may contain some content that
        cannot be easily resolved (e.g. pictures and videos), it may lead to further misleading to co-reference
        resolution. Thus, we temporarily do not do this.
        :param
            spacy_sent: A spaCy parsed sentence.
            l_nps: The list of noun phrase tuples contained in the sentence (the noun phrases should have been cleaned).
        :return:
            A NetworkX undirected and unweighted graph that represents the core clause structures. Vertices are strings.
            Edges are induced (i.e. not exactly) by dependencies.
            The node ids in the resulting graph are unique ids rather than actual texts on the nodes. The node labels
            are the actual texts.
        NOTE:
        A vertex in the resulting graph is a composed string of the format: [tw_id]|[sent_idx]|[token]
        '''
        if spacy_sent is None:
            logging.error('spacy_sent is invalid!')
            return None
        root = None
        for token in spacy_sent:
            if token.dep_ == 'ROOT':
                root = token
        # The root can be a negation token. So we put 'neg_tag_substitute' after fetching the root.
        spacy_sent = self.neg_tag_substitute(spacy_sent)
        q_nodes = deque()
        nx_cls = nx.Graph()
        if not self.is_trivial_token(root):
            cur_node, cur_pos = self.token_to_np(root, l_nps)
            nx_cls.add_node(self.build_node_id(sent_id, cur_node), txt=cur_node, pos=cur_pos, type='root')
        for child in root.children:
            q_nodes.append(child)
        while len(q_nodes) > 0:
            cur = q_nodes.pop()
            if sd_to_usd(cur.dep_) in self.m_s_core_deps and not self.is_trivial_token(cur):
                cur_node, cur_pos = self.token_to_np(cur, l_nps)
                if cur_node not in nx_cls.nodes:
                    nearest_ancestor = self.find_nearest_ancestor_in_nx_graph(nx_cls, sent_id, cur)
                    if nearest_ancestor is None:
                        nx_cls.add_node(self.build_node_id(sent_id, cur_node), txt=cur_node, pos=cur_pos, type='root')
                    elif nearest_ancestor != cur_node:
                        nx_cls.add_node(self.build_node_id(sent_id, cur_node), txt=cur_node, pos=cur_pos, type='node')
                        nx_cls.add_edge(self.build_node_id(sent_id, nearest_ancestor),
                                        self.build_node_id(sent_id, cur_node))
            for child in cur.children:
                q_nodes.append(child)
        l_roots = [node[0] for node in nx_cls.nodes(data=True) if node[1]['type'] == 'root']
        for i in range(0, len(l_roots) - 1):
            for j in range(i + 1, len(l_roots)):
                nx_cls.add_edge(l_roots[i], l_roots[j])
        if len(nx_cls.nodes) == 1 and list(nx_cls.nodes)[0] in self.m_s_neg_cues:
            return None
        return nx_cls

    def extract_sem_units_from_text(self, raw_txt, txt_id):
        '''
        Extract semantic units for a given piece of raw text.
        :param
            raw_txt: A piece of raw text which can contain multiple sentences.
            txt_id: The unique ID for the raw_txt.
        :return:
            The core clause structure graph (the union graph from all sentences), the list of noun phrases.
        TODO:
        We may add other semantic units.
        '''
        if raw_txt is None or len(raw_txt) == 0:
            logging.debug('A trivial text occurs.')
            return None, None

        # # EXTRACT NOUN PHRASES
        # l_nps = self.spacy_extract_nps_from_sent(spacy_sent)
        #
        # # EXTRACT NOUN POOL
        # s_npool = self.spacy_extract_npool_from_sent(spacy_sent)

        l_nps = []
        nx_cls = nx.Graph()
        l_spacy_sents, spacy_doc = self.spacy_pipeline_parse(raw_txt)
        for sent_id, spacy_sent in enumerate(l_spacy_sents):
            l_nps += self.spacy_extract_nps_from_sent(spacy_sent)
            nx_cls = self.union_nx_graphs(nx_cls,
                                          self.extract_cls_from_sent(txt_id + '|' + str(sent_id), spacy_sent, l_nps))
        return nx_cls, l_nps

    def text_clean(self, raw_text):
        if raw_text is None or raw_text == '':
            return None
        l_clean_sents = []
        l_dirty_sents = raw_text.split('\n')
        for raw_dirt_sent in l_dirty_sents:
            # remove url
            clean_text = re.sub(r'url: [\S]*', '', raw_dirt_sent)
            clean_text = re.sub(r'http[\S]*', '', clean_text)
            # remove hashed ids
            clean_text = re.sub(r'@un:\s[\S]{22}\s', ' ', clean_text)
            clean_text = re.sub(r'@[^\s]+', ' ', clean_text)
            clean_text = re.sub(r'\s[\S]{22}\s', ' ', clean_text)
            # remove # symbol
            clean_text = re.sub(r'#', ' ', clean_text)
            # remove unnecessary white spaces
            clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
            clean_text = clean_text.strip()
            trivial_test = re.match(r'.*[a-zA-A]*', clean_text)
            if trivial_test is not None:
                l_clean_sents.append(clean_text)
        return '\n'.join(l_clean_sents)
        # return l_clean_sents

    def union_nx_graphs(self, nx_1, nx_2):
        '''
        TODO:
        Graph union may connect two graphs together. Should we really do this?
        '''
        if nx_1 is None or nx_2 is None:
            logging.error('nx_1 or nx_2 is empty!')
            raise Exception('nx_1 or nx_2 is empty!')
        for node in nx_2.nodes(data=True):
            nx_1.add_node(node[0], txt=node[1]['txt'], pos=node[1]['pos'])
        for edge in nx_2.edges:
            nx_1.add_edge(edge[0], edge[1])
        return nx_1

    def output_nx_graph(self, nx_graph, fig_path):
        if nx_graph is None or len(nx_graph.nodes) == 0:
            return
        plt.figure(1, figsize=(15, 15), tight_layout={'pad': 1, 'w_pad': 200, 'h_pad': 200, 'rect': None})
        pos = nx.spring_layout(nx_graph, k=0.8)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)
        d_node_labels = {node[0]: node[1]['txt'] + ':' + node[1]['pos'] for node in nx_graph.nodes(data=True)}
        nx.draw_networkx_nodes(nx_graph, pos, node_size=50)
        nx.draw_networkx_labels(nx_graph, pos, labels=d_node_labels, font_size=25, font_family="sans-serif")
        nx.draw_networkx_edges(nx_graph, pos, width=2, edge_color='b')
        plt.savefig(fig_path, format="PNG")
        plt.clf()
        # plt.show()

    def task_multithreads(self, op_func, l_tasks, num_threads, output_folder=None, en_draw=False, other_params=()):
        '''
        A multithreading wrapper for a list a task with texts.
        :param
            op_func: The thread function to process a subset of tasks.
            l_tasks: A list of tasks. Each task is of the format: (task_id, text)
            num_threads: The number of threads
            output_folder: For outputs
            en_draw: True - draw outputs if necessary
        :return:
            No direct return value but outputs in the output folder and draws if any.
        '''
        timer_1 = time.time()
        batch_size = math.ceil(len(l_tasks) / num_threads)
        l_l_subtasks = []
        for i in range(0, len(l_tasks), batch_size):
            if i + batch_size < len(l_tasks):
                l_l_subtasks.append(l_tasks[i:i + batch_size])
            else:
                l_l_subtasks.append(l_tasks[i:])
        logging.debug('%s threads.' % len(l_l_subtasks))

        l_threads = []
        t_id = 0
        for l_each_batch in l_l_subtasks:
            t = threading.Thread(target=op_func, args=(l_each_batch, 't_mul_task_' + str(t_id),
                                                       output_folder, en_draw) + other_params)
            t.setName('t_mul_task_' + str(t_id))
            t.start()
            l_threads.append(t)
            t_id += 1

        while len(l_threads) > 0:
            for t in l_threads:
                if t.is_alive():
                    t.join(1)
                else:
                    l_threads.remove(t)
                    logging.debug('Thread %s is finished.' % t.getName())

        logging.debug('All done in %s sec for %s tasks.' % (time.time() - timer_1, len(l_tasks)))

    def sem_unit_extraction_thread(self, l_tasks, thread_id, output_folder, en_draw):
        '''
        Extract semantic units for a list of tasks with texts.
        TODO:
        Add modifier structures and noun pools.
        :param
            l_tasks: A list of tasks with texts, each of which is of the format: (task_id, text)
            thread_id: Thread name
            output_folder: For outputs
            en_draw: Draw outputs if necessary
        :return:
            No direct return value but outputs in the output folder and draws if any.
            Each core clause structure (of a sentence) is stored in a gml file representing an undirected and unweighted
            graph. The gml file name is of the format: [task_id]|[sent_id]_cls_graph.gml
            Each noun phrase output is also stored in a file with its name of the format: [task_id]|[sent_id]_nps.txt
        '''
        timer_start = time.time()
        cnt = 0
        unique_id = 0
        for tw_id, sent_txt in l_tasks:
            nx_cls, l_nps = self.extract_sem_units_from_text(sent_txt, tw_id + '|' + str(unique_id))
            # logging.debug('%s nodes for %s' % (len(nx_cls.nodes()), tw_id))
            if nx_cls is None:
                continue
            # nx.write_gml(nx_cls, output_folder + tw_id + '|' + str(unique_id) + '_cls_graph.gml')
            nx.write_gml(nx_cls, output_folder + tw_id + '_cls_graph.gml')
            if en_draw:
                self.output_nx_graph(nx_cls, output_folder + tw_id + '_cls_graph.png')
            nps_str = '\n'.join([item[0] for item in l_nps])
            # with open(output_folder + tw_id + '|' + str(unique_id) + '_nps.txt', 'w+') as out_fd:
            with open(output_folder + tw_id + '_nps.txt', 'w+') as out_fd:
                out_fd.write(nps_str)
                out_fd.close()
            cnt += 1
            unique_id += 1
            if cnt % 1000 == 0 and cnt >= 1000:
                logging.debug(
                    '%s: %s sentences are done in %s secs.' % (thread_id, cnt, str(time.time() - timer_start)))
        logging.debug('%s: %s sentences all done in %s secs.' % (thread_id, cnt, str(time.time() - timer_start)))

    def sem_unit_stats(self, sem_unit_folder, output_folder):
        l_degrees = []
        l_nodes = []
        l_edges = []
        l_nps = []
        cnt = 0
        for (dirpath, dirname, filenames) in walk(sem_unit_folder):
            for filename in filenames:
                if filename[-14:] == '_cls_graph.gml':
                    sem_unit_name = filename[:-14]
                    try:
                        cls_graph = nx.read_gml(dirpath + '/' + filename)
                        l_nodes.append(cls_graph.number_of_nodes())
                        l_edges.append(cls_graph.number_of_edges())
                        l_degrees.append(sum(dict(cls_graph.degree()).values()) / float(cls_graph.number_of_nodes()))
                    except:
                        pass
                elif filename[-8:] == '_nps.txt':
                    sem_unit_name = filename[:-8]
                    with open(dirpath + '/' + filename, 'r') as in_fd:
                        lns = in_fd.readlines()
                        ln_cnt = len(lns)
                        l_nps.append(ln_cnt)
                        in_fd.close()
                cnt += 1
                if cnt % 10000 == 0 and cnt >= 10000:
                    with open(output_folder + 'sem_unit_stats_' + str(cnt) + '_.txt', 'w+') as out_fd:
                        out_fd.write('cls: avg. degrees:\n')
                        out_fd.write(','.join([str(num) for num in l_degrees]))
                        out_fd.write('\n')
                        out_fd.write('cls: nodes:\n')
                        out_fd.write(','.join([str(num) for num in l_nodes]))
                        out_fd.write('\n')
                        out_fd.write('cls: edges:\n')
                        out_fd.write(','.join([str(num) for num in l_edges]))
                        out_fd.write('\n')
                        out_fd.write('nps:\n')
                        out_fd.write(','.join([str(num) for num in l_nps]))
                        out_fd.write('\n')
                        out_fd.close()
                    logging.debug('%s sem unit stats are done.' % cnt)
        with open(output_folder + 'sem_unit_stats_' + str(cnt) + '_.txt', 'w+') as out_fd:
            out_fd.write('cls: avg. degrees:\n')
            out_fd.write(','.join([str(num) for num in l_degrees]))
            out_fd.write('\n')
            out_fd.write('cls: nodes:\n')
            out_fd.write(','.join([str(num) for num in l_nodes]))
            out_fd.write('\n')
            out_fd.write('cls: edges:\n')
            out_fd.write(','.join([str(num) for num in l_edges]))
            out_fd.write('\n')
            out_fd.write('nps:\n')
            out_fd.write(','.join([str(num) for num in l_nps]))
            out_fd.write('\n')
            out_fd.close()


# def neuralcoref_parse(doc):
#     l_sents, parsed_doc = spacy_pipeline(doc)
#     print(parsed_doc._.has_coref)
#     print(parsed_doc._.coref_clusters)


def pairwise_np_comparison(d_sal_nps, en_draw=True):
    l_sal_nps = list(d_sal_nps.keys())
    g_nps = nx.Graph()
    m_sal_np_sim = np.zeros((len(l_sal_nps), len(l_sal_nps)))
    for i in range(0, len(l_sal_nps) - 1):
        for j in range(i, len(l_sal_nps)):
            if i == j:
                m_sal_np_sim[i][j] = 1.0
            else:
                m_sal_np_sim[i][j] = (2 - cosine(d_sal_nps[l_sal_nps[i]], d_sal_nps[l_sal_nps[j]])) / 2
                # m_sal_np_sim[i][j] = 1 - cosine(d_sal_nps[l_sal_nps[i]], d_sal_nps[l_sal_nps[j]])
                m_sal_np_sim[j][i] = m_sal_np_sim[i][j]
                g_nps.add_edge(l_sal_nps[i], l_sal_nps[j], weight=m_sal_np_sim[i][j])
    # m_sal_np_sim = softmax(m_sal_np_sim)
    # for i in range(0, len(l_sal_nps)-1):
    #     for j in range(i, len(l_sal_nps)):
    #         if i != j:
    #             g_nps.add_edge(l_sal_nps[i], l_sal_nps[j], weight=m_sal_np_sim[i][j])
    if en_draw:
        # nx.write_gml(g_nps, g_np_graph_path)
        # ig_nps = ig.read(g_np_graph_path)
        # ig.plot(ig_nps, g_np_graph_img_path, bbox=(1024, 1024), vertex_label_size=20)
        # nx.draw(g_nps)
        elarge = [(u, v) for (u, v, d) in g_nps.edges(data=True) if d["weight"] > 0.8]
        esmall = [(u, v) for (u, v, d) in g_nps.edges(data=True) if d["weight"] <= 0.8]
        plt.figure(1, figsize=(15, 15), tight_layout={'pad': 1, 'w_pad': 100, 'h_pad': 100, 'rect': None})
        # pos = nx.spring_layout(g_nps, scale=None,  iterations=200)
        pos = nx.spring_layout(g_nps)
        nx.draw_networkx_nodes(g_nps, pos, node_size=100)
        nx.draw_networkx_labels(g_nps, pos, font_size=25, font_family="sans-serif")
        nx.draw_networkx_edges(g_nps, pos, edgelist=elarge, width=6, alpha=0.5, edge_color='b')
        # nx.draw(g_nps)
        plt.show()


def draw_ig_graph(graph_path, img_path):
    ig_graph = ig.read(graph_path)
    # layout = ig_graph.layout("kk")
    ig.plot(ig_graph, img_path, bbox=(512, 512), vertex_label_size=10, margin=120, vertex_size=10,
            vertex_color='blue', vertex_label_dist=2)


def read_cascades():
    d_tw = dict()
    l_roots = []
    # for tw_cas_file in g_tw_cascade_files:
    with open(g_tw_cascade_1_file, 'r') as in_fd:
        ln = in_fd.readline()
        ln = in_fd.readline()
        while ln:
            l_fields = ln.split('|')
            id = l_fields[0]
            l_roots.append(id)
            txt = l_fields[1]
            sup = l_fields[2]
            kwd = l_fields[3]
            man = l_fields[4]
            child_id = l_fields[5]
            child_type = l_fields[6]
            child_ins = dict()
            child_ins[child_id] = child_type
            child_txt = l_fields[7]
            child_sup = l_fields[8]
            child_kwd = l_fields[9]
            child_man = l_fields[10]
            if id not in d_tw:
                d_tw[id] = {'txt': txt, 'sup': sup, 'kwd': kwd, 'man': man, 'children': [child_id]}
            else:
                if child_id not in d_tw[id]['children']:
                    d_tw[id]['children'].append(child_id)
            if child_id not in d_tw:
                d_tw[child_id] = {'txt': child_txt, 'sup': child_sup, 'kwd': child_kwd,
                                  'man': child_man, 'children': []}
            ln = in_fd.readline()
        in_fd.close()

    with open(g_tw_cascade_2_file, 'r') as in_fd:
        ln = in_fd.readline()
        ln = in_fd.readline()
        while ln:
            l_fields = ln.split('|')
            child_id = l_fields[0]
            child_type = l_fields[1]
            child_txt = l_fields[2]
            child_sup = l_fields[3]
            child_kwd = l_fields[4]
            child_man = l_fields[5]
            grand_id = l_fields[6]
            grand_type = l_fields[7]
            grand_ins = dict()
            grand_ins[grand_id] = grand_type
            grand_txt = l_fields[8]
            grand_sup = l_fields[9]
            grand_kwd = l_fields[10]
            grand_man = l_fields[11]
            if child_id not in d_tw:
                d_tw[child_id] = {'txt': child_txt, 'sup': child_sup, 'kwd': child_kwd, 'man': child_man,
                                  'children': [grand_id]}
            else:
                if grand_id not in d_tw[child_id]['children']:
                    d_tw[child_id]['children'].append(grand_id)
            if grand_id not in d_tw:
                d_tw[grand_id] = {'txt': grand_txt, 'sup': grand_sup, 'kwd': grand_kwd,
                                  'man': grand_man, 'children': []}
            ln = in_fd.readline()

    picked_d_tw = {k: d_tw[k] for k in l_roots}
    l_ordered_tw = sorted(picked_d_tw.keys(), key=lambda k: len(picked_d_tw[k]['children']), reverse=True)
    return d_tw, picked_d_tw, l_ordered_tw


# For COVID-19 Twitter Data
def covid_extract_tw_sem_units(agent_ins):
    for (dirpath, dirname, filenames) in walk(g_covid_tw_salient_cascades_semantic_units_folder):
        for filename in filenames:
            if filename[-9:] != '_txt.json':
                continue
            with open(dirpath + '/' + filename, 'r') as in_fd:
                cas_id = filename[:-9]
                l_nps = []
                nx_cls = nx.Graph()
                for ln in in_fd:
                    ln_json = json.loads(ln)
                    # tw_id = ln_json['tw_id']
                    tw_raw_txt = ln_json['raw_txt']
                    tw_clean_txt = agent_ins.text_clean(tw_raw_txt)
                    l_spacy_sents, spacy_doc = agent_ins.spacy_pipeline_parse(tw_clean_txt)
                    for spacy_sent in l_spacy_sents:
                        l_nps += agent_ins.spacy_extract_nps_from_sent(spacy_sent)
                        nx_cls = agent_ins.union_nx_graphs(nx_cls, agent_ins.extract_cls_from_sent(spacy_sent, l_nps))
                in_fd.close()
            nx.write_gml(nx_cls, dirpath + '/' + cas_id + '_cls_graph.gml')
            agent_ins.output_nx_graph(nx_cls, dirpath + '/' + cas_id + '_cls_graph.png')
            nps_str = '\n'.join([item[0] for item in l_nps])
            with open(dirpath + '/' + cas_id + '_nps.txt', 'w+') as out_fd:
                out_fd.write(nps_str)
                out_fd.close()


g_covid_resp_db = 'covid_tw_resp.db'
g_covid_resp_pairs_file = 'covid_tw_resp_pairs.txt'
g_covid_resp_en_sem_unit_folder = '/home/mf3jh/workspace/cp4_narratives/covid19/en_sem_units/'
def covid_resp_extract_tw_sem_units(agent_ins):
    resp_db_conn = sqlite3.connect(g_covid_resp_db)
    resp_db_cur = resp_db_conn.cursor()
    resp_sql_str = '''select resp_raw_txt from covid_tw_resp where resp_tw_id = ?'''

    s_tw_ids_for_sem_units = set([])
    with open(g_covid_resp_pairs_file, 'r') as in_fd:
        for ln in in_fd:
            l_tw_ids = [tw_id.strip() for tw_id in ln.split('|')]
            s_tw_ids_for_sem_units.add(l_tw_ids[0])
            s_tw_ids_for_sem_units.add(l_tw_ids[1])
        in_fd.close()
    logging.debug('%s tws for sem units.' % len(s_tw_ids_for_sem_units))

    l_sem_units_tasks = []
    for tw_id in s_tw_ids_for_sem_units:
        resp_db_cur.execute(resp_sql_str, (tw_id,))
        rec = resp_db_cur.fetchone()
        tw_raw_txt = rec[0]
        l_clean_sents = agent_ins.text_clean(tw_raw_txt)
        tw_clean_txt = '\n'.join(l_clean_sents)
        sem_units_task = (tw_id, tw_clean_txt)
        l_sem_units_tasks.append(sem_units_task)
    logging.debug('Sem units tasks are ready.')

    agent_ins.task_multithreads(agent_ins.sem_unit_extraction_thread,
                                l_sem_units_tasks,
                                10,
                                output_folder=g_covid_resp_en_sem_unit_folder)


# For Ven Twitter Data
version = 'v2-1'
g_ven_tw_folder = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/'
# g_ven_tw_db = g_ven_tw_folder + 'ven_tw_en.db'
g_ven_tw_cas_folder = g_ven_tw_folder + 'cascades/'
g_selected_cas_tint_dist = g_ven_tw_folder + 'ven_tw_sel_cas_tint_dist.json'
g_ven_temp_folder = g_ven_tw_folder + '6_samples/'
g_selected_cas_ids = ['ki4OfrPWfLm3mVoykrvgZA', 'NAUniYUFrv9ohzs8IeFcVg', 'eyzPCXEkTdh-dSUGkme0Ug',
                      'WcVqnWcvB4a5u45m3wr1Rg', '6RlZEobUb5uwyaSq47byqQ', 'YBdNG1rWfnZfuYEGWbu6Dg']
g_ven_orig_en_tw_ids_path = g_ven_tw_folder + 'ven_tw_orig_en_ids_' + version + '.txt'
g_ven_tw_db = g_ven_tw_folder + 'ven_tw_en_' + version + '.db'
g_ven_orig_en_tw_lens_path = g_ven_tw_folder + 'ven_tw_orig_en_lens_' + version + '.txt'
g_ven_sent_len_threshold_lower = 5
g_ven_sent_len_threshold_upper = 20
g_ven_good_sent_path = g_ven_tw_folder + 'ven_good_sents.txt'
g_ven_sem_unit_folder = g_ven_tw_folder + 'sem_units_full/'
g_ven_sem_unit_stats_path = g_ven_tw_folder + 'ven_good_sent_sem_unit_stats.txt'
g_cls_comp_folder = g_ven_tw_folder + 'cls_comparisons/'
g_cls_comp_sent_path = g_cls_comp_folder + 'cls_comp_sents.txt'
g_cls_comp_tasks_path = g_cls_comp_folder + 'cls_comp_tasks.json'
g_lexvec_model = None
g_spacy_model = None


def ven_extract_tw_sem_units(agent_ins):
    timer_1 = time.time()
    en_db_conn = sqlite3.connect(g_ven_tw_db)
    en_db_cur = en_db_conn.cursor()
    en_sql_str = '''select raw_txt from ven_tw_en where tw_id = ?'''
    timer_draw_cum = 0
    cnt = 0
    # for (dirpath, dirname, filenames) in walk(g_ven_tw_cas_folder):
    l_tasks = ven_sem_unit_task_lists()
    for filename in l_tasks:
        # if filename[-4:] != '.gml':
        #     continue
        sem_unit_name = filename[:-4]
        sem_unit_folder = g_ven_tw_cas_folder + sem_unit_name
        if not os.path.exists(sem_unit_folder):
            os.mkdir(sem_unit_folder)
        else:
            shutil.rmtree(sem_unit_folder, ignore_errors=True)
            os.mkdir(sem_unit_folder)
        cas_graph = nx.read_gml(g_ven_tw_cas_folder + filename)
        for tw_id in cas_graph.nodes:
            en_db_cur.execute(en_sql_str, (tw_id,))
            tw_rec = en_db_cur.fetchone()
            if tw_rec is None:
                logging.debug('%s does not have raw_txt.' % tw_id)
                continue
            l_nps = []
            nx_cls = nx.Graph()
            tw_raw_txt = tw_rec[0]
            tw_clean_txt = agent_ins.text_clean(tw_raw_txt)
            l_spacy_sents, spacy_doc = agent_ins.spacy_pipeline_parse(tw_clean_txt)
            for sent_id, spacy_sent in enumerate(l_spacy_sents):
                l_nps += agent_ins.spacy_extract_nps_from_sent(spacy_sent)
                nx_cls = agent_ins.union_nx_graphs(nx_cls,
                                                   agent_ins.extract_cls_from_sent(tw_id + '|' + str(sent_id),
                                                                                   spacy_sent, l_nps))
            nx.write_gml(nx_cls, sem_unit_folder + '/' + tw_id + '_cls_graph.gml')
            timer_2 = time.time()
            agent_ins.output_nx_graph(nx_cls, sem_unit_folder + '/' + tw_id + '_cls_graph.png')
            timer_3 = time.time()
            timer_draw_cum += (timer_3 - timer_2)
            nps_str = '\n'.join([item[0] for item in l_nps])
            with open(sem_unit_folder + '/' + tw_id + '_nps.txt', 'w+') as out_fd:
                out_fd.write(nps_str)
                out_fd.close()
            cnt += 1
    en_db_conn.close()
    timer_4 = time.time()
    timer_all = timer_4 - timer_1 - timer_draw_cum
    logging.debug('%s sec for %s tws in total.' % (timer_all, cnt))


def ven_tw_sem_units_thread(agent_ins, l_tasks):
    en_db_conn = sqlite3.connect(g_ven_tw_db)
    en_db_cur = en_db_conn.cursor()
    en_sql_str = '''select raw_txt from ven_tw_en where tw_id = ?'''
    cnt = 0
    for filename in l_tasks:
        sem_unit_name = filename[:-4]
        sem_unit_folder = g_ven_tw_cas_folder + sem_unit_name
        if not os.path.exists(sem_unit_folder):
            os.mkdir(sem_unit_folder)
        else:
            shutil.rmtree(sem_unit_folder, ignore_errors=True)
            os.mkdir(sem_unit_folder)
        cas_graph = nx.read_gml(g_ven_tw_cas_folder + filename)
        for tw_id in cas_graph.nodes:
            en_db_cur.execute(en_sql_str, (tw_id,))
            tw_rec = en_db_cur.fetchone()
            if tw_rec is None:
                logging.debug('%s does not have raw_txt.' % tw_id)
                continue
            l_nps = []
            nx_cls = nx.Graph()
            tw_raw_txt = tw_rec[0]
            tw_clean_txt = agent_ins.text_clean(tw_raw_txt)
            l_spacy_sents, spacy_doc = agent_ins.spacy_pipeline_parse(tw_clean_txt)
            for sent_id, spacy_sent in enumerate(l_spacy_sents):
                l_nps += agent_ins.spacy_extract_nps_from_sent(spacy_sent)
                nx_cls = agent_ins.union_nx_graphs(nx_cls,
                                                   agent_ins.extract_cls_from_sent(tw_id + '|' + str(sent_id),
                                                                                   spacy_sent, l_nps))
            # output_folder = g_ven_tw_folder + '/' + 'time_complexity/'
            output_folder = sem_unit_folder + '/'
            nx.write_gml(nx_cls, output_folder + tw_id + '_cls_graph.gml')
            agent_ins.output_nx_graph(nx_cls, output_folder + tw_id + '_cls_graph.png')
            nps_str = '\n'.join([item[0] for item in l_nps])
            with open(output_folder + tw_id + '_nps.txt', 'w+') as out_fd:
                out_fd.write(nps_str)
                out_fd.close()
            cnt += 1
    en_db_conn.close()
    logging.debug('%s tws.' % cnt)


def ven_sem_unit_task_lists():
    l_tasks = []
    for (dirpath, dirname, filenames) in walk(g_ven_tw_cas_folder):
        for filename in filenames:
            if filename[-4:] != '.gml' or filename[-14:] == '_cls_graph.gml':
                continue
            if os.path.exists(g_ven_tw_cas_folder + filename[:-4]) \
                    and len(os.listdir(g_ven_tw_cas_folder + filename[:-4])) > 0:
                continue
            l_tasks.append(filename)
    logging.debug('%s tasks in total.' % len(l_tasks))
    return l_tasks


# matplotlib in our current way doesn't support multithreading, so 'output_nx_graph' doesn't work now!!!
def ven_sem_units_multithreads(op_func):
    timer_1 = time.time()
    l_tasks = ven_sem_unit_task_lists()
    batch_size = math.ceil(len(l_tasks) / multiprocessing.cpu_count())
    l_l_subtasks = []
    for i in range(0, len(l_tasks), batch_size):
        if i + batch_size < len(l_tasks):
            l_l_subtasks.append(l_tasks[i:i + batch_size])
        else:
            l_l_subtasks.append(l_tasks[i:])

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_subtasks:
        agent_ins = NarrativeAgent('agent_config.conf')
        t = threading.Thread(target=op_func, args=(agent_ins, l_each_batch))
        t.setName('ven_sem_units_' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('Thread %s is finished.' % t.getName())

    logging.debug('All done in %s sec for %s tasks.' % (time.time() - timer_1, len(l_tasks)))


def get_selected_tint_tws():
    with open(g_selected_cas_tint_dist, 'r') as in_fd:
        d_sel_cas = json.load(in_fd)
        in_fd.close()
    d_tint_tws = dict()
    for cas_id in d_sel_cas:
        for tint in d_sel_cas[cas_id]:
            if tint in d_tint_tws:
                d_tint_tws[tint] += d_sel_cas[cas_id][tint]
            else:
                d_tint_tws[tint] = d_sel_cas[cas_id][tint]
    l_tints = sorted(list(d_tint_tws.keys()), key=lambda k: int(k.split('_')[1]))
    return l_tints, d_tint_tws


##########################################
# Semantic Unit Verification
##########################################
def external_spacy_init(spacy_model_name, coref_greedyness=0.5):
    global g_spacy_model
    g_spacy_model = spacy.load(spacy_model_name)
    # neuralcoref.add_to_pipe(g_spacy_model, greedyness=coref_greedyness)
    return g_spacy_model


def sent_shuffle(sent_str, shuffle_cnt):
    '''
    Shuffle a sentence in specific times.
    :param
        sent_str: The original sentence string
        shuffle_cnt: The required number of shuffles
    :return:
        A list of shuffled sentence strings
    '''
    l_shuffled_sents = []
    sent_seq = [token.strip() for token in sent_str.split(' ')]
    rand_seq = sent_seq
    while shuffle_cnt > 0:
        random.shuffle(rand_seq)
        rand_sent = ' '.join(rand_seq)
        l_shuffled_sents.append(rand_sent)
        shuffle_cnt -= 1
    return l_shuffled_sents


def get_lexvec_vec(phrase):
    # word_vec = g_lexvec_model.wv[word]
    phrase_vec = np.zeros(300)
    l_words = [word.strip() for word in phrase.split(' ')]
    for word in l_words:
        word_vec = g_lexvec_model.word_rep(word)
        phrase_vec += word_vec
    return phrase_vec


def load_lexvec_model():
    global g_lexvec_model
    # model_file = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.300d.W+C.pos.vectors'
    # g_lexvec_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    model_file = '%s/workspace/lib/lexvec/' % os.environ['HOME'] + 'lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
    g_lexvec_model = lexvec.Model(model_file)


def cls_graph_comp_node_match(node_1, node_2):
    if node_1 is None or node_2 is None:
        raise Exception('Either node_1_str or node_2_str is None.')
    s_node_1_tokens = set([item.strip() for item in node_1['txt'].split(' ')])
    s_node_2_tokens = set([item.strip() for item in node_2['txt'].split(' ')])
    s_intersection = s_node_1_tokens.intersection(s_node_2_tokens)
    if len(s_intersection) > 0:
        return True
    return False


def cls_graph_comp_node_subst_cost(node_1, node_2):
    if node_1 is None or node_2 is None:
        raise Exception('Either node_1_str or node_2_str is None.')
    node_1_txt = node_1['txt']
    node_2_txt = node_2['txt']
    l_node_1_spacy_lemmas = []
    for token in [item.strip() for item in node_1_txt.split(' ')]:
        spacy_doc = g_spacy_model(token)
        for spacy_token in spacy_doc:
            spacy_token_lemma = spacy_token.lemma_
            l_node_1_spacy_lemmas.append(spacy_token_lemma)

    l_node_2_spacy_lemmas = []
    for token in [item.strip() for item in node_2_txt.split(' ')]:
        spacy_doc = g_spacy_model(token)
        for spacy_token in spacy_doc:
            spacy_token_lemma = spacy_token.lemma_
            l_node_2_spacy_lemmas.append(spacy_token_lemma)

    s_node_1_tokens = set(l_node_1_spacy_lemmas)
    s_node_2_tokens = set(l_node_2_spacy_lemmas)
    if s_node_1_tokens == s_node_2_tokens:
        return 0
    if len(s_node_1_tokens.intersection(s_node_2_tokens)) == 0:
        return 40
    node_1_vect = get_lexvec_vec(node_1_txt)
    node_2_vect = get_lexvec_vec(node_2_txt)
    diff = cosine(node_1_vect, node_2_vect) * 40
    return diff


def cls_graph_comp_node_del_cost(node):
    return 40


def cls_graph_comp_node_ins_cost(node):
    return 40


def cls_graph_comp_edge_subst_cost(edge_1, edge_2):
    l_edge_1_n1_tokens = edge_1['n1_ts']
    l_edge_1_n2_tokens = edge_1['n2_ts']
    s_edge_1_n1_tokens = set(l_edge_1_n1_tokens)
    s_edge_1_n2_tokens = set(l_edge_1_n2_tokens)

    l_edge_2_n1_tokens = edge_2['n1_ts']
    l_edge_2_n2_tokens = edge_2['n2_ts']
    s_edge_2_n1_tokens = set(l_edge_2_n1_tokens)
    s_edge_2_n2_tokens = set(l_edge_2_n2_tokens)

    if (len(s_edge_1_n1_tokens.intersection(s_edge_2_n1_tokens)) > 0
        and len(s_edge_1_n2_tokens.intersection(s_edge_2_n2_tokens)) > 0) \
            or (len(s_edge_1_n1_tokens.intersection(s_edge_2_n2_tokens)) > 0
                and len(s_edge_1_n2_tokens.intersection(s_edge_2_n1_tokens)) > 0):
        return 0
    return 1


def cls_graph_comp_edge_del_cost(edge):
    return 1


def cls_graph_comp_edge_ins_cost(edge):
    return 1


def cls_graph_edit_distance_stat(graph_src, graph_trg):
    '''
    Typically graph_src is for the shuffled sentence and graph_trg is for the original sentence.
    :return:
    '''
    if graph_src is None or graph_trg is None:
        raise Exception('Either of the input graphs is None.')
    if len(graph_src.nodes()) + len(graph_trg.nodes()) + len(graph_src.edges()) + len(graph_trg.edges()) == 0:
        return {'ged': 0.0, 'posd': 0.0, 'edged': 0.0, 'pos_match': {}, 'pos_diff': {}, 'edge_match': {},
                'edge_diff': {}}
    for edge in graph_src.edges(data=True):
        node_1 = edge[0]
        node_2 = edge[1]
        l_node_1_tokens = [token.strip() for token in node_1.split('|')[-1].split(' ')]
        l_node_2_tokens = [token.strip() for token in node_2.split('|')[-1].split(' ')]
        edge[2]['n1_ts'] = l_node_1_tokens
        edge[2]['n2_ts'] = l_node_2_tokens
    for edge in graph_trg.edges(data=True):
        node_1 = edge[0]
        node_2 = edge[1]
        l_node_1_tokens = [token.strip() for token in node_1.split('|')[-1].split(' ')]
        l_node_2_tokens = [token.strip() for token in node_2.split('|')[-1].split(' ')]
        edge[2]['n1_ts'] = l_node_1_tokens
        edge[2]['n2_ts'] = l_node_2_tokens
    edit_paths, cost = nx.optimal_edit_paths(graph_src, graph_trg,
                                             node_subst_cost=cls_graph_comp_node_subst_cost,
                                             node_del_cost=cls_graph_comp_node_del_cost,
                                             node_ins_cost=cls_graph_comp_node_ins_cost,
                                             # edge_subst_cost=cls_graph_comp_edge_subst_cost,
                                             edge_del_cost=cls_graph_comp_edge_del_cost,
                                             edge_ins_cost=cls_graph_comp_edge_ins_cost)
    denom = 40 * (len(graph_src.nodes()) + len(graph_trg.nodes())) + len(graph_src.edges()) + len(graph_trg.edges())
    dist = cost / denom
    # logging.debug('ged=%s, edit_path=' % str(dist))
    # logging.debug(edit_paths)

    # recompute the distance between the two graphs only based on the POS matching over edges.
    # In other words, we firstly take the matched edges from the above computation without punishing any difference
    # of vertices on a pair of matched edges, and secondly compute the distance based on the difference of POS tags
    # of vertices on a pair of matched edges. More specifically, we stand at the graph_src side (i.e. the shuffled
    # sentence side) and see how likely the POS pair on an edge of graph_src can be wrong.
    # we only output the lowest distance case.
    d_diff_stats = {'ged': dist, 'posd': 0.0, 'edged': 0.0, 'pos_match': {}, 'pos_diff': {}, 'edge_match': {},
                    'edge_diff': {}}
    min_posd = 1000.0
    min_d_pos_match = {}
    min_d_pos_diff = {}
    min_edged = 1000.0
    min_d_edge_match = {}
    min_d_edge_diff = {}
    for edit_path_tup in edit_paths:
        posd = 0.0
        d_pos_match = dict()
        d_pos_diff = dict()
        edged = 0.0
        d_edge_match = dict()
        d_edge_diff = dict()
        edge_edit_path = edit_path_tup[1]
        for edge_edit_op in edge_edit_path:
            src_edge = edge_edit_op[0]
            trg_edge = edge_edit_op[1]
            if src_edge is None and trg_edge is None:
                raise Exception('Both None edge matching appears.')
            if src_edge is None:
                trg_n1_pos = graph_trg.nodes[trg_edge[0]]['pos']
                trg_n2_pos = graph_trg.nodes[trg_edge[1]]['pos']
                posd += 1
                if (trg_n1_pos, trg_n2_pos) in d_pos_diff:
                    d_pos_diff[(trg_n1_pos, trg_n2_pos)] += 1
                elif (trg_n2_pos, trg_n1_pos) in d_pos_diff:
                    d_pos_diff[(trg_n2_pos, trg_n1_pos)] += 1
                else:
                    d_pos_diff[(trg_n1_pos, trg_n2_pos)] = 1
                edged += 1
                if (trg_n1_pos, trg_n2_pos) in d_edge_diff:
                    d_edge_diff[(trg_n1_pos, trg_n2_pos)] += 1
                elif (trg_n2_pos, trg_n1_pos) in d_edge_diff:
                    d_edge_diff[(trg_n2_pos, trg_n1_pos)] += 1
                else:
                    d_edge_diff[(trg_n1_pos, trg_n2_pos)] = 1
                continue
            if trg_edge is None:
                src_n1_pos = graph_src.nodes[src_edge[0]]['pos']
                src_n2_pos = graph_src.nodes[src_edge[1]]['pos']
                posd += 1
                if (src_n1_pos, src_n2_pos) in d_pos_diff:
                    d_pos_diff[(src_n1_pos, src_n2_pos)] += 1
                elif (src_n2_pos, src_n1_pos) in d_pos_diff:
                    d_pos_diff[(src_n2_pos, src_n1_pos)] += 1
                else:
                    d_pos_diff[(src_n1_pos, src_n2_pos)] = 1
                edged += 1
                if (src_n1_pos, src_n2_pos) in d_edge_diff:
                    d_edge_diff[(src_n1_pos, src_n2_pos)] += 1
                elif (src_n2_pos, src_n1_pos) in d_edge_diff:
                    d_edge_diff[(src_n2_pos, src_n1_pos)] += 1
                else:
                    d_edge_diff[(src_n1_pos, src_n2_pos)] = 1
                continue

            trg_n1_pos = graph_trg.nodes[trg_edge[0]]['pos']
            trg_n2_pos = graph_trg.nodes[trg_edge[1]]['pos']
            src_n1_pos = graph_src.nodes[src_edge[0]]['pos']
            src_n2_pos = graph_src.nodes[src_edge[1]]['pos']

            if (src_n1_pos, src_n2_pos) in d_edge_match:
                d_edge_match[(src_n1_pos, src_n2_pos)] += 1
            elif (src_n2_pos, src_n1_pos) in d_edge_match:
                d_edge_match[(src_n2_pos, src_n1_pos)] += 1
            else:
                d_edge_match[(src_n1_pos, src_n2_pos)] = 1

            if (src_n1_pos == trg_n1_pos and src_n2_pos == trg_n2_pos) \
                    or (src_n1_pos == trg_n2_pos and src_n2_pos == trg_n1_pos):
                if (src_n1_pos, src_n2_pos) in d_pos_match:
                    d_pos_match[(src_n1_pos, src_n2_pos)] += 1
                elif (src_n2_pos, src_n1_pos) in d_pos_match:
                    d_pos_match[(src_n2_pos, src_n1_pos)] += 1
                else:
                    d_pos_match[(src_n1_pos, src_n2_pos)] = 1
            else:
                posd += 1
                if (src_n1_pos, src_n2_pos) in d_pos_diff:
                    d_pos_diff[(src_n1_pos, src_n2_pos)] += 1
                elif (src_n2_pos, src_n1_pos) in d_pos_diff:
                    d_pos_diff[(src_n2_pos, src_n1_pos)] += 1
                else:
                    d_pos_diff[(src_n1_pos, src_n2_pos)] = 1
        if posd < min_posd:
            min_posd = posd
            min_d_pos_diff = d_pos_diff
            min_d_pos_match = d_pos_match
        if edged < min_edged:
            min_edged = edged
            min_d_edge_diff = d_edge_diff
            min_d_edge_match = d_edge_match
    pos_demon = len(graph_src.edges()) + len(graph_trg.edges()) + 1
    min_posd = float(min_posd) / pos_demon
    min_edged = float(min_edged) / pos_demon
    d_diff_stats['posd'] = min_posd
    d_diff_stats['edged'] = min_edged
    d_diff_stats['pos_match'] = min_d_pos_match
    d_diff_stats['pos_diff'] = min_d_pos_diff
    d_diff_stats['edge_match'] = min_d_edge_match
    d_diff_stats['edge_diff'] = min_d_edge_diff
    return d_diff_stats


# def cls_graph_edit_distance_stat(graph_src, graph_trg):
#     '''
#     Graph edit distance between two core clause graphs, i.e. the cost of transforming graph_src to graph_trg.
#     Typically, graph_src is the graph for a shuffled sentence, and graph_trg is for the original sentence.
#     :param
#         graph_src: From
#         graph_trg: To
#     :return:
#         A dict of a set of stats.
#     '''
#     # match: an edge exists in both graph_src and graph_trg.
#     # diff: an edge exists in graph_src but not in graph_trg, though the two vertices are agreed in both graphs.
#     # del: an edge exists in graph_src but not in graph_trg, and either (of both) of the vertices is not in graph_trg.
#     d_stat = {'dis': 0, 'src_n_cnt': 0, 'trg_n_cnt': 0, 'src_e_cnt': 0, 'trg_e_cnt': 0, 'match': {}, 'diff':{}, 'del': {}}
#     d_stat['src_n_cnt'] = len(graph_src.nodes())
#     d_stat['trg_n_cnt'] = len(graph_trg.nodes())
#     d_stat['src_e_cnt'] = len(graph_src.edges())
#     d_stat['trg_e_cnt'] = len(graph_trg.edges())
#
#     trg_miss_cost = 0
#     src_miss_cost = 0
#     src_node_prefix = None
#     if len(graph_src.nodes()) > 0:
#         src_node_prefix_fields = graph_src.nodes()[0].split('|')
#         src_node_prefix = '|'.join(src_node_prefix_fields[:2])
#     trg_node_prefix = None
#     if len(graph_trg.nodes()) > 0:
#         trg_node_prefix_fields = graph_trg.nodes()[0].split('|')
#         trg_node_prefix = '|'.join(trg_node_prefix_fields[:2])
#     l_src_nodes = graph_src.nodes(data=True)
#     l_trg_nodes = graph_trg.nodes(data=True)
#     for edge in graph_src.edges(data=True):
#         src_node_1 = edge[0]
#         src_node_2 = edge[1]
#         src_node_1_fields = src_node_1.split('|')
#         src_node_2_fields = src_node_2.split('|')
#         src_node_1_pos = l_src_nodes[src_node_1]['pos']
#         src_node_2_pos = l_src_nodes[src_node_2]['pos']
#         if trg_node_prefix is not None:
#             trg_node_1 = trg_node_prefix + src_node_1_fields[2]
#             trg_node_2 = trg_node_prefix + src_node_2_fields[2]
#             trg_node_1_pos = l_trg_nodes[trg_node_1]['pos']
#             trg_node_2_pos = l_trg_nodes[trg_node_2]['pos']
#             if not graph_trg.has_edge(trg_node_1, trg_node_2):
#                 trg_miss_cost += 1
#                 # diff
#                 if trg_node_1 in l_trg_nodes and trg_node_2 in l_trg_nodes:
#                     if (trg_node_1_pos, trg_node_2_pos) not in d_stat['diff'] \
#                             and (trg_node_1_pos, trg_node_2_pos) not in d_stat['diff']:
#                         d_stat['diff'][(trg_node_1_pos, trg_node_2_pos)] = 1
#                     elif (trg_node_1_pos, trg_node_2_pos) in d_stat['diff']:
#                         d_stat['diff'][(trg_node_1_pos, trg_node_2_pos)] += 1
#                     elif (trg_node_2_pos, trg_node_1_pos) in d_stat['diff']:
#                         d_stat['diff'][(trg_node_2_pos, trg_node_1_pos)] += 1
#                 # del
#                 else:
#                     if (src_node_1_pos, src_node_2_pos) not in d_stat['del'] \
#                             and (src_node_2_pos, src_node_1_pos) not in d_stat['del']:
#                         d_stat['del'][(src_node_1_pos, src_node_2_pos)] = 1
#                     elif (src_node_1_pos, src_node_2_pos) in d_stat['del']:
#                         d_stat['del'][(src_node_1_pos, src_node_2_pos)] += 1
#                     elif (src_node_2_pos, src_node_1_pos) in d_stat['del']:
#                         d_stat['del'][(src_node_2_pos, src_node_1_pos)] += 1
#             # match
#             else:
#                 if (trg_node_1_pos, trg_node_2_pos) not in d_stat['match'] \
#                         and (trg_node_2_pos, trg_node_1_pos) not in d_stat['match']:
#                     d_stat['match'][(src_node_1_pos, src_node_2_pos)] = 1
#                 elif (trg_node_1_pos, trg_node_2_pos) in d_stat['match']:
#                     d_stat['match'][(trg_node_1_pos, trg_node_2_pos)] += 1
#                 elif (trg_node_2_pos, trg_node_1_pos) in d_stat['match']:
#                     d_stat['match'][(trg_node_2_pos, trg_node_1_pos)] += 1
#         else:
#             trg_miss_cost += 1
#             # del
#             if (src_node_1_pos, src_node_2_pos) not in d_stat['del'] \
#                     and (src_node_2_pos, src_node_1_pos) not in d_stat['del']:
#                 d_stat['del'][(src_node_1_pos, src_node_2_pos)] = 1
#             elif (src_node_1_pos, src_node_2_pos) in d_stat['del']:
#                 d_stat['del'][(src_node_1_pos, src_node_2_pos)] += 1
#             elif (src_node_2_pos, src_node_1_pos) in d_stat['del']:
#                 d_stat['del'][(src_node_2_pos, src_node_1_pos)] += 1
#
#     for edge in graph_trg.edges(data=True):
#         trg_node_1 = edge[0]
#         trg_node_2 = edge[1]
#         trg_node_1_fields = trg_node_1.split('|')
#         trg_node_2_fields = trg_node_2.split('|')
#         trg_node_1_pos = l_trg_nodes[trg_node_1]['pos']
#         trg_node_2_pos = l_trg_nodes[trg_node_2]['pos']
#         if src_node_prefix is not None:
#             src_node_1 = src_node_prefix + trg_node_1_fields[2]
#             src_node_2 = src_node_prefix + trg_node_2_fields[2]
#             if not graph_src.has_edge(src_node_1, src_node_2):
#                 src_miss_cost += 1
#                 # diff
#                 if src_node_1 in l_src_nodes and src_node_2 in l_src_nodes:
#                     if (trg_node_1_pos, trg_node_2_pos) not in d_stat['diff'] \
#                             and (trg_node_1_pos, trg_node_2_pos) not in d_stat['diff']:
#                         d_stat['diff'][(trg_node_1_pos, trg_node_2_pos)] = 1
#                     elif (trg_node_1_pos, trg_node_2_pos) in d_stat['diff']:
#                         d_stat['diff'][(trg_node_1_pos, trg_node_2_pos)] += 1
#                     elif (trg_node_2_pos, trg_node_1_pos) in d_stat['diff']:
#                         d_stat['diff'][(trg_node_2_pos, trg_node_1_pos)] += 1
#                 # del
#                 else:
#                     if (trg_node_1_pos, trg_node_2_pos) not in d_stat['del'] \
#                             and (trg_node_1_pos, trg_node_2_pos) not in d_stat['del']:
#                         d_stat['del'][(trg_node_1_pos, trg_node_2_pos)] = 1
#                     elif (trg_node_1_pos, trg_node_2_pos) in d_stat['del']:
#                         d_stat['del'][(trg_node_1_pos, trg_node_2_pos)] += 1
#                     elif (trg_node_2_pos, trg_node_1_pos) in d_stat['del']:
#                         d_stat['del'][(trg_node_2_pos, trg_node_1_pos)] += 1
#         else:
#             src_miss_cost += 1
#             # del
#             if (trg_node_1_pos, trg_node_2_pos) not in d_stat['del'] \
#                     and (trg_node_1_pos, trg_node_2_pos) not in d_stat['del']:
#                 d_stat['del'][(trg_node_1_pos, trg_node_2_pos)] = 1
#             elif (trg_node_1_pos, trg_node_2_pos) in d_stat['del']:
#                 d_stat['del'][(trg_node_1_pos, trg_node_2_pos)] += 1
#             elif (trg_node_2_pos, trg_node_1_pos) in d_stat['del']:
#                 d_stat['del'][(trg_node_2_pos, trg_node_1_pos)] += 1
#     return d_stat


def get_orig_en_sents(agent_ins):
    l_sel_spacy_sents = []
    en_db_conn = sqlite3.connect(g_ven_tw_db)
    en_db_cur = en_db_conn.cursor()
    en_sql_str = '''select raw_txt from ven_tw_en where tw_id = ?'''
    sent_cnt = 0
    good_sent_cnt = 0
    with open(g_ven_orig_en_tw_ids_path, 'r') as in_fd:
        with open(g_ven_good_sent_path, 'w+') as out_fd:
            for tw_id in in_fd:
                tw_id = tw_id.strip()
                en_db_cur.execute(en_sql_str, (tw_id,))
                tw_rec = en_db_cur.fetchone()
                if tw_rec is not None:
                    tw_raw_txt = tw_rec[0]
                    l_tw_clean_txt = agent_ins.text_clean(tw_raw_txt)
                    if l_tw_clean_txt is None:
                        if good_sent_cnt % 5000 == 0 and good_sent_cnt >= 5000:
                            logging.debug('%s good sentences.' % good_sent_cnt)
                        continue
                    for tw_clean_txt in l_tw_clean_txt:
                        l_spacy_sents, spacy_doc = agent_ins.spacy_pipeline_parse(tw_clean_txt)
                        for spacy_sent in l_spacy_sents:
                            sent_cnt += 1
                            if g_ven_sent_len_threshold_upper >= len(spacy_sent) >= g_ven_sent_len_threshold_lower:
                                l_sel_spacy_sents.append(spacy_sent)
                                out_fd.write(tw_id + '|' + spacy_sent.text.strip())
                                out_fd.write('\n')
                                good_sent_cnt += 1
                if good_sent_cnt % 5000 == 0 and good_sent_cnt >= 5000:
                    logging.debug('%s good sentences.' % good_sent_cnt)
            out_fd.close()
        in_fd.close()
    en_db_conn.close()
    logging.debug('%s good ones out of %s sentences.' % (good_sent_cnt, sent_cnt))
    return l_sel_spacy_sents


def sent_len_stat(agent_ins):
    l_len = []
    en_db_conn = sqlite3.connect(g_ven_tw_db)
    en_db_cur = en_db_conn.cursor()
    en_sql_str = '''select raw_txt from ven_tw_en where tw_id = ?'''
    with open(g_ven_orig_en_tw_ids_path, 'r') as in_fd:
        for tw_id in in_fd:
            tw_id = tw_id.strip()
            en_db_cur.execute(en_sql_str, (tw_id,))
            tw_rec = en_db_cur.fetchone()
            if tw_rec is None:
                logging.debug('%s does not exist.' % tw_id)
                continue
            tw_raw_txt = tw_rec[0]
            if tw_raw_txt is None:
                logging.debug('%s does not have raw_txt.' % tw_id)
                continue
            tw_clean_txt = agent_ins.text_clean(tw_raw_txt)
            if tw_clean_txt is None:
                continue
            l_spacy_sents, spacy_doc = agent_ins.spacy_pipeline_parse(tw_clean_txt)
            for spacy_sent in l_spacy_sents:
                l_len.append(len(spacy_sent))
        in_fd.close()
    en_db_conn.close()
    with open(g_ven_orig_en_tw_lens_path, 'w+') as out_fd:
        out_str = '\n'.join([str(item) for item in l_len])
        out_fd.write(out_str)
        out_fd.close()


def get_sent_task_list():
    s_already_done = set([])
    # for (dirpath, dirname, filenames) in walk(g_ven_sem_unit_folder):
    #     for filename in filenames:
    #         if filename[-14:] != '_cls_graph.gml':
    #             continue
    #         tw_id = filename.split('|')[0].strip()
    #         if len(tw_id) != 22:
    #             print(tw_id)
    #         s_already_done.add(tw_id)
    l_sent_task_list = []
    with open(g_ven_good_sent_path, 'r') as in_fd:
        for ln in in_fd:
            fields = ln.split('|')
            tw_id = fields[0].strip()
            if tw_id in s_already_done:
                continue
            sent_txt = fields[1].strip()
            l_sent_task_list.append((tw_id, sent_txt))
        in_fd.close()
    logging.debug('%s sent tasks.' % len(l_sent_task_list))
    return l_sent_task_list


def indexing_good_sents():
    sent_id = 0
    with open(g_cls_comp_sent_path, 'w+') as out_fd:
        with open(g_ven_good_sent_path, 'r') as in_fd:
            for ln in in_fd:
                fields = [item.strip() for item in ln.split('|')]
                sent = fields[1]
                new_ln = str(sent_id) + '|' + sent
                out_fd.write(new_ln)
                out_fd.write('\n')
                sent_id += 1
            in_fd.close()
        out_fd.close()


def get_cls_comparison_tasks():
    d_tasks = dict()
    with open(g_cls_comp_sent_path, 'r') as in_fd:
        for ln in in_fd:
            fields = [item.strip() for item in ln.split('|')]
            task_id = fields[0]
            orig_sent = fields[1]
            l_shuf_sents = sent_shuffle(orig_sent, 10)
            if task_id not in d_tasks:
                d_tasks[task_id] = {'orig': orig_sent, 'shuf_sents': l_shuf_sents}
        in_fd.close()
    with open(g_cls_comp_tasks_path, 'w+') as out_fd:
        json.dump(d_tasks, out_fd)
        out_fd.close()


def load_cls_comp_tasks():
    with open(g_cls_comp_tasks_path, 'r') as out_fd:
        d_tasks = json.load(out_fd)
        out_fd.close()
    return d_tasks


def process_one_cls_comp_task(agent_ins, task_id, orig_sent, l_shuf_sents):
    timer_start = time.time()
    d_diff_sum = dict()
    output_folder = g_cls_comp_folder + task_id + '/'
    if not path.exists(output_folder):
        os.mkdir(output_folder)
    orig_cls_graph, _ = agent_ins.extract_sem_units_from_text(orig_sent, task_id)
    if orig_cls_graph is None:
        return
    nx.write_gml(orig_cls_graph, output_folder + task_id + '_orig.gml')
    # agent_ins.output_nx_graph(orig_cls_graph, output_folder + task_id + '_orig.png')
    l_ged = []
    l_posd = []
    l_edged = []
    d_pos_match = dict()
    d_pos_diff = dict()
    d_edge_match = dict()
    d_edge_diff = dict()
    for sent_id, shuf_sent in enumerate(l_shuf_sents):
        shuf_task_id = task_id + '#' + str(sent_id)
        shuf_cls_graph, _ = agent_ins.extract_sem_units_from_text(shuf_sent, shuf_task_id)
        if shuf_cls_graph is None:
            continue
        nx.write_gml(shuf_cls_graph, output_folder + shuf_task_id + '_shuf.gml')
        # agent_ins.output_nx_graph(shuf_cls_graph, output_folder + shuf_task_id + '_shuf.png')
        d_diff_stat = cls_graph_edit_distance_stat(shuf_cls_graph, orig_cls_graph)
        d_diff_stat_out = {'ged': d_diff_stat['ged'], 'posd': d_diff_stat['posd'], 'edged': d_diff_stat['edged'],
                           'pos_match': {key[0] + '_' + key[1]: d_diff_stat['pos_match'][key] for key in
                                         d_diff_stat['pos_match']},
                           'pos_diff': {key[0] + '_' + key[1]: d_diff_stat['pos_diff'][key] for key in
                                        d_diff_stat['pos_diff']},
                           'edge_match': {key[0] + '_' + key[1]: d_diff_stat['edge_match'][key] for key in
                                          d_diff_stat['edge_match']},
                           'edge_diff': {key[0] + '_' + key[1]: d_diff_stat['edge_diff'][key] for key in
                                         d_diff_stat['edge_diff']}}
        with open(output_folder + shuf_task_id + '_diff_stats.json', 'w+') as out_fd:
            json.dump(d_diff_stat_out, out_fd)
            out_fd.close()
        l_ged.append(d_diff_stat['ged'])
        l_posd.append(d_diff_stat['posd'])
        l_edged.append(d_diff_stat['edged'])
        for pos_pair in d_diff_stat['pos_match']:
            pos_1 = pos_pair[0]
            pos_2 = pos_pair[1]
            if pos_pair in d_pos_match:
                d_pos_match[pos_pair] += 1
            elif (pos_2, pos_1) in d_pos_match:
                d_pos_match[(pos_2, pos_1)] += 1
            else:
                d_pos_match[pos_pair] = 1
        for pos_pair in d_diff_stat['pos_diff']:
            pos_1 = pos_pair[0]
            pos_2 = pos_pair[1]
            if pos_pair in d_pos_diff:
                d_pos_diff[pos_pair] += 1
            elif (pos_2, pos_1) in d_pos_diff:
                d_pos_diff[(pos_2, pos_1)] += 1
            else:
                d_pos_diff[pos_pair] = 1
        for pos_pair in d_diff_stat['edge_match']:
            pos_1 = pos_pair[0]
            pos_2 = pos_pair[1]
            if pos_pair in d_edge_match:
                d_edge_match[pos_pair] += 1
            elif (pos_2, pos_1) in d_edge_match:
                d_edge_match[(pos_2, pos_1)] += 1
            else:
                d_edge_match[pos_pair] = 1
        for pos_pair in d_diff_stat['edge_diff']:
            pos_1 = pos_pair[0]
            pos_2 = pos_pair[1]
            if pos_pair in d_edge_diff:
                d_edge_diff[pos_pair] += 1
            elif (pos_2, pos_1) in d_edge_diff:
                d_edge_diff[(pos_2, pos_1)] += 1
            else:
                d_edge_diff[pos_pair] = 1
    with open(output_folder + task_id + '_ged.txt', 'w+') as out_fd:
        out_str = '\n'.join([str(item) for item in l_ged])
        out_fd.write(out_str)
        out_fd.close()
    with open(output_folder + task_id + '_posd.txt', 'w+') as out_fd:
        out_str = '\n'.join([str(item) for item in l_posd])
        out_fd.write(out_str)
        out_fd.close()
    with open(output_folder + task_id + '_edged.txt', 'w+') as out_fd:
        out_str = '\n'.join([str(item) for item in l_edged])
        out_fd.write(out_str)
        out_fd.close()
    d_pos_match_out = {key[0] + '_' + key[1]: d_pos_match[key] for key in d_pos_match}
    with open(output_folder + task_id + '_pos_match.json', 'w+') as out_fd:
        json.dump(d_pos_match_out, out_fd)
        out_fd.close()
    d_pos_diff_out = {key[0] + '_' + key[1]: d_pos_diff[key] for key in d_pos_diff}
    with open(output_folder + task_id + '_pos_diff.json', 'w+') as out_fd:
        json.dump(d_pos_diff_out, out_fd)
        out_fd.close()
    d_edge_match_out = {key[0] + '_' + key[1]: d_edge_match[key] for key in d_edge_match}
    with open(output_folder + task_id + '_edge_match.json', 'w+') as out_fd:
        json.dump(d_edge_match_out, out_fd)
        out_fd.close()
    d_edge_diff_out = {key[0] + '_' + key[1]: d_edge_diff[key] for key in d_edge_diff}
    with open(output_folder + task_id + '_edge_diff.json', 'w+') as out_fd:
        json.dump(d_edge_diff_out, out_fd)
        out_fd.close()
    logging.debug('Task: %s is done in % secs.' % (task_id, str(time.time() - timer_start)))


def process_cls_comp_tasks_thread(agent_ins, l_tasks, thread_id):
    for task_id, d_task in l_tasks:
        orig_sent = d_task['orig']
        l_shuf_sents = d_task['shuf_sents']
        process_one_cls_comp_task(agent_ins, task_id, orig_sent, l_shuf_sents)
    logging.debug('Thread: %s with %s tasks is done.' % (thread_id, str(len(l_tasks))))


def cls_comp_multithreads(op_func, l_tasks, num_threads, agent_ins):
    timer_1 = time.time()
    batch_size = math.ceil(len(l_tasks) / num_threads)
    l_l_subtasks = []
    for i in range(0, len(l_tasks), batch_size):
        if i + batch_size < len(l_tasks):
            l_l_subtasks.append(l_tasks[i:i + batch_size])
        else:
            l_l_subtasks.append(l_tasks[i:])
    logging.debug('%s threads.' % len(l_l_subtasks))

    l_threads = []
    t_id = 0
    for l_each_batch in l_l_subtasks:
        t = threading.Thread(target=op_func, args=(agent_ins, l_each_batch, 't_mul_task_' + str(t_id)))
        t.setName('t_mul_task_' + str(t_id))
        t.start()
        l_threads.append(t)
        t_id += 1

    while len(l_threads) > 0:
        for t in l_threads:
            if t.is_alive():
                t.join(1)
            else:
                l_threads.remove(t)
                logging.debug('Thread %s is finished.' % t.getName())

    logging.debug('All done in %s sec for %s tasks.' % (time.time() - timer_1, len(l_tasks)))


def cls_comp_stats():
    l_avg_ged = []
    l_avg_posd = []
    l_avg_edged = []
    d_sum_pos_match = {}
    d_sum_pos_diff = {}
    d_sum_edge_match = {}
    d_sum_edge_diff = {}
    for (dirpath, dirname, filenames) in walk(g_cls_comp_folder):
        for filename in filenames:
            if filename[-8:] == '_ged.txt':
                l_cur_ged = []
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for ln in in_fd:
                        l_cur_ged.append(float(ln))
                    in_fd.close()
                if len(l_cur_ged) > 0:
                    cur_avg_ged = sum(l_cur_ged) / len(l_cur_ged)
                    l_avg_ged.append(cur_avg_ged)
            elif filename[-9:] == '_posd.txt':
                l_cur_posd = []
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for ln in in_fd:
                        l_cur_posd.append(float(ln))
                    in_fd.close()
                if len(l_cur_posd) > 0:
                    cur_avg_posd = sum(l_cur_posd) / len(l_cur_posd)
                    l_avg_posd.append(cur_avg_posd)
            elif filename[-10:] == '_edged.txt':
                l_cur_edged = []
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    for ln in in_fd:
                        l_cur_edged.append(float(ln))
                    in_fd.close()
                if len(l_cur_edged) > 0:
                    cur_avg_edged = sum(l_cur_edged) / len(l_cur_edged)
                    l_avg_edged.append(cur_avg_edged)
            elif filename[-15:] == '_pos_match.json':
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    pos_match_json = json.load(in_fd)
                    in_fd.close()
                for key in pos_match_json:
                    if key not in d_sum_pos_match:
                        d_sum_pos_match[key] = pos_match_json[key]
                    else:
                        d_sum_pos_match[key] += pos_match_json[key]
            elif filename[-14:] == '_pos_diff.json':
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    pos_diff_json = json.load(in_fd)
                    in_fd.close()
                for key in pos_diff_json:
                    if key not in d_sum_pos_diff:
                        d_sum_pos_diff[key] = pos_diff_json[key]
                    else:
                        d_sum_pos_diff[key] += pos_diff_json[key]
            elif filename[-16:] == '_edge_match.json':
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    edge_match_json = json.load(in_fd)
                    in_fd.close()
                for key in edge_match_json:
                    if key not in d_sum_edge_match:
                        d_sum_edge_match[key] = edge_match_json[key]
                    else:
                        d_sum_edge_match[key] += edge_match_json[key]
            elif filename[-15:] == '_edge_diff.json':
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    edge_diff_json = json.load(in_fd)
                    in_fd.close()
                for key in edge_diff_json:
                    if key not in d_sum_edge_diff:
                        d_sum_edge_diff[key] = edge_diff_json[key]
                    else:
                        d_sum_edge_diff[key] += edge_diff_json[key]

    with open(g_ven_tw_folder + 'cls_comp_stat_avg_ged.txt', 'w+') as out_fd:
        out_str = '\n'.join([str(item) for item in l_avg_ged])
        out_fd.write(out_str)
        out_fd.close()
    with open(g_ven_tw_folder + 'cls_comp_stat_avg_posd.txt', 'w+') as out_fd:
        out_str = '\n'.join([str(item) for item in l_avg_posd])
        out_fd.write(out_str)
        out_fd.close()
    with open(g_ven_tw_folder + 'cls_comp_stat_avg_edged.txt', 'w+') as out_fd:
        out_str = '\n'.join([str(item) for item in l_avg_edged])
        out_fd.write(out_str)
        out_fd.close()
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_pos_match.json', 'w+') as out_fd:
        json.dump(d_sum_pos_match, out_fd)
        out_fd.close()
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_pos_diff.json', 'w+') as out_fd:
        json.dump(d_sum_pos_diff, out_fd)
        out_fd.close()
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_edge_match.json', 'w+') as out_fd:
        json.dump(d_sum_edge_match, out_fd)
        out_fd.close()
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_edge_diff.json', 'w+') as out_fd:
        json.dump(d_sum_edge_diff, out_fd)
        out_fd.close()


def merge_pos_diff_match():
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_pos_match.json', 'r') as in_fd:
        d_pos_match = json.load(in_fd)
        in_fd.close()
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_pos_diff.json', 'r') as in_fd:
        d_pos_diff = json.load(in_fd)
        in_fd.close()
    d_pos_sum = dict()
    for key in d_pos_diff:
        if key not in d_pos_sum:
            d_pos_sum[key] = [d_pos_diff[key], 0]
        else:
            d_pos_sum[key][0] += d_pos_diff[key]
    for key in d_pos_match:
        key_fields = [item.strip() for item in key.split('_')]
        reversed_key = key_fields[1] + '_' + key_fields[0]
        if key in d_pos_sum:
            d_pos_sum[key][1] += d_pos_match[key]
        else:
            if reversed_key in d_pos_sum:
                d_pos_sum[reversed_key][1] += d_pos_match[key]
            else:
                d_pos_sum[key] = [0, d_pos_match[key]]
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_pos_diff_match.json', 'w+') as out_fd:
        for key in d_pos_sum:
            out_str = key + '|' + str(d_pos_sum[key][0]) + '|' + str(d_pos_sum[key][1])
            out_fd.write(out_str)
            out_fd.write('\n')
        out_fd.close()


def merge_edge_diff_match():
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_edge_match.json', 'r') as in_fd:
        d_edge_match = json.load(in_fd)
        in_fd.close()
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_edge_diff.json', 'r') as in_fd:
        d_edge_diff = json.load(in_fd)
        in_fd.close()
    d_edge_sum = dict()
    for key in d_edge_diff:
        if key not in d_edge_sum:
            d_edge_sum[key] = [d_edge_diff[key], 0]
        else:
            d_edge_sum[key][0] += d_edge_diff[key]
    for key in d_edge_match:
        key_fields = [item.strip() for item in key.split('_')]
        reversed_key = key_fields[1] + '_' + key_fields[0]
        if key in d_edge_sum:
            d_edge_sum[key][1] += d_edge_match[key]
        else:
            if reversed_key in d_edge_sum:
                d_edge_sum[reversed_key][1] += d_edge_match[key]
            else:
                d_edge_sum[key] = [0, d_edge_match[key]]
    with open(g_ven_tw_folder + 'cls_comp_stat_sum_edge_diff_match.json', 'w+') as out_fd:
        for key in d_edge_sum:
            out_str = key + '|' + str(d_edge_sum[key][0]) + '|' + str(d_edge_sum[key][1])
            out_fd.write(out_str)
            out_fd.write('\n')
        out_fd.close()


def get_full_sem_unit_tasks():
    en_db_conn = sqlite3.connect(g_ven_tw_db)
    en_db_cur = en_db_conn.cursor()
    en_sql_str = '''select tw_id, raw_txt from ven_tw_en'''
    en_db_cur.execute(en_sql_str)
    l_tw_recs = en_db_cur.fetchall()
    l_txt_tasks = []
    total_cnt = 0
    eff_cnt = 0
    for tw_rec in l_tw_recs:
        total_cnt += 1
        tw_id = tw_rec[0]
        tw_raw_txt = tw_rec[1]
        if tw_raw_txt is None or tw_raw_txt == '':
            continue
        l_txt_tasks.append((tw_id, tw_raw_txt))
        eff_cnt += 1
    logging.debug('%s effective tw tasks out of %s tws.' % (eff_cnt, total_cnt))
    return l_txt_tasks


def get_full_sem_unit_task_20news_bbc_reuters(data_name):
    db_folder = '/home/mf3jh/workspace/data/docsim/'
    if data_name == '20news50short10' or 'bbc' or 'reuters':
        db_path = db_folder + data_name + '.db'
    else:
        return None
    db_conn = sqlite3.connect(db_path)
    db_cur = db_conn.cursor()
    sql_str = '''select doc_id, pre_ner from docs'''
    db_cur.execute(sql_str)
    l_rec = db_cur.fetchall()
    l_tasks = []
    for rec in l_rec:
        doc_id = rec[0]
        doc_txt = rec[1]
        if data_name == '20news50short10':
            doc_id = re.sub(r'/', '_', doc_id)
        l_tasks.append((doc_id, doc_txt))
    db_conn.close()
    logging.debug('%s tasks in %s.' % (len(l_tasks), data_name))
    return l_tasks


def pos_pair_stats(sem_unit_folder, pos_pair_stats_out_folder):
    # pos_pair_stats_out_folder = g_ven_tw_folder + 'pos_pair_stats/'
    d_pos_percents = dict()
    d_pos_cnt = dict()
    file_cnt = 0
    rec_cnt = 0
    timer_start = time.time()
    for (dirpath, dirname, filenames) in walk(sem_unit_folder):
        for filename in filenames:
            if filename[-14:] == '_cls_graph.gml':
                file_cnt += 1
                cls_graph = nx.read_gml(dirpath + '/' + filename)
                for comp in nx.connected_components(cls_graph):
                    sent_cls_graph = nx.subgraph(cls_graph, comp)
                    edge_cnt = len(sent_cls_graph.edges())
                    if edge_cnt <= 0:
                        continue
                    rec_cnt += 1
                    d_pos_stats = dict()
                    for edge in sent_cls_graph.edges():
                        node_1 = edge[0]
                        node_2 = edge[1]
                        node_1_pos = sent_cls_graph.nodes(data=True)[node_1]['pos']
                        node_2_pos = sent_cls_graph.nodes(data=True)[node_2]['pos']
                        if (node_1_pos + '_' + node_2_pos not in d_pos_stats) \
                                and (node_2_pos + '_' + node_1_pos not in d_pos_stats):
                            if (node_1_pos + '_' + node_2_pos not in d_pos_percents) \
                                    and (node_2_pos + '_' + node_1_pos not in d_pos_percents):
                                d_pos_percents[node_1_pos + '_' + node_2_pos] = []
                                d_pos_cnt[node_1_pos + '_' + node_2_pos] = 0
                                d_pos_stats[node_1_pos + '_' + node_2_pos] = 1
                            elif node_1_pos + '_' + node_2_pos in d_pos_percents:
                                d_pos_stats[node_1_pos + '_' + node_2_pos] = 1
                            elif node_2_pos + '_' + node_1_pos in d_pos_percents:
                                d_pos_stats[node_2_pos + '_' + node_1_pos] = 1
                        elif node_1_pos + '_' + node_2_pos in d_pos_stats:
                            d_pos_stats[node_1_pos + '_' + node_2_pos] += 1
                        elif node_2_pos + '_' + node_1_pos in d_pos_stats:
                            d_pos_stats[node_2_pos + '_' + node_1_pos] += 1
                    for pos_pair in d_pos_stats:
                        if d_pos_stats[pos_pair] > edge_cnt:
                            raise Exception('POS stat goes wrong for %s.' % filename)
                        d_pos_percents[pos_pair].append(float(d_pos_stats[pos_pair]) / edge_cnt)
                        d_pos_cnt[pos_pair] += d_pos_stats[pos_pair]
                if rec_cnt % 1000 == 0 and rec_cnt >= 1000:
                    logging.debug('%s out of %s files have been done.' % (rec_cnt, file_cnt))
    logging.debug('%s out of %s files have been done.' % (rec_cnt, file_cnt))
    for pos_pair in d_pos_percents:
        with open(pos_pair_stats_out_folder + 'pos_percent_' + pos_pair + '.txt', 'w+') as out_fd:
            out_str = '\n'.join([str(item) for item in d_pos_percents[pos_pair]])
            out_fd.write(out_str)
            out_fd.close()
    logging.debug('pos_percent files are written.')
    l_pos_cnt_lns = []
    for pos_pair in d_pos_cnt:
        out_ln = pos_pair + '|' + str(d_pos_cnt[pos_pair])
        l_pos_cnt_lns.append(out_ln)
    pos_cnt_out_str = '\n'.join(l_pos_cnt_lns)
    with open(pos_pair_stats_out_folder + 'pos_cnt.txt', 'w+') as out_fd:
        out_fd.write(pos_cnt_out_str)
        out_fd.close()
    logging.debug('pos_cnt file is written.')


def draw_pos_percent_dists():
    pos_pair_stats_out_folder = g_ven_tw_folder + 'pos_pair_stats/'
    plt.figure(1, figsize=(15, 15), tight_layout={'pad': 1, 'w_pad': 100, 'h_pad': 100, 'rect': None})
    for (dirpath, dirname, filenames) in walk(pos_pair_stats_out_folder):
        for filename in filenames:
            if filename[:12] == 'pos_percent_' and filename[-4:] == '.txt':
                plot_name = filename[12:-4]
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    l_data = []
                    for ln in in_fd:
                        data_str = ln.strip()
                        if data_str != '':
                            l_data.append(float(data_str))
                    in_fd.close()
                mean = np.mean(l_data)
                std = np.mean(l_data)
                x = np.linspace(0.0, 1.0, 500)
                dist = scipy.stats.norm.cdf(x, mean, std)
                plt.plot(x, dist, label=plot_name)
                # plt.show()
                # plt.grid()
                # plt.savefig(pos_pair_stats_out_folder + filename[:-4] + '.png', format="PNG")
                # plt.clf()
    plt.grid()
    # plt.legend()
    plt.show()


def pos_percent_means(pos_pair_stats_out_folder):
    # pos_pair_stats_out_folder = g_ven_tw_folder + 'pos_pair_stats/'
    l_out = []
    for (dirpath, dirname, filenames) in walk(pos_pair_stats_out_folder):
        for filename in filenames:
            if filename[:12] == 'pos_percent_' and filename[-4:] == '.txt':
                pos_name = filename[12:-4]
                with open(dirpath + '/' + filename, 'r') as in_fd:
                    l_data = []
                    for ln in in_fd:
                        data_str = ln.strip()
                        if data_str != '':
                            l_data.append(float(data_str))
                    in_fd.close()
                mean = np.mean(l_data)
                l_out.append(pos_name + '|' + str(mean))
    with open(pos_pair_stats_out_folder + 'pos_percentage_means.txt', 'w+') as out_fd:
        out_fd.write('\n'.join(l_out))
        out_fd.close()


def draw_cls_graphs_20news_bbc_reuters(data_name, agent_ins):
    cls_graph_folder = '/home/mf3jh/workspace/data/docsim/sem_units_' + data_name + '/'
    for (dirpath, dirname, filenames) in walk(cls_graph_folder):
        for filename in filenames:
            if filename[-14:] == '_cls_graph.gml':
                outpath = cls_graph_folder + filename[:-4] + '.png'
                cls_graph = nx.read_gml(dirpath + '/' + filename)
                agent_ins.output_nx_graph(cls_graph, outpath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    agent_ins = NarrativeAgent('agent_config.conf')
    # load_lexvec_model()
    # external_spacy_init('en_core_web_lg')

    # d_tw, picked_d_tw, l_ordered_tw = read_cascades()
    # for tw_id in l_ordered_tw:
    #     l_picked_children = []
    #     for child_id in picked_d_tw[tw_id]['children']:
    #         # child_id = list(child.keys())[0]
    #         if len(d_tw[child_id]['children']) > 0:
    #             l_picked_children.append(child_id)
    #     picked_d_tw[tw_id]['children'] = sorted(l_picked_children, key=lambda k: len(d_tw[k]['children']))
    #     if len(picked_d_tw[tw_id]['children']) <= 0:
    #         del picked_d_tw[tw_id]
    # l_ordered_tw = sorted(picked_d_tw.keys(), key=lambda k: len(picked_d_tw[k]['children']), reverse=True)
    #
    # nx_cas = nx.DiGraph()
    # for tw_id in picked_d_tw:
    #     if tw_id not in nx_cas:
    #         nx_cas.add_node(tw_id)
    #     for child_id in picked_d_tw[tw_id]['children']:
    #         nx_cas.add_edge(tw_id, child_id)
    #         for grand_id in d_tw[child_id]['children']:
    #             # grand_id = list(grand.keys())[0]
    #             nx_cas.add_edge(child_id, grand_id)
    # # nx.write_gml(nx_cas, g_tw_cascade_graph_path)
    # # draw_ig_graph(g_tw_cascade_graph_path, g_tw_cascade_graph_img_path)
    #
    # d_train = dict()
    # d_nps = dict()
    # q_nodes = deque()
    # q_nodes.append(g_tw_cascade_root)
    # while len(q_nodes) > 0:
    #     cur = q_nodes.pop()
    #     for child in nx_cas.successors(cur):
    #         q_nodes.append(child)
    #     cur_txt = agent_ins.text_clean(d_tw[cur]['txt'])
    #     d_train[cur] = {'sup': d_tw[cur]['sup'], 'kwd': d_tw[cur]['kwd'], 'man': d_tw[cur]['man']}
    #     l_spacy_sents, spacy_doc = agent_ins.spacy_pipeline_parse(cur_txt)
    #     l_nps = []
    #     nx_cls = nx.Graph()
    #     for spacy_sent in l_spacy_sents:
    #         l_nps += agent_ins.spacy_extract_nps_from_sent(spacy_sent)
    #         # nx_cls = union(nx_cls, agent_ins.extract_cls_from_sent(spacy_sent, l_nps))
    #         nx_cls = union_nx_graphs(nx_cls, agent_ins.extract_cls_from_sent(spacy_sent, l_nps))
    #     if len(nx_cls.nodes) > 0:
    #         nx.write_gml(nx_cls, g_tw_cascade_core_cls_graph_format.format(cur))
    #         draw_ig_graph(g_tw_cascade_core_cls_graph_format.format(cur),
    #                       g_tw_cascade_core_cls_graph_img_format.format(cur, 0))
    #     d_nps[cur] = l_nps
    # with open(g_tw_cascade_train, 'w+') as out_fd:
    #     json.dump(d_train, out_fd)
    #     out_fd.close()
    # with open(g_tw_cascade_nps, 'w+') as out_fd:
    #     json.dump(d_nps, out_fd)
    #     out_fd.close()

    ##########################################
    # testing
    ##########################################
    # tw_id = 'g_test_tw_8'
    # l_clean_sents = agent_ins.text_clean('''take the responsibility to protect their people from radical Palestinian elements who are opposed to the peace process, is reprehensible.''')
    # for clean_sent_txt in l_clean_sents:
    #     l_spacy_sents, spacy_doc = agent_ins.spacy_pipeline_parse(clean_sent_txt)
        # sent_id = 0
        # nx_cls = nx.Graph()
        # for sent_id, spacy_sent in enumerate(l_spacy_sents):
        #     l_nps = agent_ins.spacy_extract_nps_from_sent(spacy_sent)
        #     # agent_ins.spacy_dep_parse_tree_to_nx_graph(spacy_sent)
        #     # agent_ins.extract_cls_and_mdf_from_sent(spacy_sent)
        #     nx_cls = agent_ins.union_nx_graphs(nx_cls,
        #                                        agent_ins.extract_cls_from_sent(tw_id + '|' + str(sent_id),
        #                                                                        spacy_sent, l_nps))
        #     # nx.write_gml(nx_cls, g_core_cls_graph_format.format(tw_id, sent_id))
        #     # draw_ig_graph(g_core_cls_graph_format.format(tw_id, sent_id),
        #     #               g_core_cls_graph_img_format.format(tw_id, sent_id))
        #     sent_id += 1
        #     agent_ins.output_nx_graph(nx_cls, 'test_fig.png')

    # cls_graph_path = '/home/mf3jh/workspace/data/cp4_challenge/Venezuela/Twitter/cascades/-Jhlv1hgSlEe8RTFaKGCTA/3IL8MdIrpnbrz7SEwJycCQ_cls_graph.gml'
    # cls_graph = nx.read_gml(cls_graph_path)
    # for node in cls_graph.nodes(data=True):
    #     node[1]['label'] = node[0].split('|')[2].strip()
    # agent_ins.output_nx_graph(cls_graph, 'nCTpXnX6yE5MkSSgnqQZCQ_cls_graph.png')

    ##########################################
    # For COVID-19 Twitter Data
    ##########################################
    # covid_extract_tw_sem_units(agent_ins)
    # agent_ins.sem_unit_stats(g_covid_tw_salient_cascades_semantic_units_folder,
    #                          '/home/mf3jh/workspace/cp4_narratives/covid19')
    # covid_resp_extract_tw_sem_units(agent_ins)

    ##########################################
    # For CP4 Venezeula Twitter Data
    ##########################################
    # ven_extract_tw_sem_units(agent_ins)
    # ven_sem_units_multithreads(ven_tw_sem_units_thread)
    # agent_ins.task_multithreads(agent_ins.sem_unit_extraction_thread,
    #                             get_full_sem_unit_tasks(),
    #                             10,
    #                             g_ven_sem_unit_folder)
    # pos_pair_stats()
    # draw_pos_percent_dists()
    # pos_percent_means()

    ##########################################
    # For 20news50short10, bbc, reuters
    ##########################################
    # data_name = '20news50short10'
    # sem_units_folder = '/home/mf3jh/workspace/data/docsim/sem_units_' + data_name + '/'
    # agent_ins.task_multithreads(agent_ins.sem_unit_extraction_thread,
    #                             get_full_sem_unit_task_20news_bbc_reuters(data_name),
    #                             10,
    #                             sem_units_folder)
    # pos_pair_stats_folder = '/home/mf3jh/workspace/data/docsim/pos_pair_stats_' + data_name + '/'
    # pos_pair_stats(sem_units_folder, pos_pair_stats_folder)
    # pos_percent_means(pos_pair_stats_folder)
    # draw_cls_graphs_20news_bbc_reuters(data_name, agent_ins)

    ##########################################
    # For Semantic Unit Verification
    ##########################################
    # sent_len_stat(agent_ins)
    # then we get a reasonable min (and max ?) sentence length to cut off, and obtain a sample sentence dataset.
    # get_orig_en_sents(agent_ins)
    # agent_ins.task_multithreads(agent_ins.sem_unit_extraction_thread, get_sent_task_list(), 10, g_ven_tw_folder + 'sem_units_new/')
    # agent_ins.sem_unit_stats(g_ven_tw_folder + 'sem_units_new/', g_ven_tw_folder)
    # indexing_good_sents()
    # get_cls_comparison_tasks()
    # agent_ins.sem_unit_stats(g_ven_sem_unit_folder, g_ven_tw_folder)
    # cls_comp_stats()
    # merge_pos_diff_match()
    # merge_edge_diff_match()

    # cls comp tasks
    # d_cls_comp_tasks = load_cls_comp_tasks()
    # logging.debug('%s cls comp tasks in total.' % len(d_cls_comp_tasks))
    # l_cls_comp_tasks = [(key, d_cls_comp_tasks[key]) for key in d_cls_comp_tasks]
    # cls_comp_multithreads(process_cls_comp_tasks_thread, l_cls_comp_tasks, 6, agent_ins)

    # for task_id in d_cls_comp_tasks:
    #     orig_sent = d_cls_comp_tasks[task_id]['orig']
    #     l_shuf_sents = d_cls_comp_tasks[task_id]['shuf_sents']
    #     process_one_cls_comp_task(agent_ins, task_id, orig_sent, l_shuf_sents)

    # test for cls_graph_comp
    # shuf_cls_graph = nx.read_gml(g_cls_comp_folder + '0/0#3_shuf.gml')
    # orig_cls_graph = nx.read_gml(g_cls_comp_folder + '0/0_orig.gml')
    # d_diff_stat = cls_graph_edit_distance_stat(shuf_cls_graph, orig_cls_graph)

    # spacy_init()
    # # allennlp_load_elmo()
    # load_stopwords()
    # # allennlp_elmo_test(g_test_tw_1)
    # d_sal_nps = dict()
    # for tw_text in g_l_test_tws:
    #     d_each_sal_nps = extract_nps_n_vects_from_text(tw_text)
    #     d_sal_nps.update(d_each_sal_nps)
    # pairwise_np_comparison(d_sal_nps)

# allennlp_coref(g_test_text)

# neuralcoref_parse(g_test_text)

# l_sents, parsed_doc = spacy_pipeline(g_test_text)
# for sent in l_sents:
#     extract_semantic_units_from_spacy_dep_parse_tree(sent)


# allennlp_dep_parse('The Coronavirus was leaked on purpose to stop the protests.')
