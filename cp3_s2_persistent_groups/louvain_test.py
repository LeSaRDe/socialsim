# import community
import igraph as ig
import louvain
import networkx as nx
import json
# import matplotlib.pyplot as plt
import logging

# NOTE:
# run this file from console!!!

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
# G = nx.erdos_renyi_graph(30, 0.05)

logging.basicConfig(level=logging.DEBUG)

g_fsgraph_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/20180415_20180422/tw_wh_normfsgraph_20180415_20180422_full.json'
g_fsgraph_gml_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/20180415_20180422/tw_wh_normfsgraph_20180415_20180422_full.gml'
g_fsgraph_pg_img_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/sampled_data/updated_data/20180819_20180826/tw_wh_normfsgraph_20180819_20180826_full_pg_8k.pdf'

# g_fsgraph_gml_path = '/home/mf3jh/workspace/data/white_helmet/White_Helmet/Twitter/Tng_an_WH_Twitter_v2_friendship_graph_norm.gml'

with open(g_fsgraph_path, 'r') as in_fd:
    fs_graph_data = json.load(in_fd)
    fs_graph = nx.adjacency_graph(fs_graph_data)
    nx.write_gml(fs_graph, g_fsgraph_gml_path)
    fs_ig = ig.load(g_fsgraph_gml_path)
    in_fd.close()

#first compute the best partition
# partition = community.best_partition(fs_graph)
partition = louvain.find_partition(fs_ig, louvain.ModularityVertexPartition)
logging.debug('partition is done.')
ig.plot(partition, g_fsgraph_pg_img_path, bbox=(7680, 7680), vertex_label_size=0)

# ig.plot(partition, g_fsgraph_pg_img_path, bbox=(7680, 7680), vertex_label_size=50, margin=400)

#drawing
# size = float(len(set(partition.values())))
# pos = nx.spring_layout(fs_graph)
# count = 0.
# for com in set(partition.values()):
#     count = count + 1.
#     list_nodes = [nodes for nodes in partition.keys()
#                                 if partition[nodes] == com]
#     # nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
#     #                             node_color = str(count / size))
#     print()

# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()