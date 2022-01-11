import networkx as nx
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl


g_covid_tw_resp_map_graph_file = 'covid_tw_resp_map_graph.gml'


def output_nx_graph(nx_graph, fig_path):
    if nx_graph is None or len(nx_graph.nodes) == 0:
        return
    plt.figure(1, figsize=(40, 40), tight_layout={'pad': 1, 'w_pad': 50, 'h_pad': 50, 'rect': None})
    pos = nx.spring_layout(nx_graph, k=0.6)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.10
    plt.xlim(x_min - x_margin, x_max + x_margin)
    d_node_labels = {node[0]: node[0] for node in nx_graph.nodes(data=True)}
    nx.draw_networkx_nodes(nx_graph, pos, node_size=20)
    nx.draw_networkx_labels(nx_graph, pos, labels=d_node_labels, font_size=30, font_color='r', font_weight='semibold')
    num_edges = nx_graph.number_of_edges()
    edge_colors = range(2, num_edges + 2)
    l_edges = nx_graph.edges()
    drawn_edges = nx.draw_networkx_edges(nx_graph,
                                         pos,
                                         edgelist=l_edges,
                                         width=4,
                                         edge_color=edge_colors,
                                         edge_cmap=plt.get_cmap('Blues'),
                                         arrows=True,
                                         arrowsize=40)
    max_edge_cnt = max([edge[2]['cnt'] for edge in nx_graph.edges(data=True)])
    # for idx, edge in enumerate(l_edges):
    #     edge_alpha = tmp_resp_map_graph.edges()[(edge[0], edge[1])]['cnt'] / float(max_edge_cnt)
    #     drawn_edges[idx].set_alpha(edge_alpha)
    # pc = mpl.collections.PatchCollection(drawn_edges, cmap=plt.get_cmap('Blues'))
    # pc.set_array(edge_colors)
    # plt.colorbar(pc)
    plt.savefig(fig_path, format="PNG")
    plt.clf()


if __name__ == '__main__':
    tmp_resp_map_graph = nx.read_gml(g_covid_tw_resp_map_graph_file)
    l_sorted_edges = sorted(tmp_resp_map_graph.edges(data=True), key=lambda k: k[2]['cnt'], reverse=True)
    sub_edges = [edge for edge in l_sorted_edges if edge[2]['cnt'] >= 1000]
    sub_graph = nx.DiGraph()
    for edge in sub_edges:
        node_1 = edge[0]
        node_2 = edge[1]
        if node_1 not in sub_graph.nodes():
            sub_graph.add_node(node_1, fulltxt=tmp_resp_map_graph.nodes(data=True)[node_1]['fulltxt'])
        if node_2 not in sub_graph.nodes():
            sub_graph.add_node(node_2, fulltxt=tmp_resp_map_graph.nodes(data=True)[node_2]['fulltxt'])
        sub_graph.add_edge(node_1, node_2, cnt=edge[2]['cnt'])
    output_nx_graph(sub_graph, 'covid_tw_resp_map_graph_gt1000.png')
    print()