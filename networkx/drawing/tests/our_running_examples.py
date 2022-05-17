import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def draw_forest(num_of_trees, num_of_vxs, title1, title2, title3, title4):
    # Forest graph (5 tree of the size of 50 each)
    f: list[nx.Graph] = []
    for i in range(1, num_of_trees):
        f.append(nx.random_tree(num_of_vxs, i))
    pos_nx = nx.spring_layout(f[i - 1], iterations=700)
    plt.title(title1)
    nx.draw(f[i - 1], pos_nx, node_size=70)
    plt.show()
    pos = nx.force_directed(f[i - 1], 1, iterations=700, centrality=nx.closeness_centrality)
    plt.title(title2)
    nx.draw(f[i - 1], pos, node_size=70)
    plt.show()
    pos = nx.force_directed(f[i - 1], 1, iterations=700, centrality=nx.betweenness_centrality)
    plt.title(title3)
    nx.draw(f[i - 1], pos, node_size=70)
    plt.show()
    pos = nx.force_directed(f[i - 1], 1, iterations=700, centrality=nx.degree_centrality)
    plt.title(title4)
    nx.draw(f[i - 1], pos, node_size=70)
    plt.show()


def draw_graph(g, title1, title2, title3, title4):
    pos_nx = nx.spring_layout(g, iterations=700)
    plt.title(title1)
    nx.draw(g, pos_nx, node_size=70)
    plt.show()
    pos = nx.force_directed(g, 1, iterations=1000, centrality=nx.closeness_centrality)
    plt.title(title2)
    nx.draw(g, pos, node_size=70)
    plt.show()
    pos = nx.force_directed(g, 1, iterations=1000, centrality=nx.betweenness_centrality)
    plt.title(title3)
    nx.draw(g, pos, node_size=70)
    plt.show()
    pos = nx.force_directed(g, 1, iterations=1000, centrality=nx.degree_centrality)
    plt.title(title4)
    nx.draw(g, pos, node_size=70)
    plt.show()


def force_directed_tree():
    # Tree
    g: nx.Graph = nx.random_tree(70, 1)
    draw_graph(g, "Networkx tree", "Our tree with closeness centrality",
               "Our tree with betweenes centrality", "Our tree with degree centrality")


def force_directed_regular():
    # Regular graph
    # Note: gravity here is 0 therefore different centralities has no affect
    g: nx.Graph = nx.random_regular_graph(3, 90, 1)
    pos_nx = nx.spring_layout(g, iterations=700)
    plt.title('Networkx regular graph plot')
    nx.draw(g, pos_nx, node_size=70)
    plt.show()
    pos = nx.force_directed(g, 1, iterations=700, gravity=0, threshold=1e-4)
    pp = {}
    for i in range(len(pos)):
        pp[np.array(g.nodes)[i]] = np.array(pos[i])
    plt.title('Our regular graph plot')
    nx.draw(g, pp, node_size=70)
    plt.show()


def force_directed_small_forest():
    draw_forest(5, 70, "Networkx small forest", "Our small forest with closeness centrality",
                "Our small forest with betweenes centrality", "Our small forest with degree centrality")


def force_directed_large_forest():
    draw_forest(50, 100, "Networkx large forest", "Our large forest with closeness centrality",
                "Our large forest with betweenes centrality", "Our large forest with degree centrality")


if __name__ == '__main__':
    # regular graphs examples for the force_directed function (first article)
    # force_directed_tree()
    # force_directed_small_forest()
    # force_directed_large_forest()
    # force_directed_regular()
    pass