import math

import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing import hypergraph_layout
from networkx.drawing.hypergraph_layout import hyperedge, hypergraph

__all__ = [
    "force_directed_hyper_graphs_using_social_and_gravity_scaling",
]


def rep(x, k):
    return k ** 2 / x ** 2


def att(x, k):
    return x / k


def force_directed(G: nx.Graph, seed: int, iterations: int = 50, threshold=70e-4, centrality=None):
    import numpy as np
    A = nx.to_numpy_array(G)
    k = math.sqrt(1 / len(A))
    np.random.seed(seed)
    pos = np.asarray(np.random.rand(len(A), 2))
    I = np.zeros(shape=(2, len(A)), dtype=float)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    gamma_t = 0
    if centrality is None:
        mass = [v for v in nx.closeness_centrality(G).values()]
    else:
        mass = [v for v in centrality(G).values()]
    center = (np.sum(pos, axis=0) / len(pos))

    for iteration in range(iterations):
        I *= 0
        for v in range(len(A)):
            delta = (pos[v] - pos).T
            distance = np.sqrt((delta ** 2).sum(axis=0))
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = A[v]
            # displacement "force"
            I[:, v] += (
                               delta * (k * k / distance ** 2 - Ai * distance / k)
                       ).sum(axis=1) + gamma_t * mass[v] * (center - pos[v])
        length = np.sqrt((I ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (I * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        if gamma_t > 125:
            print('gamma')
            break
        if (np.linalg.norm(delta_pos) / len(A)) < threshold:
            print("adasds")
            print(iteration)
            threshold /= 3
            # break
            gamma_t += 6 * round(iteration / 200)
        iteration += 1
    print(iteration)
    return pos


# @nx.not_implemented_for("directed")
def force_directed_hyper_graphs_using_social_and_gravity_scaling(G: hypergraph_layout.hypergraph,
                                                                 iterations=50, threshold=70e-4, centrality=None,
                                                                 graph_type=None):
    """Positions nodes using Fruchterman-Reingold force-directed algorithm combined with Hyper-Graphs and Social and
    Gravitational Forces.

    Receives a Hyper-Graph and changes it to be a normal graph. Then calculates the Value of each node according to
    Social centrality Parameters. Nodes with high value are counted as central nodes in the graph and are more attracted
    to the center of the graph dimension.

    After running the algorithm the pos will be updated to reflect the social and force-directed values of the nodes.


    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    centrality_type: int optional (default=0)
        Centrality type for the Social gravity field used in the algorithm.

    graph_type: int optional (default=0)
        Graph type for chosing type of conversion from hyper-graph to graph (cycle/wheel/star/complete)


    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Notes
    -----
    This algorithm currently only works on hyper-graphs.

    The algorithm is based on the work of Fruchterman-Reingold and adding Forces that mimic the Social interaction of
    social networks. Forces such as closeness, betweenness and degree centrality. using these forces to force place the
    nodes of the graph in a circular way and minimize the space used by the graph in pointing it.


    References
    ----------
    .. [1] Michael J. Bannister, David Eppstein, Michael T. Goodrich and Lowell Trott:
       Force-Directed Graph Drawing Using Social Gravity and Scaling.
       Graph Drawing. GD 2012. Lecture Notes in Computer Science, vol 7704. Springer, Berlin, Heidelberg.
       https://doi.org/10.1007/978-3-642-36763-2_37
    .. [2] Naheed Anjum Arafat and St´ephane Bressan:
       Hypergraph Drawing by Force-Directed Placement
       DEXA 2017. Lecture Notes in Computer Science(), vol 10439. Springer, Cham.
      https://doi.org/10.1007/978-3-319-64471-4_31

    """
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from scipy.interpolate import splprep
    from scipy.interpolate import splev

    if graph_type is None:
        graph_type = hypergraph_layout.complete_algorithm
    g: nx.Graph = graph_type(G)
    pos = force_directed(g, 1, iterations, threshold, centrality)
    if graph_type is hypergraph_layout.star_algorithm or graph_type is hypergraph_layout.wheel_algorithm:
        pos = pos[:len(pos) - len(G.hyperedges)]
    fig, ax = plt.subplots()
    ax.scatter(pos[:, 0], pos[:, 1], zorder=2)
    for ei in G.hyperedges:
        indexes = []
        for v in ei.vertices:
            indexes.append(np.where(G.vertices == v)[0][0])
        if len(indexes) <= 2:
            ax.plot(pos[indexes, 0], pos[indexes, 1], 'k-', color='red')
            continue
        hull = ConvexHull(pos[indexes])
        ax.plot(pos[:, 0], pos[:, 1], 'o')
        # calculate center of hull
        # take two x calculate y -> check if (x,y) is in the hull
        for simplex in hull.simplices:
            # tck, u = splprep(k.points.T, u=None, s=0.0, per=1)
            # u_new = np.linspace(u.min(), u.max(), 1000)
            # x_new, y_new = splev(u_new, tck, der=0)
            #
            # ax.plot(k.points[simplex, 0], k.points[simplex, 1], 'ro')
            # ax.plot(x_new, y_new)
            # plt.show()
            ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')
    for i, txt in enumerate(G.vertices):
        ax.annotate(txt, pos[i], color='blue')
    plt.show()
    return pos


if __name__ == '__main__':
    import numpy as np

    # g = nx.Graph()
    # g.add_edge(0, 1)
    # g.add_edge(2, 3)
    # g.add_edge(4, 3)
    # g.add_edge(2, 1)
    # g.add_edge(0, 4)
    # g.add_edge(4, 5)
    # g.add_edge(6, 7)
    # # g.add_edge(5,6)
    # g.add_edge(7, 8)
    # g.add_edge(6, 8)
    # # for i in nx.closeness_centrality(g):
    # b = random.Random()
    # b.seed(1)
    # # g = nx.Graph()
    # while g.edges.__len__() != 40:
    #     d = b.randint(0, 30)
    #     a = b.randint(0, 30)
    #     if a != d and g.has_edge(a, d) is False:
    #         g.add_edge(a, d)
    # plt.show()
    # g.nodes.keys()
    g: nx.Graph = nx.random_regular_graph(3, 90, 1)
    # g: nx.Graph = nx.random_tree(70,1)
    # f: list[nx.Graph] = []
    # for i in range(1, 5):
    #     f.append(nx.random_tree(70, i))
    # pos_nx = nx.spring_layout(f[i-1], iterations=700)
    # nx.draw(f[i-1], pos_nx, node_size=70)
    # # nx.draw_spring
    # plt.show()
    #
    # for i in range(1, 5):
    #     for j in f[i-1].edges:
    #         g.add_edge(j[0]+70*(i+3)+1, j[1]+70*(i+3)+1)
    v1 = 1
    v2 = 2
    v3 = 3
    v4 = 4
    v5 = 5
    v6 = 6
    E1 = hyperedge([v1, v2, v3, v4])
    E2 = hyperedge([v3, v4])
    E3 = hyperedge([v1, v5, v6])
    G = hypergraph([v1, v2, v3, v4, v5, v6], [E1, E2, E3])
    force_directed_hyper_graphs_using_social_and_gravity_scaling(G, 1000, graph_type=hypergraph_layout.star_algorithm)
    pos_nx = nx.spring_layout(g, iterations=700)
    nx.draw(g, pos_nx, node_size=70)
    # nx.draw_spring
    plt.show()
    pos = force_directed(g, 1, iterations=1000)
    pos = nx.rescale_layout(pos)
    # pos = force_directed_hyper_graphs_using_social_and_gravity_scaling(g)
    pp = {}
    for i in range(len(pos)):
        pp[np.array(g.nodes)[i]] = np.array(pos[i])
    nx.draw(g, pp, node_size=50)
    # print(np.zeros(shape=(5, 2), dtype=float))
    plt.show()
