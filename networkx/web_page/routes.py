from flask import render_template, url_for, redirect, request

import networkx as nx
from networkx.drawing import hypergraph_layout
from networkx.web_page import app
from networkx.web_page.templates.forms import Parameters
from networkx.drawing.our_layout import force_directed_hyper_graphs_using_social_and_gravity_scaling


def random_hypergraph(num_of_vtx, num_of_edges):
    import random
    vtx = list(range(num_of_vtx))
    edges = []
    for edge in range(num_of_edges):
        v = set()
        num_of_vtx_in_edge = random.randint(1, num_of_vtx)
        for _ in range(num_of_vtx_in_edge):
            rand_vtx = random.randint(0, num_of_vtx - 1)
            v.add(rand_vtx)
        E = hypergraph_layout.hyperedge(list(v))
        edges.append(E)
    return hypergraph_layout.hypergraph(vtx, edges)


def find_centrality(cent):
    if cent == 'cl':
        return nx.closeness_centrality
    if cent == 'bt':
        return nx.betweenness_centrality
    if cent == 'dg':
        return nx.degree_centrality


def find_algo(algo):
    if algo == 'Comp':
        return hypergraph_layout.complete_algorithm
    if algo == 'Cyc':
        return hypergraph_layout.cycle_algorithm
    if algo == 'Str':
        return hypergraph_layout.star_algorithm
    if algo == 'Wh':
        return hypergraph_layout.wheel_algorithm


@app.route("/result")
def result():
    return render_template('resultpage.html')


@app.route("/", methods=['GET', 'POST'])
def home():
    form = Parameters()
    if request.method == 'POST':
        force_directed_hyper_graphs_using_social_and_gravity_scaling(
            G=random_hypergraph(form.vtx.data, form.edges.data),
            iterations=form.iter.data,
            centrality=find_centrality(form.centrality.data),
            graph_type=find_algo(form.type.data), gravity=form.gravity.data)
        return redirect(url_for('result'))
    else:
        return render_template('homepage.html', form=form)
