"""Unit tests for layout functions."""
import networkx as nx

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


class TestLayout:
    @classmethod
    def setup_class(cls):
        cls.Gi = nx.grid_2d_graph(5, 5)
        cls.Gs = nx.Graph()
        nx.add_path(cls.Gs, "abcdef")
        cls.bigG = nx.grid_2d_graph(25, 25)  # > 500 nodes for sparse

    def test_smoke_empty_graph(self):
        G = []
        nx.force_directed_hyper_graphs_using_social_and_gravity_scaling(G)

    def test_smoke_int(self):
        G = self.Gi
        nx.force_directed_hyper_graphs_using_social_and_gravity_scaling(G)

    def test_smoke_string(self):
        G = self.Gs
        nx.force_directed_hyper_graphs_using_social_and_gravity_scaling(G)

    def test_force_directed_hyper_graphs_using_social_and_gravity_scaling_layout(self):
        import math

        G = self.Gs
        pytest.raises(ValueError, nx.force_directed_hyper_graphs_using_social_and_gravity_scaling, "hello")
        pos = nx.force_directed_hyper_graphs_using_social_and_gravity_scaling(G)
        has_nan = any(math.isnan(c) for coords in pos.values() for c in coords)
        assert not has_nan, "values should not be nan"
