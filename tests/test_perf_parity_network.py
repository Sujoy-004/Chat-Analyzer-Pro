"""Performance-refactor parity tests for the interaction-network module.

Proves the pure-performance change in build_interaction_network is
behavior-identical: the per-row Python `.iloc` loop that counts
sender-switch edges is replaced with a vectorized pandas groupby.

The reference implementation below is the ORIGINAL pre-refactor loop
captured verbatim; every fixture/threshold must produce an identical
graph (node set, edge set, edge weights) from both paths.

These are FAST tests (no pytest.mark.slow).
"""

import os

# Headless-first (Pitfall 7): network_graph imports matplotlib.pyplot at
# module import; pin Agg BEFORE any pyplot import so figures are headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

from collections import defaultdict

import networkx as nx
import pandas as pd

from chat_analyzer.analysis.network_graph import build_interaction_network
from chat_analyzer.ingest.ingestion import messages_to_dataframe

# ============================================================================
# Reference implementation: the FULL ORIGINAL build_interaction_network,
# captured verbatim BEFORE the vectorization refactor.
# ============================================================================

def _reference_build_interaction_network(
    df: pd.DataFrame, weight_threshold: int = 0
) -> nx.DiGraph:
    """
    Build a directed network graph from chat interactions.

    Args:
        df: DataFrame with 'datetime', 'sender', 'message' columns
        weight_threshold: Minimum interactions to include an edge

    Returns:
        NetworkX directed graph with interaction weights
    """
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    # Create directed graph
    G = nx.DiGraph()

    # Add all participants as nodes
    participants = df['sender'].unique()
    G.add_nodes_from(participants)

    # Build edges based on reply patterns (consecutive messages)
    interaction_counts = defaultdict(int)

    for i in range(1, len(df)):
        prev_sender = df.iloc[i-1]['sender']
        curr_sender = df.iloc[i]['sender']

        # If different senders, it's an interaction
        if prev_sender != curr_sender:
            interaction_counts[(prev_sender, curr_sender)] += 1

    # Add edges with weights
    for (from_node, to_node), weight in interaction_counts.items():
        if weight > weight_threshold:
            G.add_edge(from_node, to_node, weight=weight)

    return G


# ============================================================================
# Fixture builders
# ============================================================================

def _build_df(senders: list[str]) -> pd.DataFrame:
    """Build a fixture df (1-hour spaced messages) via messages_to_dataframe."""
    start = pd.Timestamp('2024-01-01 09:00:00')
    messages = [
        {
            'datetime': start + pd.Timedelta(hours=i),
            'sender': sender,
            'message': f'message {i} from {sender}',
        }
        for i, sender in enumerate(senders)
    ]
    return messages_to_dataframe(messages)


NETWORK_FIXTURES = {
    'alternating': ['Alice', 'Bob', 'Alice', 'Bob', 'Alice', 'Bob', 'Alice', 'Bob'],
    'same_sender_run': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Alice', 'Alice'],
    'three_senders': ['Alice', 'Bob', 'Carol', 'Alice', 'Bob', 'Carol', 'Alice', 'Bob', 'Carol'],
    'single_message': ['Alice'],
    'all_same_sender': ['Alice', 'Alice', 'Alice', 'Alice', 'Alice'],
    'mixed': ['Alice', 'Alice', 'Bob', 'Carol', 'Carol', 'Carol', 'Dana', 'Dana', 'Alice'],
}

# Thresholds covering the default (no pruning) and edge pruning.
THRESHOLDS = (0, 2)


# ============================================================================
# Parity tests
# ============================================================================

def _edge_items(G: nx.DiGraph) -> frozenset:
    """Canonical (u, v, {'weight': w}) edge comparison — dicts are unhashable,
    so project `edges(data=True)` to a frozenset of (u, v, weight) triples."""
    return frozenset((u, v, data['weight']) for u, v, data in G.edges(data=True))


def test_network_vectorized_matches_original():
    """Vectorized edge counting is identical to the original loop.

    For every fixture and threshold, the edge set + weights and the node set
    must match exactly: `set(G.edges(data=True)) == set(reference.edges(data=True))`
    compared as `(u, v, {'weight': w})` tuples.
    """
    for name, senders in NETWORK_FIXTURES.items():
        df = _build_df(senders)

        for threshold in THRESHOLDS:
            reference = _reference_build_interaction_network(df, threshold)
            actual = build_interaction_network(df, threshold)

            assert set(actual.nodes()) == set(reference.nodes()), (
                f'{name} threshold={threshold}: node sets differ'
            )

            # Compare as (u, v, {'weight': w}) tuples — exact weight parity.
            assert _edge_items(actual) == _edge_items(reference), (
                f'{name} threshold={threshold}: edge sets/weights differ'
            )

            # Weight type parity: ints, not numpy ints.
            for _u, _v, data in actual.edges(data=True):
                assert isinstance(data['weight'], int), (
                    f'{name} threshold={threshold}: weight {data["weight"]!r} '
                    f'is {type(data["weight"]).__name__}, expected int'
                )


def test_network_single_message_has_no_edges():
    """Single-message chat: no prev row to switch against, so no edges."""
    df = _build_df(['Alice'])
    G = build_interaction_network(df)

    assert list(G.nodes()) == ['Alice']
    assert list(G.edges()) == []


def test_network_threshold_prunes_low_weight_edges():
    """weight_threshold=4 keeps only edges with weight > 4.

    alternating A/B (8 messages) yields A->B x4 and B->A x3, so every edge is
    pruned and only the nodes remain — same as the reference.
    """
    df = _build_df(['Alice', 'Bob', 'Alice', 'Bob', 'Alice', 'Bob', 'Alice', 'Bob'])

    reference = _reference_build_interaction_network(df, 4)
    actual = build_interaction_network(df, 4)

    assert set(actual.nodes()) == set(reference.nodes())
    assert _edge_items(actual) == _edge_items(reference)
    assert list(actual.edges()) == []


# ============================================================================
# compute-once regression: network_figure reuses analyze_network's result
# ============================================================================

import io
from pathlib import Path

import pytest

from chat_analyzer.analysis import network_graph as network_graph_module
from chat_analyzer.cli.pipeline import run_pipeline

try:
    from rich.console import Console
except ImportError:
    Console = None


def test_network_computed_once_per_pipeline(monkeypatch):
    """run_pipeline computes analyze_network exactly once — network_figure
    reuses the result instead of re-running the full graph build (betweenness,
    PageRank, community detection)."""
    if Console is None:
        pytest.skip("rich not installed")

    real = network_graph_module.analyze_network
    calls = {"count": 0}

    def counting_wrapper(df, weight_threshold=0):
        calls["count"] += 1
        return real(df, weight_threshold=weight_threshold)

    monkeypatch.setattr(network_graph_module, "analyze_network", counting_wrapper)

    sample = Path(__file__).resolve().parent.parent / "data" / "sample_chats" / "whatsapp_sample.txt"
    assert sample.exists(), "sample chat fixture missing"

    console = Console(file=io.StringIO(), force_terminal=False)
    results = run_pipeline(sample, console, nlp_enabled=False)

    assert calls["count"] == 1, (
        f"analyze_network ran {calls['count']}x — network_figure must reuse "
        "run_pipeline's computed result, not re-run the graph build"
    )
    assert isinstance(results["network"]["density"], float), (
        "the pipeline still produced a real network analysis"
    )
