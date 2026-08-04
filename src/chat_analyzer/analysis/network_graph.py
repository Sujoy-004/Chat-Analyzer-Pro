"""
Network Graph Analysis Module

This module analyzes conversation networks in group chats, including:
- Reply networks (who responds to whom)
- Interaction frequencies and weights
- Centrality measures (who's most central in the conversation)
- Community detection
- Network visualizations

Designed for group chats with 3+ participants, but works with any size.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional


def build_interaction_network(df: pd.DataFrame, weight_threshold: int = 0) -> nx.DiGraph:
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


def calculate_network_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate various network centrality and connectivity metrics.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Dictionary with network metrics
    """
    metrics = {}
    
    # Basic network properties
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Centrality measures
    try:
        metrics['degree_centrality'] = nx.degree_centrality(G)
        metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
        metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
        
        # Betweenness centrality (may fail for disconnected graphs)
        if nx.is_weakly_connected(G):
            metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        else:
            metrics['betweenness_centrality'] = {}
        
        # PageRank (works for any graph)
        metrics['pagerank'] = nx.pagerank(G)
        
    except Exception as e:
        print(f"Warning: Could not calculate some centrality metrics: {e}")
        metrics['betweenness_centrality'] = {}
        metrics['pagerank'] = {}
    
    # Connectivity
    metrics['is_strongly_connected'] = nx.is_strongly_connected(G)
    metrics['is_weakly_connected'] = nx.is_weakly_connected(G)
    
    if metrics['is_weakly_connected']:
        metrics['diameter'] = nx.diameter(G.to_undirected())
    else:
        metrics['diameter'] = None
    
    return metrics


def identify_key_participants(G: nx.DiGraph, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify key participants based on network position.
    
    Args:
        G: NetworkX directed graph
        metrics: Network metrics from calculate_network_metrics()
        
    Returns:
        Dictionary with key participant roles
    """
    key_roles = {}
    
    # Most active (highest out-degree)
    if 'out_degree_centrality' in metrics:
        out_degrees = metrics['out_degree_centrality']
        if out_degrees:
            most_active = max(out_degrees, key=out_degrees.get)
            key_roles['most_active'] = {
                'participant': most_active,
                'score': out_degrees[most_active]
            }
    
    # Most responsive (highest in-degree)
    if 'in_degree_centrality' in metrics:
        in_degrees = metrics['in_degree_centrality']
        if in_degrees:
            most_responsive = max(in_degrees, key=in_degrees.get)
            key_roles['most_responsive'] = {
                'participant': most_responsive,
                'score': in_degrees[most_responsive]
            }
    
    # Most influential (highest PageRank)
    if 'pagerank' in metrics and metrics['pagerank']:
        pageranks = metrics['pagerank']
        most_influential = max(pageranks, key=pageranks.get)
        key_roles['most_influential'] = {
            'participant': most_influential,
            'score': pageranks[most_influential]
        }
    
    # Bridge connector (highest betweenness)
    if 'betweenness_centrality' in metrics and metrics['betweenness_centrality']:
        betweenness = metrics['betweenness_centrality']
        bridge = max(betweenness, key=betweenness.get)
        key_roles['bridge_connector'] = {
            'participant': bridge,
            'score': betweenness[bridge]
        }
    
    return key_roles


def analyze_interaction_patterns(df: pd.DataFrame, G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze detailed interaction patterns between participants.
    
    Args:
        df: Original DataFrame
        G: NetworkX directed graph
        
    Returns:
        Dictionary with interaction pattern analysis
    """
    patterns = {}
    
    # Get edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    
    if not edge_weights:
        return {'error': 'No interactions found'}
    
    # Strongest connections
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
    patterns['strongest_connections'] = [
        {
            'from': edge[0],
            'to': edge[1],
            'interactions': weight
        }
        for edge, weight in sorted_edges[:5]
    ]
    
    # Reciprocity analysis
    reciprocal_pairs = []
    for (from_node, to_node), weight in edge_weights.items():
        reverse_weight = edge_weights.get((to_node, from_node), 0)
        if reverse_weight > 0:
            reciprocal_pairs.append({
                'pair': (from_node, to_node),
                'forward': weight,
                'backward': reverse_weight,
                'balance': abs(weight - reverse_weight) / max(weight, reverse_weight)
            })
    
    patterns['reciprocal_interactions'] = reciprocal_pairs
    
    # Calculate reciprocity score
    if len(edge_weights) > 0:
        reciprocal_count = len([p for p in reciprocal_pairs if p['balance'] < 0.5])
        patterns['reciprocity_score'] = reciprocal_count / (len(edge_weights) / 2)
    else:
        patterns['reciprocity_score'] = 0.0
    
    # Interaction matrix
    participants = list(G.nodes())
    interaction_matrix = pd.DataFrame(0, index=participants, columns=participants)
    
    for (from_node, to_node), weight in edge_weights.items():
        interaction_matrix.loc[from_node, to_node] = weight
    
    patterns['interaction_matrix'] = interaction_matrix
    
    return patterns


def detect_subgroups(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Detect subgroups/communities within the network.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Dictionary with community detection results
    """
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    subgroups = {}
    
    try:
        # Louvain community detection (best modularity)
        from networkx.algorithms import community
        
        communities = community.greedy_modularity_communities(G_undirected)
        
        subgroups['communities'] = [list(c) for c in communities]
        subgroups['num_communities'] = len(communities)
        
        # Modularity score
        subgroups['modularity'] = community.modularity(G_undirected, communities)
        
    except Exception as e:
        # Fallback: use connected components
        components = list(nx.connected_components(G_undirected))
        subgroups['communities'] = [list(c) for c in components]
        subgroups['num_communities'] = len(components)
        subgroups['modularity'] = None
    
    return subgroups


def plot_network_graph(
    G: nx.DiGraph,
    metrics: Dict[str, Any] = None,
    layout: str = 'spring',
    figsize: Tuple[int, int] = (14, 10),
    node_size_metric: str = 'degree',
    title: str = 'Conversation Network Graph'
) -> None:
    """
    Visualize the conversation network with customizable layout.
    
    Args:
        G: NetworkX directed graph
        metrics: Optional network metrics for sizing/coloring
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        figsize: Figure size tuple
        node_size_metric: Metric for node sizing ('degree', 'pagerank', 'betweenness')
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Calculate node sizes based on metric
    if metrics and node_size_metric in ['pagerank', 'betweenness', 'degree']:
        if node_size_metric == 'pagerank' and 'pagerank' in metrics:
            sizes = [metrics['pagerank'].get(node, 0.1) * 10000 for node in G.nodes()]
        elif node_size_metric == 'betweenness' and 'betweenness_centrality' in metrics:
            sizes = [metrics['betweenness_centrality'].get(node, 0.1) * 10000 for node in G.nodes()]
        else:
            sizes = [G.degree(node) * 500 for node in G.nodes()]
    else:
        sizes = [G.degree(node) * 500 for node in G.nodes()]
    
    # Ensure minimum size
    sizes = [max(size, 1000) for size in sizes]
    
    # Get edge weights for width
    edge_weights = nx.get_edge_attributes(G, 'weight')
    max_weight = max(edge_weights.values()) if edge_weights else 1
    edge_widths = [edge_weights.get((u, v), 1) / max_weight * 5 for u, v in G.edges()]
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=sizes,
        node_color='lightblue',
        edgecolors='darkblue',
        linewidths=2,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        alpha=0.6,
        ax=ax
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        font_family='sans-serif',
        ax=ax
    )
    
    # Add edge labels (weights)
    edge_labels = {(u, v): f"{w}" for (u, v), w in edge_weights.items()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=9,
        font_color='red',
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_network_dashboard(
    G: nx.DiGraph,
    metrics: Dict[str, Any],
    patterns: Dict[str, Any],
    figsize: Tuple[int, int] = (20, 14)
) -> None:
    """
    Create comprehensive network analysis dashboard.
    
    Args:
        G: NetworkX directed graph
        metrics: Network metrics
        patterns: Interaction patterns
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main network graph
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes by degree
    node_sizes = [G.degree(node) * 800 for node in G.nodes()]
    node_sizes = [max(size, 800) for size in node_sizes]
    
    # Edge widths by weight
    edge_weights = nx.get_edge_attributes(G, 'weight')
    max_weight = max(edge_weights.values()) if edge_weights else 1
    edge_widths = [edge_weights.get((u, v), 1) / max_weight * 5 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                          edgecolors='darkblue', linewidths=2, ax=ax1)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray',
                          arrows=True, arrowsize=20, arrowstyle='->',
                          connectionstyle='arc3,rad=0.1', alpha=0.6, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax1)
    
    edge_labels = {(u, v): f"{w}" for (u, v), w in edge_weights.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=9, font_color='red', ax=ax1)
    
    ax1.set_title('Conversation Network Graph', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Centrality comparison
    ax2 = fig.add_subplot(gs[0, 2])
    if 'pagerank' in metrics and metrics['pagerank']:
        participants = list(metrics['pagerank'].keys())
        pagerank_values = list(metrics['pagerank'].values())
        
        bars = ax2.barh(participants, pagerank_values, color='#3498DB', alpha=0.7)
        ax2.set_xlabel('PageRank Score', fontsize=10)
        ax2.set_title('Influence (PageRank)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Degree distribution
    ax3 = fig.add_subplot(gs[1, 2])
    degrees = dict(G.degree())
    if degrees:
        participants = list(degrees.keys())
        degree_values = list(degrees.values())
        
        ax3.bar(participants, degree_values, color='#E74C3C', alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Total Connections', fontsize=10)
        ax3.set_title('Connection Degree', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Interaction matrix heatmap
    ax4 = fig.add_subplot(gs[2, :2])
    if 'interaction_matrix' in patterns:
        matrix = patterns['interaction_matrix']
        sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd',
                   cbar_kws={'label': 'Interactions'}, ax=ax4,
                   linewidths=0.5, linecolor='gray')
        ax4.set_title('Interaction Matrix (Who talks to whom)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('To →', fontsize=10)
        ax4.set_ylabel('From ↓', fontsize=10)
    
    # 5. Network statistics text
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    stats_text = f"""NETWORK STATISTICS

Nodes: {metrics['num_nodes']}
Edges: {metrics['num_edges']}
Density: {metrics['density']:.3f}

Connected: {'Yes' if metrics['is_weakly_connected'] else 'No'}
Diameter: {metrics['diameter'] if metrics['diameter'] else 'N/A'}

Reciprocity: {patterns.get('reciprocity_score', 0):.2%}

Strongest Connection:
"""
    
    if 'strongest_connections' in patterns and patterns['strongest_connections']:
        top = patterns['strongest_connections'][0]
        stats_text += f"{top['from']} → {top['to']}\n({top['interactions']} interactions)"
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Network Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def analyze_network(df: pd.DataFrame, weight_threshold: int = 0) -> Dict[str, Any]:
    """
    Complete network analysis pipeline.
    
    Args:
        df: DataFrame with chat messages
        weight_threshold: Minimum interactions for edge inclusion
        
    Returns:
        Dictionary with complete network analysis
    """
    # Build network
    G = build_interaction_network(df, weight_threshold)
    
    # Calculate metrics
    metrics = calculate_network_metrics(G)
    
    # Identify key participants
    key_participants = identify_key_participants(G, metrics)
    
    # Analyze patterns
    patterns = analyze_interaction_patterns(df, G)
    
    # Detect subgroups
    subgroups = detect_subgroups(G)
    
    return {
        'graph': G,
        'metrics': metrics,
        'key_participants': key_participants,
        'patterns': patterns,
        'subgroups': subgroups
    }


# Example usage
def example_usage():
    """Example of how to use network analysis functions."""
    # Sample data
    sample_data = {
        'datetime': pd.date_range('2024-01-01', periods=30, freq='1H'),
        'sender': ['Alice', 'Bob', 'Charlie'] * 10,
        'message': ['Hello'] * 30
    }
    
    df = pd.DataFrame(sample_data)
    
    # Run analysis
    results = analyze_network(df)
    
    print("=== NETWORK ANALYSIS ===")
    print(f"Nodes: {results['metrics']['num_nodes']}")
    print(f"Edges: {results['metrics']['num_edges']}")
    print(f"Density: {results['metrics']['density']:.3f}")
    
    # Visualize
    plot_network_dashboard(
        results['graph'],
        results['metrics'],
        results['patterns']
    )
    
    return results


if __name__ == "__main__":
    example_usage()
