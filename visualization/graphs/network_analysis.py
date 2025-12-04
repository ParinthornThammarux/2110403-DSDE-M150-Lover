"""
Graph Network Visualization for Complaint Relationships
Analyzes co-occurrence patterns, organization networks, and community detection

Adapted for clean_data.csv from Bangkok Traffy dataset
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
import plotly.graph_objects as go
from networkx.algorithms import community as nx_community
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplaintNetworkAnalyzer:
    """Network graph analysis for urban complaints"""

    def __init__(self):
        self.G = nx.Graph()
        self.complaint_network = None
        self.organization_network = None

    def build_complaint_type_network(self, df: pd.DataFrame):
        """
        Build network based on co-occurrence of complaint types
        Edge weight = number of times two types appear together
        """
        logger.info("Building complaint type co-occurrence network...")

        # Parse multi-label complaint types
        def parse_types(type_str):
            if pd.isna(type_str) or type_str == '{}' or type_str == 'ไม่ระบุ':
                return []
            # Remove curly braces and split
            cleaned = str(type_str).strip('{}')
            return [t.strip() for t in cleaned.split(',') if t.strip()]

        df['types_list'] = df['type'].apply(parse_types)

        # Count co-occurrences
        co_occurrence = Counter()

        for types in df['types_list']:
            if len(types) > 1:
                # All pairs of types in same complaint
                for pair in itertools.combinations(sorted(types), 2):
                    co_occurrence[pair] += 1

        # Build graph
        self.complaint_network = nx.Graph()

        for (type1, type2), weight in co_occurrence.items():
            if weight >= 5:  # Minimum threshold
                self.complaint_network.add_edge(type1, type2, weight=weight)

        # Add isolated nodes for completeness
        all_types = set()
        for types in df['types_list']:
            all_types.update(types)

        # Only add most common types to keep graph manageable
        type_counts = Counter()
        for types in df['types_list']:
            type_counts.update(types)

        # Add top 30 types
        top_types = [t for t, _ in type_counts.most_common(30)]
        for t in top_types:
            if t not in self.complaint_network:
                self.complaint_network.add_node(t)

        logger.info(f"Created network with {self.complaint_network.number_of_nodes()} nodes "
                   f"and {self.complaint_network.number_of_edges()} edges")

        return self.complaint_network

    def build_organization_network(self, df: pd.DataFrame):
        """
        Build network of organizations handling similar complaints
        """
        logger.info("Building organization collaboration network...")

        # Parse type for primary type
        def get_primary_type(type_str):
            if pd.isna(type_str) or type_str == '{}' or type_str == 'ไม่ระบุ':
                return 'Unknown'
            cleaned = str(type_str).strip('{}')
            types = [t.strip() for t in cleaned.split(',') if t.strip()]
            return types[0] if types else 'Unknown'

        df['primary_type'] = df['type'].apply(get_primary_type)

        # Create edges between organizations handling same complaint type
        type_orgs = df.groupby('primary_type')['organization'].apply(list).to_dict()

        self.organization_network = nx.Graph()

        for complaint_type, orgs in type_orgs.items():
            orgs_unique = list(set([str(o) for o in orgs if pd.notna(o) and str(o) != 'ไม่ระบุหน่วยงาน']))

            if len(orgs_unique) > 1:
                for org1, org2 in itertools.combinations(orgs_unique, 2):
                    if self.organization_network.has_edge(org1, org2):
                        self.organization_network[org1][org2]['weight'] += 1
                    else:
                        self.organization_network.add_edge(org1, org2, weight=1)

        # Filter to keep only well-connected organizations (degree >= 2)
        nodes_to_remove = [node for node, degree in dict(self.organization_network.degree()).items() if degree < 2]
        self.organization_network.remove_nodes_from(nodes_to_remove)

        logger.info(f"Created org network with {self.organization_network.number_of_nodes()} nodes "
                   f"and {self.organization_network.number_of_edges()} edges")

        return self.organization_network

    def detect_communities(self, G: nx.Graph):
        """Detect communities using greedy modularity algorithm"""
        logger.info("Detecting communities...")

        if len(G.nodes()) == 0:
            return {}

        # Use greedy modularity communities from NetworkX
        communities_generator = nx_community.greedy_modularity_communities(G, weight='weight')
        communities_list = list(communities_generator)

        # Convert to dict format {node: community_id}
        partition = {}
        for comm_id, comm_nodes in enumerate(communities_list):
            for node in comm_nodes:
                partition[node] = comm_id

        n_communities = len(communities_list)
        logger.info(f"Detected {n_communities} communities")

        # Log community members
        for i, comm in enumerate(communities_list):
            logger.info(f"Community {i}: {list(comm)[:5]}{'...' if len(comm) > 5 else ''}")

        return partition

    def calculate_centrality_metrics(self, G: nx.Graph):
        """Calculate various centrality metrics"""
        logger.info("Calculating centrality metrics...")

        if len(G.nodes()) == 0:
            return {}

        metrics = {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G, weight='weight'),
        }

        # Only calculate these for small enough graphs
        if len(G.nodes()) < 100:
            metrics['closeness_centrality'] = nx.closeness_centrality(G, distance='weight')
            try:
                metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            except:
                logger.warning("Could not calculate eigenvector centrality")
                metrics['eigenvector_centrality'] = {}

        return metrics

    def visualize_network_matplotlib(self, G: nx.Graph, title: str = "Complaint Network",
                                    communities: dict = None, save_path: str = None):
        """Visualize network using matplotlib"""
        if len(G.nodes()) == 0:
            logger.warning("Empty graph, skipping matplotlib visualization")
            return None

        plt.figure(figsize=(20, 16))

        # Layout - use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, weight='weight', seed=42)

        # Node colors based on communities
        if communities:
            node_colors = [communities.get(node, 0) for node in G.nodes()]
            n_communities = len(set(communities.values()))
            cmap = plt.cm.get_cmap('tab20', n_communities)
        else:
            node_colors = 'lightblue'
            cmap = None

        # Node sizes based on degree
        node_sizes = [500 + 200 * G.degree(node) for node in G.nodes()]

        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=cmap,
            alpha=0.8,
            edgecolors='black',
            linewidths=2
        )

        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v].get('weight', 1) for u, v in edges]
        max_weight = max(weights) if weights else 1

        nx.draw_networkx_edges(
            G, pos,
            width=[w/max_weight * 5 for w in weights],
            alpha=0.4,
            edge_color='gray'
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=12,
            font_weight='bold',
            font_family='sans-serif'
        )

        plt.title(title, fontsize=24, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = f"visualization/graphs/outputs/{title.lower().replace(' ', '_')}.png"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Network visualization saved to {save_path}")

        plt.close()
        return save_path

    def visualize_network_plotly(self, G: nx.Graph, title: str = "Complaint Network",
                                communities: dict = None, save_path: str = None):
        """Create interactive network visualization with Plotly"""
        logger.info("Creating interactive Plotly visualization...")

        if len(G.nodes()) == 0:
            logger.warning("Empty graph, skipping visualization")
            return None

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, weight='weight', seed=42)

        # Edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]].get('weight', 1)

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight/5, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node info
            degree = G.degree(node)
            neighbors = list(G.neighbors(node))
            neighbor_str = ', '.join([str(n) for n in neighbors[:5]])
            if len(neighbors) > 5:
                neighbor_str += f'... ({len(neighbors)-5} more)'

            node_text.append(f"<b>{node}</b><br>Degree: {degree}<br>Connected to: {neighbor_str}")
            node_size.append(20 + degree * 5)

            if communities:
                node_color.append(communities.get(node, 0))
            else:
                node_color.append(degree)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(n) for n in G.nodes()],
            hovertext=node_text,
            textposition='top center',
            textfont=dict(size=10),
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Community' if communities else 'Degree',
                    thickness=15,
                    xanchor='left'
                ),
                line=dict(width=2, color='white')
            )
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1400,
            height=1000
        )

        # Save
        if save_path is None:
            save_path = f"visualization/graphs/outputs/{title.lower().replace(' ', '_')}_interactive.html"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Interactive visualization saved to {save_path}")

        return fig

    def analyze_network_properties(self, G: nx.Graph):
        """Analyze graph properties and statistics"""
        logger.info("Analyzing network properties...")

        if len(G.nodes()) == 0:
            return {}

        properties = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
        }

        if nx.is_connected(G):
            properties['diameter'] = nx.diameter(G)
            properties['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            # For disconnected graphs, analyze largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc).copy()
            properties['num_components'] = nx.number_connected_components(G)
            properties['largest_component_size'] = len(largest_cc)
            if len(G_largest) > 1:
                properties['diameter_largest_component'] = nx.diameter(G_largest)
                properties['avg_shortest_path_largest_component'] = nx.average_shortest_path_length(G_largest)

        properties['avg_clustering'] = nx.average_clustering(G)
        properties['transitivity'] = nx.transitivity(G)

        # Degree statistics
        degrees = [d for n, d in G.degree()]
        properties['avg_degree'] = np.mean(degrees) if degrees else 0
        properties['max_degree'] = max(degrees) if degrees else 0
        properties['min_degree'] = min(degrees) if degrees else 0
        properties['std_degree'] = np.std(degrees) if degrees else 0

        logger.info("\n" + "=" * 60)
        logger.info("NETWORK PROPERTIES")
        logger.info("=" * 60)
        for key, value in properties.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 60)

        return properties

    def export_graph(self, G: nx.Graph, filename: str):
        """Export graph to various formats"""
        output_dir = Path("visualization/graphs/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # GraphML format
        nx.write_graphml(G, str(output_dir / f"{filename}.graphml"))

        # GML format
        nx.write_gml(G, str(output_dir / f"{filename}.gml"))

        # Edge list
        nx.write_edgelist(G, str(output_dir / f"{filename}_edges.txt"))

        logger.info(f"Exported graph to {output_dir}/{filename}.*")


def main():
    """Main network analysis pipeline"""
    logger.info("=" * 80)
    logger.info("Complaint Network Analysis")
    logger.info("=" * 80)

    # Load data from clean_data.csv
    csv_path = 'clean_data.csv'

    if not Path(csv_path).exists():
        logger.error(f"Data file not found: {csv_path}")
        logger.error("Please ensure clean_data.csv is in the root directory")
        return

    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} complaints")

    # Initialize analyzer
    analyzer = ComplaintNetworkAnalyzer()

    # 1. Build complaint type network
    logger.info("\n" + "=" * 80)
    logger.info("COMPLAINT TYPE CO-OCCURRENCE NETWORK")
    logger.info("=" * 80)
    complaint_graph = analyzer.build_complaint_type_network(df)

    # 2. Detect communities
    communities = analyzer.detect_communities(complaint_graph)

    # 3. Calculate centrality
    centrality = analyzer.calculate_centrality_metrics(complaint_graph)

    if 'degree_centrality' in centrality:
        logger.info("\nTop 10 nodes by degree centrality:")
        sorted_centrality = sorted(centrality['degree_centrality'].items(),
                                  key=lambda x: x[1], reverse=True)
        for node, score in sorted_centrality[:10]:
            logger.info(f"  {node}: {score:.3f}")

    # 4. Visualize
    analyzer.visualize_network_matplotlib(complaint_graph,
                                         title="Complaint Type Co-occurrence Network",
                                         communities=communities)

    analyzer.visualize_network_plotly(complaint_graph,
                                      title="Interactive Complaint Network",
                                      communities=communities)

    # 5. Analyze properties
    properties = analyzer.analyze_network_properties(complaint_graph)

    # 6. Build organization network
    logger.info("\n" + "=" * 80)
    logger.info("ORGANIZATION COLLABORATION NETWORK")
    logger.info("=" * 80)
    org_graph = analyzer.build_organization_network(df)

    if org_graph.number_of_nodes() > 0:
        org_communities = analyzer.detect_communities(org_graph)

        analyzer.visualize_network_matplotlib(org_graph,
                                             title="Organization Collaboration Network",
                                             communities=org_communities)

        analyzer.visualize_network_plotly(org_graph,
                                          title="Interactive Organization Network",
                                          communities=org_communities)

        # Analyze org network properties
        analyzer.analyze_network_properties(org_graph)

        # 7. Export graphs
        analyzer.export_graph(complaint_graph, "complaint_network")
        analyzer.export_graph(org_graph, "organization_network")
    else:
        logger.warning("Organization network is empty, skipping visualization")

    logger.info("\n" + "=" * 80)
    logger.info("Network analysis completed successfully!")
    logger.info("=" * 80)
    logger.info("\nOutput files:")
    logger.info("  - visualization/graphs/outputs/complaint_type_co-occurrence_network.png")
    logger.info("  - visualization/graphs/outputs/interactive_complaint_network_interactive.html")
    if org_graph.number_of_nodes() > 0:
        logger.info("  - visualization/graphs/outputs/organization_collaboration_network.png")
        logger.info("  - visualization/graphs/outputs/interactive_organization_network_interactive.html")


if __name__ == "__main__":
    main()
