"""
Interactive Network Analysis Dashboard for Bangkok Complaint Data
Built with Streamlit and Pyvis
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import os
from collections import Counter
import itertools
from typing import Dict, Optional, Tuple
from pathlib import Path

# Configure matplotlib for Thai fonts
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Tahoma', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_complaint_data():
    """Load complaint data from CSV"""
    csv_path = 'clean_data.csv'

    if not Path(csv_path).exists():
        st.error(f"❌ Data file not found: {csv_path}")
        st.info("Please ensure clean_data.csv is in the project root directory")
        st.stop()

    df = pd.read_csv(csv_path)
    st.success(f"✅ Loaded {len(df):,} complaints from {csv_path}")
    return df

@st.cache_data
def build_complaint_network(df: pd.DataFrame, min_weight: int = 5):
    """Build complaint type co-occurrence network"""

    def parse_types(type_str):
        if pd.isna(type_str) or type_str == '{}' or type_str == 'ไม่ระบุ':
            return []
        cleaned = str(type_str).strip('{}')
        return [t.strip() for t in cleaned.split(',') if t.strip()]

    df['types_list'] = df['type'].apply(parse_types)

    # Count co-occurrences
    co_occurrence = Counter()
    for types in df['types_list']:
        if len(types) > 1:
            for pair in itertools.combinations(sorted(types), 2):
                co_occurrence[pair] += 1

    # Build graph
    G = nx.Graph()
    for (type1, type2), weight in co_occurrence.items():
        if weight >= min_weight:
            G.add_edge(type1, type2, weight=weight)

    # Add top isolated nodes
    all_types = set()
    for types in df['types_list']:
        all_types.update(types)

    type_counts = Counter()
    for types in df['types_list']:
        type_counts.update(types)

    top_types = [t for t, _ in type_counts.most_common(30)]
    for t in top_types:
        if t not in G:
            G.add_node(t)

    return G

@st.cache_data
def build_organization_network(df: pd.DataFrame):
    """Build organization collaboration network (memory-efficient)"""

    def get_primary_type(type_str):
        if pd.isna(type_str) or type_str == '{}' or type_str == 'ไม่ระบุ':
            return 'Unknown'
        cleaned = str(type_str).strip('{}')
        types = [t.strip() for t in cleaned.split(',') if t.strip()]
        return types[0] if types else 'Unknown'

    df_copy = df.copy()
    df_copy['primary_type'] = df_copy['type'].apply(get_primary_type)

    # Get organization counts per type
    type_org_pairs = df_copy.groupby(['primary_type', 'organization']).size().reset_index(name='count')
    type_org_pairs = type_org_pairs[type_org_pairs['organization'].notna()]
    type_org_pairs = type_org_pairs[type_org_pairs['organization'] != 'ไม่ระบุหน่วยงาน']

    G = nx.Graph()

    for complaint_type, group in type_org_pairs.groupby('primary_type'):
        orgs = group['organization'].unique().tolist()

        # Limit to top 20 organizations per type
        if len(orgs) > 20:
            top_orgs = group.nlargest(20, 'count')['organization'].tolist()
            orgs = top_orgs

        if len(orgs) > 1:
            for org1, org2 in itertools.combinations(orgs, 2):
                if G.has_edge(org1, org2):
                    G[org1][org2]['weight'] += 1
                else:
                    G.add_edge(org1, org2, weight=1)

    # Filter to well-connected organizations
    if len(G.nodes()) > 0:
        nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree < 3]
        G.remove_nodes_from(nodes_to_remove)

    return G

class NetworkVisualizer:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.colors = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000"
        ]

    def _get_layout(self, layout_type: str, G: nx.Graph, k_space: float = 2.0):
        """Calculate layout positions"""
        if layout_type == "spring":
            k = 1/np.sqrt(len(G.nodes())) * k_space if len(G.nodes()) > 0 else 1.0
            return nx.spring_layout(G, k=k, iterations=50, seed=42)
        elif layout_type == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        elif layout_type == "circular":
            return nx.circular_layout(G)
        elif layout_type == "random":
            return nx.random_layout(G, seed=42)
        else:
            return nx.spring_layout(G, seed=42)

    def create_interactive_network(
        self,
        communities: Optional[Dict] = None,
        layout: str = "spring",
        centrality_metric: str = "degree",
        scale_factor: float = 1000,
        node_spacing: float = 2.0,
        node_size_range: Tuple[int, int] = (10, 50),
        show_edges: bool = True,
        font_size: int = 14,
        edge_width_scale: float = 1.0
    ) -> str:
        """Create interactive network visualization"""

        if len(self.G.nodes()) == 0:
            st.warning("⚠️ Network has no nodes to visualize")
            return None

        # Get layout positions
        pos = self._get_layout(layout, self.G, node_spacing)

        # Scale positions
        pos = {node: (coord[0] * scale_factor, coord[1] * scale_factor)
               for node, coord in pos.items()}

        # Calculate centrality
        try:
            if centrality_metric == "degree":
                centrality = nx.degree_centrality(self.G)
            elif centrality_metric == "betweenness":
                centrality = nx.betweenness_centrality(self.G)
            elif centrality_metric == "closeness":
                centrality = nx.closeness_centrality(self.G)
            else:  # pagerank
                centrality = nx.pagerank(self.G)
        except:
            centrality = nx.degree_centrality(self.G)
            st.warning(f"Failed to compute {centrality_metric} centrality, using degree centrality")

        # Scale node sizes
        min_cent = min(centrality.values()) if centrality.values() else 0
        max_cent = max(centrality.values()) if centrality.values() else 1
        min_size, max_size = node_size_range

        if max_cent > min_cent:
            size_scale = lambda x: min_size + (x - min_cent) * (max_size - min_size) / (max_cent - min_cent)
        else:
            size_scale = lambda x: (min_size + max_size) / 2

        # Create visualization graph
        G_vis = self.G.copy()

        # Prepare color map for communities
        if communities:
            unique_communities = sorted(set(communities.values()))
            color_map = {com: self.colors[i % len(self.colors)]
                        for i, com in enumerate(unique_communities)}

        # Set node attributes
        for node in G_vis.nodes():
            degree = self.G.degree(node)
            node_weight = G_vis.nodes[node].get('weight', 1)

            title_text = f"Node: {node}\nDegree: {degree}"
            if communities:
                title_text += f"\nCommunity: {communities[node]}"

            G_vis.nodes[node].update({
                'label': str(node),
                'size': size_scale(centrality[node]),
                'x': pos[node][0],
                'y': pos[node][1],
                'physics': False,
                'title': title_text,
                'color': color_map[communities[node]] if communities else None
            })

        # Create pyvis network
        nt = Network(
            height="720px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=self.G.is_directed()
        )

        # Convert from networkx
        nt.from_nx(G_vis)

        # Disable physics
        nt.toggle_physics(False)

        # Set options
        nt.set_options("""
        {
            "nodes": {
                "font": {"size": %d},
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "shape": "dot"
            },
            "edges": {
                "color": {"color": "#666666"},
                "width": %f,
                "smooth": {
                    "type": "continuous",
                    "roundness": 0.5
                },
                "hidden": %s
            },
            "physics": {
                "enabled": false
            },
            "interaction": {
                "hover": true,
                "multiselect": true,
                "navigationButtons": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragView": true,
                "zoomSpeed": 0.5,
                "minZoom": 0.5,
                "maxZoom": 3.0
            }
        }
        """ % (font_size, edge_width_scale, str(not show_edges).lower()))

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tmp:
            nt.save_graph(tmp.name)
            return tmp.name

class NetworkAnalyzer:
    def __init__(self, G: nx.Graph):
        self.G = G

    def get_basic_stats(self) -> Dict:
        """Calculate basic network statistics"""
        try:
            if nx.is_connected(self.G):
                diameter = nx.diameter(self.G)
            else:
                diameter = None
        except:
            diameter = None

        return {
            "Nodes": len(self.G.nodes()),
            "Edges": len(self.G.edges()),
            "Density": nx.density(self.G),
            "Diameter": diameter,
            "Connected": nx.is_connected(self.G)
        }

    def get_centralities(self) -> pd.DataFrame:
        """Calculate centrality metrics"""
        centrality_metrics = {}

        try:
            centrality_metrics['Degree'] = nx.degree_centrality(self.G)
        except:
            pass

        try:
            centrality_metrics['Betweenness'] = nx.betweenness_centrality(self.G)
        except:
            pass

        try:
            centrality_metrics['Closeness'] = nx.closeness_centrality(self.G)
        except:
            pass

        try:
            centrality_metrics['PageRank'] = nx.pagerank(self.G)
        except:
            pass

        if not centrality_metrics:
            return pd.DataFrame()

        df = pd.DataFrame(centrality_metrics)
        df.index.name = 'Node'
        return df.reset_index()

    def get_top_central_nodes(self, top_n: int = 5) -> Dict:
        """Get top N nodes for each centrality measure"""
        top_nodes = {}

        try:
            degree_cent = nx.degree_centrality(self.G)
            top_nodes['Degree'] = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except:
            pass

        try:
            betweenness_cent = nx.betweenness_centrality(self.G)
            top_nodes['Betweenness'] = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except:
            pass

        try:
            closeness_cent = nx.closeness_centrality(self.G)
            top_nodes['Closeness'] = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except:
            pass

        try:
            pagerank = nx.pagerank(self.G)
            top_nodes['PageRank'] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except:
            pass

        return top_nodes

    def plot_degree_distribution(self) -> plt.Figure:
        """Plot degree distribution"""
        degrees = [d for n, d in self.G.degree()]

        if not degrees:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Linear scale
        max_degree = max(degrees)
        bins = min(max_degree + 1, 50)
        ax1.hist(degrees, bins=bins, align='left', rwidth=0.8, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Degree', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Degree Distribution (Linear Scale)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Log-log scale
        degree_counts = pd.Series(degrees).value_counts().sort_index()
        if len(degree_counts) > 1:
            ax2.loglog(degree_counts.index, degree_counts.values, 'bo-', alpha=0.6, markersize=8)
            ax2.set_xlabel('Degree (log scale)', fontsize=12)
            ax2.set_ylabel('Frequency (log scale)', fontsize=12)
            ax2.set_title('Degree Distribution (Log-Log Scale)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        return fig

@st.cache_data
def detect_communities(_G: nx.Graph):
    """Detect communities using greedy modularity"""
    try:
        communities_iter = nx.community.greedy_modularity_communities(_G, weight='weight')
        communities = {}
        community_list = []

        for idx, community in enumerate(communities_iter):
            community_list.append(community)
            for node in community:
                communities[node] = idx

        return communities, community_list
    except Exception as e:
        st.error(f"Community detection failed: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="Bangkok Complaint Network Analysis",
        page_icon="🕸️",
        layout="wide"
    )

    st.title("🕸️ Bangkok Complaint Network Analysis")
    st.markdown("### Interactive Network Visualization and Analysis")
    st.markdown("---")

    # Load data
    with st.spinner("Loading complaint data..."):
        df = load_complaint_data()

    # Create tabs
    tab_viz, tab_analysis = st.tabs(["🗺️ Network Visualization", "📊 Network Analysis"])

    # Visualization tab
    with tab_viz:
        with st.sidebar:
            st.subheader("Network Selection")

            network_type = st.radio(
                "Choose Network Type",
                ["Complaint Type Co-occurrence", "Organization Collaboration"]
            )

            # Build selected network
            with st.spinner(f"Building {network_type.lower()} network..."):
                if network_type == "Complaint Type Co-occurrence":
                    min_weight = st.slider("Minimum co-occurrence count", 1, 50, 5)
                    G = build_complaint_network(df, min_weight=min_weight)
                else:
                    G = build_organization_network(df)

            if G is None or len(G.nodes()) == 0:
                st.error("❌ Failed to build network or network is empty")
                st.stop()

            st.success(f"✅ Network built: {len(G.nodes())} nodes, {len(G.edges())} edges")

            # Visualization options
            st.markdown("---")
            st.subheader("Visualization Options")

            layout_option = st.selectbox(
                "Layout Algorithm",
                ["spring", "kamada_kawai", "circular", "random"]
            )

            centrality_option = st.selectbox(
                "Node Size By",
                ["degree", "betweenness", "closeness", "pagerank"]
            )

            scale_factor = st.slider(
                "Graph Size",
                min_value=500,
                max_value=3000,
                value=1200,
                step=100
            )

            if layout_option == "spring":
                node_spacing = st.slider(
                    "Node Spacing",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5
                )
            else:
                node_spacing = 2.0

            node_size_range = st.slider(
                "Node Size Range",
                min_value=5,
                max_value=100,
                value=(15, 40),
                step=5
            )

            font_size = st.slider(
                "Label Font Size",
                min_value=8,
                max_value=24,
                value=12,
                step=2
            )

            edge_width = st.slider(
                "Edge Width",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.5
            )

            show_edges = st.toggle("Show Edges", value=True)

            # Community detection
            st.markdown("---")
            show_communities = st.checkbox("Detect Communities")

            communities = None
            community_list = None

            if show_communities:
                with st.spinner("Detecting communities..."):
                    communities, community_list = detect_communities(G)

                if communities and community_list:
                    st.caption("📊 Community Statistics")
                    st.metric("Number of Communities", len(community_list))

                    community_sizes = [len(c) for c in community_list]
                    avg_size = sum(community_sizes) / len(community_sizes)
                    st.metric("Average Size", f"{avg_size:.1f}")

                    community_df = pd.DataFrame({
                        'Community': range(len(community_sizes)),
                        'Size': community_sizes
                    }).sort_values('Size', ascending=False)

                    st.dataframe(community_df, hide_index=True, height=200)

        # Main visualization area
        if G is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", len(G.nodes()))
            with col2:
                st.metric("Edges", len(G.edges()))
            with col3:
                st.metric("Density", f"{nx.density(G):.3f}")

            # Create visualization
            visualizer = NetworkVisualizer(G)

            with st.spinner("Creating interactive visualization..."):
                html_file = visualizer.create_interactive_network(
                    communities=communities,
                    layout=layout_option,
                    centrality_metric=centrality_option,
                    scale_factor=scale_factor,
                    node_spacing=node_spacing,
                    node_size_range=node_size_range,
                    show_edges=show_edges,
                    font_size=font_size,
                    edge_width_scale=edge_width
                )

            if html_file:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                st.components.v1.html(html_content, height=800)
                os.unlink(html_file)

    # Analysis tab
    with tab_analysis:
        if G is not None:
            analyzer = NetworkAnalyzer(G)

            st.header("📊 Network Analysis")

            # Basic statistics
            stats = analyzer.get_basic_stats()

            st.subheader("Basic Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes", stats["Nodes"])
            with col2:
                st.metric("Edges", stats["Edges"])
            with col3:
                st.metric("Density", f"{stats['Density']:.4f}")
            with col4:
                st.metric("Connected", "Yes" if stats["Connected"] else "No")

            if stats["Diameter"]:
                st.metric("Network Diameter", stats["Diameter"])

            # Top central nodes
            st.markdown("---")
            st.subheader("Top 5 Most Central Nodes")

            with st.spinner("Computing centrality measures..."):
                top_nodes = analyzer.get_top_central_nodes(top_n=5)

            if top_nodes:
                cols = st.columns(len(top_nodes))
                for idx, (centrality_type, nodes) in enumerate(top_nodes.items()):
                    with cols[idx]:
                        st.write(f"**{centrality_type}**")
                        top_df = pd.DataFrame(nodes, columns=['Node', 'Score'])
                        top_df['Rank'] = range(1, len(top_df) + 1)
                        top_df['Score'] = top_df['Score'].round(4)
                        st.dataframe(
                            top_df[['Rank', 'Node', 'Score']],
                            hide_index=True,
                            height=220
                        )

            # Degree distribution
            st.markdown("---")
            st.subheader("Degree Distribution")
            fig = analyzer.plot_degree_distribution()
            if fig:
                st.pyplot(fig)

            # Full centrality table
            st.markdown("---")
            st.subheader("Complete Centrality Analysis")

            centrality_df = analyzer.get_centralities()

            if not centrality_df.empty:
                search_node = st.text_input("🔍 Search for a specific node:", "")
                if search_node:
                    filtered_df = centrality_df[centrality_df['Node'].str.contains(search_node, case=False, na=False)]
                    st.dataframe(filtered_df, height=400, use_container_width=True)
                else:
                    st.dataframe(centrality_df, height=400, use_container_width=True)

                # Download button
                csv = centrality_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Centrality Data (CSV)",
                    data=csv,
                    file_name="centrality_analysis.csv",
                    mime="text/csv"
                )

            # Community analysis
            st.markdown("---")
            st.subheader("Community Analysis")

            with st.spinner("Analyzing communities..."):
                communities, community_list = detect_communities(G)

            if communities and community_list:
                community_details = []
                for idx, community in enumerate(community_list):
                    community_nodes = list(community)
                    subgraph = G.subgraph(community_nodes)

                    # Get top nodes
                    if len(community_nodes) > 0:
                        degrees = dict(subgraph.degree())
                        top_3 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
                        top_nodes_str = ', '.join([node for node, _ in top_3])
                    else:
                        top_nodes_str = ""

                    community_details.append({
                        'Community': idx,
                        'Size': len(community_nodes),
                        'Edges': subgraph.number_of_edges(),
                        'Density': nx.density(subgraph) if len(community_nodes) > 1 else 0,
                        'Top Nodes': top_nodes_str
                    })

                comm_df = pd.DataFrame(community_details).sort_values('Size', ascending=False)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Communities", len(community_list))
                with col2:
                    avg_size = sum([c['Size'] for c in community_details]) / len(community_details)
                    st.metric("Avg Size", f"{avg_size:.1f}")
                with col3:
                    largest = max(community_details, key=lambda x: x['Size'])
                    st.metric("Largest", largest['Size'])

                st.dataframe(comm_df, hide_index=True, use_container_width=True)

                # View community members
                with st.expander("👥 View nodes in each community"):
                    selected_comm = st.selectbox(
                        "Select community:",
                        range(len(community_list))
                    )

                    community_nodes = sorted(list(community_list[selected_comm]))
                    st.write(f"**Community {selected_comm}** ({len(community_nodes)} nodes):")

                    cols = st.columns(3)
                    for i, node in enumerate(community_nodes):
                        with cols[i % 3]:
                            st.write(f"- {node}")
        else:
            st.info("Please build a network in the Visualization tab first")

if __name__ == "__main__":
    main()
