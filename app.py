import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import random
from io import StringIO

# Set page configuration
st.set_page_config(layout="wide", page_title="Cluster Influence Analysis")


# Sample data with your exact provided default data
@st.cache_data
def generate_sample_data():
    # Your exact CSV data
    csv_data = """cluster_name,total_population,influences_count,willing_to_move_count,willing_names,not_willing_names
Cluster 1,15,2,0,,"Lino, Shamila"
Cluster 2,30,8,0,,"Stani, Shani, Lakshman, Ps. Shamika, Therica, Asangi, Rehan, Lushanthi"
Cluster 3,30,5,0,,"Rochelle, Ps. Prasad, Gillian, Sacha, Anita"
Cluster 4,30,3,1,Key,"Judha, Josiah"
Cluster 5,15,6,4,"Dirk, Jakie, Brindly, Nilanka","Nimal, Krishni"
Cluster 6,30,6,4,"Berty, Melinda, Eranga, Lakmini","Nalaka, Nayomi"
Cluster 7,30,6,2,"Jessi, Ronan","Rajeev, Famina, Mel, Enya, "
Cluster 8,30,4,0,,"Pathum, Minon, Lynnara, Jessica"
Cluster 9,20,3,0,,"Judy, Ruwan, Tanya"
Cluster 10,30,4,0,,"Chinta, Rajith, Harshi, Mindi"
Cluster 11,0,0,0,,
Cluster 12,0,0,0,,
Cluster 13,10,0,0,,
Cluster 14,10,0,0,,
Cluster 15,0,0,0,,
Cluster 16,0,0,0,,
Cluster 17,25,3,1,Sonali,"Ps. Christo, Maurine\""""

    # Parse the CSV data
    df = pd.read_csv(StringIO(csv_data))
    
    # Process the data to match our expected format
    processed_data = []
    for _, row in df.iterrows():
        total_population = row['total_population']
        influences = row['influences_count']
        willing_to_move = row['willing_to_move_count']
        not_willing_to_move = influences - willing_to_move
        normal_population = total_population - influences
        
        # Check if this is a dormant cluster - only 11, 12, 15, 16
        is_dormant = (row['cluster_name'] in ['Cluster 11', 'Cluster 12', 'Cluster 15', 'Cluster 16']) or total_population == 0
        
        # Handle NaN values for names
        willing_names = row['willing_names'] if pd.notna(row['willing_names']) else ''
        not_willing_names = row['not_willing_names'] if pd.notna(row['not_willing_names']) else ''
        
        processed_data.append({
            'name': row['cluster_name'],
            'total_population': total_population,
            'normal_population': normal_population,
            'influences': influences,
            'willing_to_move': willing_to_move,
            'not_willing_to_move': not_willing_to_move,
            'influence_percentage': (influences / total_population) * 100 if total_population > 0 else 0,
            'willing_names': willing_names,
            'not_willing_names': not_willing_names,
            'is_dormant': is_dormant
        })
    
    return pd.DataFrame(processed_data)

# Function to create cluster visualization
def plot_clusters(df, title, show_willing=True):
    fig, axes = plt.subplots(5, 4, figsize=(20, 24))
    axes = axes.flatten()
    
    # Define colors
    normal_color = '#3498db'
    willing_color = '#e74c3c'
    not_willing_color = '#f39c12'
    dormant_color = '#95a5a6'
    text_color = '#2c3e50'
    border_color = '#e74c3c'  # CHANGED to RED
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        ax.clear()
        
        # Check if cluster is dormant
        is_dormant = row.get('is_dormant', False) or row['total_population'] == 0
        
        # Create fancy bounding box with THIN RED BORDERS
        if is_dormant:
            bbox = FancyBboxPatch((0.05, 0.05), 0.9, 0.85,
                                 boxstyle="round,pad=0.04", 
                                 linewidth=2,  # CHANGED to thin border
                                 edgecolor='#e74c3c',  # CHANGED to RED
                                 facecolor='#ecf0f1',
                                 linestyle='--')
            face_color = '#ecf0f1'
        else:
            bbox = FancyBboxPatch((0.05, 0.05), 0.9, 0.85,
                                 boxstyle="round,pad=0.04", 
                                 linewidth=2,  # CHANGED to thin border
                                 edgecolor='#e74c3c',  # CHANGED to RED
                                 facecolor='#f8f9fa')
            face_color = '#f8f9fa'
        
        ax.add_patch(bbox)
        
        # CLUSTER NAME INSIDE THE BOX - MOVED LOWER
        ax.text(0.5, 0.88, row['name'], ha='center', va='center', 
                fontsize=16, fontweight='bold', 
                color='#2c3e50', alpha=1.0)
        
        if is_dormant:
            # Display dormant cluster message
            ax.text(0.5, 0.5, "DORMANT", ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='#7f8c8d', 
                    style='italic', alpha=0.7)
            ax.text(0.5, 0.4, "No Population", ha='center', va='center', 
                    fontsize=12, color='#7f8c8d', alpha=0.7)
        else:
            # Calculate bar heights
            total_height = 0.4  # Reduced height to make space for labels
            if row['total_population'] > 0:
                normal_height = (row['normal_population'] / row['total_population']) * total_height
                if show_willing:
                    willing_height = (row['willing_to_move'] / row['total_population']) * total_height
                    not_willing_height = (row['not_willing_to_move'] / row['total_population']) * total_height
                else:
                    influences_height = (row['influences'] / row['total_population']) * total_height
            else:
                normal_height = willing_height = not_willing_height = influences_height = 0
            
            # Draw population bars - starting position adjusted
            y_pos = 0.4  # Adjusted starting position
            
            # Normal population
            normal_bar = patches.Rectangle((0.15, y_pos), 0.7, normal_height, 
                                         facecolor=normal_color, alpha=0.9,
                                         edgecolor='white', linewidth=1)
            ax.add_patch(normal_bar)
            
            if show_willing:
                # Willing to move
                y_pos += normal_height
                willing_bar = patches.Rectangle((0.15, y_pos), 0.7, willing_height, 
                                              facecolor=willing_color, alpha=0.9,
                                              edgecolor='white', linewidth=1)
                ax.add_patch(willing_bar)
                
                # Not willing to move
                y_pos += willing_height
                not_willing_bar = patches.Rectangle((0.15, y_pos), 0.7, not_willing_height, 
                                                  facecolor=not_willing_color, alpha=0.9,
                                                  edgecolor='white', linewidth=1)
                ax.add_patch(not_willing_bar)
            else:
                # Total influences (without breakdown)
                y_pos += normal_height
                influences_bar = patches.Rectangle((0.15, y_pos), 0.7, influences_height, 
                                                 facecolor=willing_color, alpha=0.9,
                                                 edgecolor='white', linewidth=1)
                ax.add_patch(influences_bar)
        
        # NEW LABELS INSIDE THE CLUSTERS
        if not is_dormant:
            # Members label
            ax.text(0.5, 0.32, f"Members: {row['normal_population']}", 
                    ha='center', va='center', fontsize=11, fontweight='bold', 
                    color='#2c3e50', alpha=0.9)
            
            # Influences willing to move label
            ax.text(0.5, 0.27, f"Influences willing to move: {row['willing_to_move']}", 
                    ha='center', va='center', fontsize=10, fontweight='bold', 
                    color=willing_color, alpha=0.9)
            
            # Influences willing not to move label
            ax.text(0.5, 0.22, f"Influences not willing to move: {row['not_willing_to_move']}", 
                    ha='center', va='center', fontsize=10, fontweight='bold', 
                    color=not_willing_color, alpha=0.9)
            
            # Total label
            ax.text(0.5, 0.17, f"Total: {row['total_population']}", 
                    ha='center', va='center', fontsize=11, fontweight='bold', 
                    color='#2c3e50', alpha=0.9)
            
            # Influence percentage (additional info)
            influence_pct = row['influence_percentage']
            ax.text(0.5, 0.12, f"Influence %: {influence_pct:.1f}%", 
                    ha='center', va='center', fontsize=11, fontweight='bold', 
                    color='#7f8c8d', alpha=0.8)
        
        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(df), len(axes)):
        axes[i].axis('off')
    
    # Add legend
    legend_fig = plt.figure(figsize=(14, 1))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    
    if show_willing:
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor=normal_color, alpha=0.9, label='Normal Population'),
            patches.Rectangle((0, 0), 1, 1, facecolor=willing_color, alpha=0.9, label='Willing to Move'),
            patches.Rectangle((0, 0), 1, 1, facecolor=not_willing_color, alpha=0.9, label='Not Willing to Move'),
            patches.Rectangle((0, 0), 1, 1, facecolor=dormant_color, alpha=0.7, label='Dormant Cluster', linestyle='--', edgecolor='#e74c3c')
        ]
    else:
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor=normal_color, alpha=0.9, label='Normal Population'),
            patches.Rectangle((0, 0), 1, 1, facecolor=willing_color, alpha=0.9, label='Influences'),
            patches.Rectangle((0, 0), 1, 1, facecolor=dormant_color, alpha=0.7, label='Dormant Cluster', linestyle='--', edgecolor='#e74c3c')
        ]
    
    legend_ax.legend(handles=legend_elements, loc='center', ncol=4, 
                    fontsize=12, frameon=True, fancybox=True, 
                    framealpha=0.9, edgecolor=border_color)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig, legend_fig

# Function to normalize influences across clusters
def normalize_influences(df):
    normalized_df = df.copy()
    
    # Only consider active, non-dormant clusters for normalization
    active_clusters = normalized_df[(normalized_df['total_population'] > 0) & 
                                   (~normalized_df.get('is_dormant', False))].copy()
    
    if len(active_clusters) == 0:
        return normalized_df, []
    
    # Calculate target influence percentage (average of current percentages)
    target_percentage = active_clusters['influence_percentage'].mean()
    
    # Calculate total willing to move influences
    total_willing = active_clusters['willing_to_move'].sum()
    
    # Calculate required influences for each active cluster to reach target percentage
    required_influences = []
    active_indices = []
    for idx, row in active_clusters.iterrows():
        required = int((target_percentage / 100) * row['total_population'])
        required_influences.append(max(1, required))
        active_indices.append(idx)
    
    # Calculate deficits and surpluses
    current_influences = active_clusters['influences'].values
    deficits = np.maximum(0, required_influences - current_influences)
    surpluses = np.maximum(0, current_influences - required_influences)
    
    # Redistribute willing influences
    total_deficit = deficits.sum()
    redistribution_plan = []
    
    if total_deficit > 0 and total_willing > 0:
        # Calculate redistribution factors
        redistribution_factors = deficits / total_deficit if total_deficit > 0 else np.zeros(len(deficits))
        
        # Redistribute willing influences
        for i, idx in enumerate(active_indices):
            cluster = active_clusters.loc[idx]
            current_inf = cluster['influences']
            required_inf = required_influences[i]
            
            if current_inf < required_inf:
                # This cluster needs more influences
                additional = min(int(redistribution_factors[i] * total_willing), 
                               required_inf - current_inf,
                               total_willing)
                normalized_df.at[idx, 'influences'] = current_inf + additional
                # All redistributed influences become "not willing" in their new cluster
                normalized_df.at[idx, 'not_willing_to_move'] += additional
                
                if additional > 0:
                    redistribution_plan.append({
                        'from': 'Willing Pool',
                        'to': cluster['name'],
                        'count': additional
                    })
            else:
                # This cluster has enough or too many influences
                normalized_df.at[idx, 'influences'] = max(required_inf, cluster['not_willing_to_move'])
            
            # Update willing to move (set to 0 after redistribution)
            normalized_df.at[idx, 'willing_to_move'] = 0
            
            # Update normal population
            normalized_df.at[idx, 'normal_population'] = (normalized_df.at[idx, 'total_population'] - 
                                                        normalized_df.at[idx, 'influences'])
            
            # Update influence percentage
            if normalized_df.at[idx, 'total_population'] > 0:
                normalized_df.at[idx, 'influence_percentage'] = (normalized_df.at[idx, 'influences'] / 
                                                               normalized_df.at[idx, 'total_population']) * 100
    
    return normalized_df, redistribution_plan

# Function to parse uploaded CSV
def parse_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['cluster_name', 'total_population', 'influences_count', 
                          'willing_to_move_count', 'willing_names', 'not_willing_names']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("""
            Required CSV format:
            - cluster_name: Name of the cluster
            - total_population: Total population in the cluster
            - influences_count: Total number of influences
            - willing_to_move_count: Number of influences willing to move
            - willing_names: Comma-separated names of willing influences
            - not_willing_names: Comma-separated names of not-willing influences
            """)
            return None
        
        # Process the data for 17 clusters
        processed_data = []
        for _, row in df.iterrows():
            total_population = row['total_population']
            influences = row['influences_count']
            willing_to_move = row['willing_to_move_count']
            not_willing_to_move = influences - willing_to_move
            normal_population = total_population - influences
            
            # Check if this is a dormant cluster - only 11, 12, 15, 16
            is_dormant = (row['cluster_name'] in ['Cluster 11', 'Cluster 12', 'Cluster 15', 'Cluster 16']) or total_population == 0
            
            # Handle NaN values for names
            willing_names = row['willing_names'] if pd.notna(row['willing_names']) else ''
            not_willing_names = row['not_willing_names'] if pd.notna(row['not_willing_names']) else ''
            
            processed_data.append({
                'name': row['cluster_name'],
                'total_population': total_population,
                'normal_population': normal_population,
                'influences': influences,
                'willing_to_move': willing_to_move,
                'not_willing_to_move': not_willing_to_move,
                'influence_percentage': (influences / total_population) * 100 if total_population > 0 else 0,
                'willing_names': willing_names,
                'not_willing_names': not_willing_names,
                'is_dormant': is_dormant
            })
        
        return pd.DataFrame(processed_data)
    
    except Exception as e:
        st.error(f"Error parsing CSV file: {str(e)}")
        return None

# Function to create influence details table
def create_influence_details_table(df):
    influence_data = []
    
    for _, row in df.iterrows():
        cluster_name = row['name']
        
        # Skip dormant clusters
        if row.get('is_dormant', False) or row['total_population'] == 0:
            continue
        
        # Process willing influences - with robust NaN/empty handling
        willing_names_str = row['willing_names']
        if pd.notna(willing_names_str) and str(willing_names_str).strip():
            # Convert to string and split, then clean each name
            names_list = str(willing_names_str).split(',')
            for name in names_list:
                name_clean = name.strip()
                if name_clean:  # Only add non-empty names
                    influence_data.append({
                        'Cluster': cluster_name,
                        'Influence Name': name_clean,
                        'Willing to Move': 'Yes',
                        'Population Type': 'Influence'
                    })
        
        # Process not willing influences - with robust NaN/empty handling
        not_willing_names_str = row['not_willing_names']
        if pd.notna(not_willing_names_str) and str(not_willing_names_str).strip():
            # Convert to string and split, then clean each name
            names_list = str(not_willing_names_str).split(',')
            for name in names_list:
                name_clean = name.strip()
                if name_clean:  # Only add non-empty names
                    influence_data.append({
                        'Cluster': cluster_name,
                        'Influence Name': name_clean,
                        'Willing to Move': 'No',
                        'Population Type': 'Influence'
                    })
        
        # Add normal population entries (simplified representation)
        for i in range(row['normal_population']):
            influence_data.append({
                'Cluster': cluster_name,
                'Influence Name': f'Normal Person {i+1}',
                'Willing to Move': 'N/A',
                'Population Type': 'Normal'
            })
    
    return pd.DataFrame(influence_data)

# Main application
def main():
    st.title("👥 Cluster Influence Analysis")
    
    # Sidebar for data management
    st.sidebar.header("📁 Data Management")
    
    if st.sidebar.button("🔄 Generate Sample Data"):
        st.session_state.df = generate_sample_data()
        st.session_state.data_source = "sample"
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Your Data")
    
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv", 
                                           help="Upload your cluster data in CSV format")
    
    if uploaded_file is not None:
        processed_df = parse_uploaded_csv(uploaded_file)
        if processed_df is not None:
            st.session_state.df = processed_df
            st.session_state.data_source = "uploaded"
            st.sidebar.success("✅ Data uploaded successfully!")
    
    # Initialize dataframe in session state if not exists
    if 'df' not in st.session_state:
        st.session_state.df = generate_sample_data()
        st.session_state.data_source = "sample"
    
    # Display data source info
    if st.session_state.data_source == "sample":
        st.sidebar.info("📊 Using sample data")
    else:
        st.sidebar.success("📁 Using uploaded data")
    
    # Calculate total population from actual data
    total_population = st.session_state.df['total_population'].sum()
    active_clusters_count = len(st.session_state.df[(st.session_state.df['total_population'] > 0) & 
                                                  (~st.session_state.df.get('is_dormant', False))])
    
    # Display cluster configuration info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Cluster Configuration")
    st.sidebar.write("**Total:** 17 Clusters")
    st.sidebar.write(f"**Active:** {active_clusters_count} Clusters")
    st.sidebar.write(f"**Total Population:** {total_population}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Current Clusters", "🔄 Normalized Recommendations", "👤 Influence Details"])
    
    with tab1:
        st.header("Current Cluster Distribution")
        
        # Display overall statistics
        total_pop = st.session_state.df['total_population'].sum()
        total_influences = st.session_state.df['influences'].sum()
        total_willing = st.session_state.df['willing_to_move'].sum()
        active_clusters = len(st.session_state.df[(st.session_state.df['total_population'] > 0) & 
                                                (~st.session_state.df.get('is_dormant', False))])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Clusters", len(st.session_state.df))
        with col2:
            st.metric("Active Clusters", active_clusters)
        with col3:
            st.metric("Total Population", total_pop)
        with col4:
            st.metric("Total Influences", total_influences)
        
        # Display the visualization
        fig_current, legend_fig = plot_clusters(st.session_state.df, "Current Cluster Distribution (17 Clusters)", show_willing=True)
        st.pyplot(fig_current)
        st.pyplot(legend_fig)
        
        # Display data table
        st.subheader("Detailed Cluster Data")
        display_df = st.session_state.df.copy()
        display_df['influence_percentage'] = display_df['influence_percentage'].round(2)
        st.dataframe(display_df[['name', 'total_population', 'normal_population', 
                               'influences', 'willing_to_move', 'not_willing_to_move', 
                               'influence_percentage']], 
                    use_container_width=True)
    
    with tab2:
        st.header("Normalized Cluster Distribution")
        
        # Calculate normalized distribution
        normalized_df, redistribution_plan = normalize_influences(st.session_state.df)
        
        # Display statistics
        original_active = st.session_state.df[(st.session_state.df['total_population'] > 0) & 
                                            (~st.session_state.df.get('is_dormant', False))]
        normalized_active = normalized_df[(normalized_df['total_population'] > 0) & 
                                        (~normalized_df.get('is_dormant', False))]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            original_avg = original_active['influence_percentage'].mean()
            new_avg = normalized_active['influence_percentage'].mean()
            st.metric("Avg Influence %", f"{new_avg:.1f}%", f"{new_avg - original_avg:+.1f}%")
        with col2:
            original_std = original_active['influence_percentage'].std()
            new_std = normalized_active['influence_percentage'].std()
            st.metric("Std Dev %", f"{new_std:.1f}%", f"{new_std - original_std:+.1f}%")
        with col3:
            total_moved = st.session_state.df['willing_to_move'].sum() - normalized_df['willing_to_move'].sum()
            st.metric("Influences Moved", total_moved)
        with col4:
            st.metric("Remaining Willing", normalized_df['willing_to_move'].sum())
        
        # Display the normalized visualization
        fig_normalized, legend_fig2 = plot_clusters(normalized_df, 
                                                   "Normalized Cluster Distribution (After Redistribution)", 
                                                   show_willing=False)
        st.pyplot(fig_normalized)
        st.pyplot(legend_fig2)
        
        # Display comparison
        st.subheader("Before vs After Comparison")
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.write("**Original Distribution**")
            orig_display = st.session_state.df[['name', 'influences', 'influence_percentage']].copy()
            orig_display['influence_percentage'] = orig_display['influence_percentage'].round(2)
            st.dataframe(orig_display, use_container_width=True)
        
        with comp_col2:
            st.write("**Normalized Distribution**")
            norm_display = normalized_df[['name', 'influences', 'influence_percentage']].copy()
            norm_display['influence_percentage'] = norm_display['influence_percentage'].round(2)
            st.dataframe(norm_display, use_container_width=True)
        
        # Show movement summary
        if redistribution_plan:
            st.subheader("Movement Plan")
            movement_df = pd.DataFrame(redistribution_plan)
            st.dataframe(movement_df, use_container_width=True)
        else:
            st.info("No movements required - clusters are already well balanced!")
    
    with tab3:
        st.header("Influence Details - Table View")
        
        # Create the influence details table
        influence_df = create_influence_details_table(st.session_state.df)
        
        if not influence_df.empty:
            # Display statistics
            total_influences = len(influence_df[influence_df['Population Type'] == 'Influence'])
            willing_influences = len(influence_df[(influence_df['Population Type'] == 'Influence') & 
                                                (influence_df['Willing to Move'] == 'Yes')])
            normal_population = len(influence_df[influence_df['Population Type'] == 'Normal'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Influences", total_influences)
            with col2:
                st.metric("Willing to Move", willing_influences)
            with col3:
                st.metric("Normal Population", normal_population)
            
            # Add filters
            st.subheader("Filter Data")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                clusters = ['All'] + sorted(influence_df['Cluster'].unique().tolist())
                selected_cluster = st.selectbox("Filter by Cluster", clusters)
            
            with filter_col2:
                willingness_options = ['All', 'Yes', 'No', 'N/A']
                selected_willingness = st.selectbox("Filter by Willingness", willingness_options)
            
            with filter_col3:
                population_types = ['All', 'Influence', 'Normal']
                selected_population = st.selectbox("Filter by Population Type", population_types, index=1)
            
            # Apply filters
            filtered_df = influence_df.copy()
            
            if selected_cluster != 'All':
                filtered_df = filtered_df[filtered_df['Cluster'] == selected_cluster]
            
            if selected_willingness != 'All':
                filtered_df = filtered_df[filtered_df['Willing to Move'] == selected_willingness]
            
            if selected_population != 'All':
                filtered_df = filtered_df[filtered_df['Population Type'] == selected_population]
            
            # Display the table
            st.subheader("Influence Details Table")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Add download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name="influence_details.csv",
                mime="text/csv"
            )
        else:
            st.warning("No influence data available. Please check your data source.")
        
        # Export data section
        st.subheader("Export Data")
        
        if st.button("📥 Download Sample CSV Template"):
            # Create sample template based on your exact data structure
            template_data = [
                {'cluster_name': 'Cluster 1', 'total_population': 15, 'influences_count': 2, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': 'Lino, Shamila'},
                {'cluster_name': 'Cluster 2', 'total_population': 30, 'influences_count': 8, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': 'Stani, Shani, Lakshman, Ps. Shamika, Therica, Asangi, Rehan, Lushanthi'},
                {'cluster_name': 'Cluster 3', 'total_population': 30, 'influences_count': 5, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': 'Rochelle, Ps. Prasad, Gillian, Sacha, Anita'},
                {'cluster_name': 'Cluster 4', 'total_population': 30, 'influences_count': 3, 'willing_to_move_count': 1, 'willing_names': 'Key', 'not_willing_names': 'Judha, Josiah'},
                {'cluster_name': 'Cluster 5', 'total_population': 15, 'influences_count': 6, 'willing_to_move_count': 4, 'willing_names': 'Dirk, Jakie, Brindly, Nilanka', 'not_willing_names': 'Nimal, Krishni'},
                {'cluster_name': 'Cluster 6', 'total_population': 30, 'influences_count': 6, 'willing_to_move_count': 4, 'willing_names': 'Berty, Melinda, Eranga, Lakmini', 'not_willing_names': 'Nalaka, Nayomi'},
                {'cluster_name': 'Cluster 7', 'total_population': 30, 'influences_count': 6, 'willing_to_move_count': 2, 'willing_names': 'Jessi, Ronan', 'not_willing_names': 'Rajeev, Famina, Mel, Enya, '},
                {'cluster_name': 'Cluster 8', 'total_population': 30, 'influences_count': 4, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': 'Pathum, Minon, Lynnara, Jessica'},
                {'cluster_name': 'Cluster 9', 'total_population': 20, 'influences_count': 3, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': 'Judy, Ruwan, Tanya'},
                {'cluster_name': 'Cluster 10', 'total_population': 30, 'influences_count': 4, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': 'Chinta, Rajith, Harshi, Mindi'},
                {'cluster_name': 'Cluster 11', 'total_population': 0, 'influences_count': 0, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': ''},
                {'cluster_name': 'Cluster 12', 'total_population': 0, 'influences_count': 0, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': ''},
                {'cluster_name': 'Cluster 13', 'total_population': 10, 'influences_count': 0, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': ''},
                {'cluster_name': 'Cluster 14', 'total_population': 10, 'influences_count': 0, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': ''},
                {'cluster_name': 'Cluster 15', 'total_population': 0, 'influences_count': 0, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': ''},
                {'cluster_name': 'Cluster 16', 'total_population': 0, 'influences_count': 0, 'willing_to_move_count': 0, 'willing_names': '', 'not_willing_names': ''},
                {'cluster_name': 'Cluster 17', 'total_population': 25, 'influences_count': 3, 'willing_to_move_count': 1, 'willing_names': 'Sonali', 'not_willing_names': 'Ps. Christo, Maurine'}
            ]
            
            template_df = pd.DataFrame(template_data)
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Template",
                data=csv,
                file_name="17_cluster_data_template.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()