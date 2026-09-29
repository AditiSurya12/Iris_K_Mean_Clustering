import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.stats import mode
import os

# Set page config
st.set_page_config(page_title="Iris K-Means Explorer", layout="wide")

# ==========================================
# 1. DATA GATHERING & CACHING
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    csv_filename = 'Iris.csv'
    # Try finding the CSV locally, otherwise fallback to generating it from the text context
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        # Construct fallback structure if user runs app directly without generating local Iris.csv
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
        df['Species'] = [iris.target_names[i] for i in iris.target]
        if 'Id' not in df.columns:
            df.insert(0, 'Id', range(1, len(df) + 1))
            
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = df[features].copy()
    y = df['Species'].astype('category').cat.codes.values
    class_names = list(df['Species'].astype('category').cat.categories)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, y, class_names, features

try:
    df, X_scaled, y, class_names, feature_cols = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error initializing data: {e}")
    st.stop()

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🛠️ Model Configuration")
st.sidebar.write("Configure unsupervised K-Means properties below.")

# Dynamic K input selector
k_selected = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3, step=1)
st.sidebar.markdown("---")

# Precompute diagnostics for the static reference plots
@st.cache_data
def precompute_diagnostics(_X_scaled, _y):
    k_range = range(2, 7)
    inertias = []
    silhouettes = []
    accuracies = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        lbls = km.fit_predict(_X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(_X_scaled, lbls))
        
        # Post-hoc accuracy majority mapping logic
        mapped = np.zeros_like(lbls)
        for cluster in range(k):
            mask = lbls == cluster
            if np.sum(mask) > 0:
                mapped[mask] = mode(_y[mask], keepdims=True).mode[0]
        accuracies.append(accuracy_score(_y, mapped))
        
    return list(k_range), inertias, silhouettes, accuracies

k_vals, inertias, silhouettes, accuracies = precompute_diagnostics(X_scaled, y)

# ==========================================
# 3. INTERACTIVE MODEL EXECUTION
# ==========================================
kmeans = KMeans(n_clusters=k_selected, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Map dynamic clusters to actual majority label names for validation matrix
mapped_labels = np.zeros_like(cluster_labels)
for c in range(k_selected):
    m = cluster_labels == c
    if np.sum(m) > 0:
        mapped_labels[m] = mode(y[m], keepdims=True).mode[0]

current_accuracy = accuracy_score(y, mapped_labels)
current_silhouette = silhouette_score(X_scaled, cluster_labels)

# ==========================================
# 4. DASHBOARD PRESENTATION LAYOUT
# ==========================================
st.title("🌸 Unsupervised K-Means Clustering Dashboard")
st.write("Explore feature clusters mapped out using the Iris measurement profile metrics.")

# Top-level metric cards
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Selected Clusters (K)", value=k_selected)
m2.metric(label="Post-hoc Accuracy Mapping", value=f"{current_accuracy * 100:.2f} %")
m3.metric(label="Silhouette Validation Score", value=f"{current_silhouette:.4f}")
m4.metric(label="Model Structural Inertia", value=f"{kmeans.inertia_:.2f}")

st.markdown("---")

# Main Content Windows
tab1, tab2, tab3 = st.tabs(["📊 Cluster Visualization (PCA)", "📈 Diagnostics Metrics", "📋 Structured Data Output"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("PCA 2-Dimensional Cluster Layout Projection")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = [f"Cluster {i}" for i in cluster_labels]
        pca_df['Actual Species'] = df['Species'].values
        
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=80, ax=ax)
        ax.set_title(f"K-Means Configuration Output Scatter (K = {k_selected})")
        ax.set_xlabel("Principal Component 1 (PC1)")
        ax.set_ylabel("Principal Component 2 (PC2)")
        st.pyplot(fig)
        
    with col_b:
        st.subheader("Actual Species vs Cluster Crosstab Match")
        crosstab_df = pd.crosstab(df['Species'], [f"Cluster {i}" for i in cluster_labels], rownames=['Actual Target'], colnames=['Assigned Cluster'])
        st.dataframe(crosstab_df, use_container_width=True)
        
        st.subheader("Confusion Assessment Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y, mapped_labels)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', cbar=False, ax=ax_cm)
        ax_cm.set_xlabel('Mapped Predicted Label')
        ax_cm.set_ylabel('Ground Truth Label')
        st.pyplot(fig_cm)

with tab2:
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.subheader("The Elbow Evaluation Method")
        fig_el, ax_el = plt.subplots(figsize=(6, 4))
        ax_el.plot(k_vals, inertias, marker='o', linestyle='-', color='b')
        ax_el.axvline(x=k_selected, color='r', linestyle='--', label=f"Active K={k_selected}")
        ax_el.set_xlabel("Number of Clusters (K)")
        ax_el.set_ylabel("Inertia Metric")
        ax_el.set_xticks(k_vals)
        ax_el.legend()
        st.pyplot(fig_el)
        
    with col_y:
        st.subheader("Silhouette Optimization Scan")
        fig_sil, ax_sil = plt.subplots(figsize=(6, 4))
        ax_sil.plot(k_vals, silhouettes, marker='o', linestyle='-', color='g')
        ax_sil.axvline(x=k_selected, color='r', linestyle='--', label=f"Active K={k_selected}")
        ax_sil.set_xlabel("Number of Clusters (K)")
        ax_sil.set_ylabel("Silhouette Score")
        ax_sil.set_xticks(k_vals)
        ax_sil.legend()
        st.pyplot(fig_sil)

with tab3:
    st.subheader("Clustered Dataset Records (First 50 Rows Sample View)")
    result_df = df.copy()
    result_df.insert(result_df.columns.get_loc('Species'), 'Assigned_Cluster_ID', cluster_labels)
    result_df.insert(result_df.columns.get_loc('Species'), 'Mapped_Predicted_Species', [class_names[i] for i in mapped_labels])
    st.dataframe(result_df.head(50), use_container_width=True)
