"""
app.py
VERO - Facility Entity Matching Platform
Upload → Clean → Match → Show Clusters → Download
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import sys

# Import the VERO engine
from vero_engine import run_vero_pipeline

st.set_page_config(
    page_title="VERO - Facility Matcher",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_excel_file(uploaded_file, sheet_name):
    """Load specific sheet from Excel file"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"Error loading sheet '{sheet_name}': {str(e)}")
        return None

def validate_dataframe(df, required_cols, dataset_name):
    """Validate that dataframe has required columns"""
    if df is None:
        return False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"{dataset_name} is missing columns: {', '.join(missing_cols)}")
        return False
    return True

def to_excel(dataframes_dict):
    """Convert multiple dataframes to Excel with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    st.subheader("Matching Thresholds")
    high_threshold = st.slider(
        "High Confidence",
        min_value=0.7,
        max_value=1.0,
        value=0.90,
        step=0.05,
        help="Matches above this threshold are automatically accepted"
    )
    
    medium_threshold = st.slider(
        "Medium Confidence",
        min_value=0.6,
        max_value=0.9,
        value=0.75,
        step=0.05,
        help="Matches in this range require strong name similarity"
    )
    
    st.markdown("---")
    
    st.subheader("Blocking Settings")
    district_threshold = st.slider(
        "District Match Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Higher = stricter district matching"
    )
    
    st.markdown("---")
    
    st.subheader("About VERO")
    st.info("""
    **VERO Entity Resolution**
    
    This tool helps match facility records across different data sources to create a unified master registry.
    
    📊 Upload your data  
    🤖 AI matches entities  
    📁 Download golden records
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<div class="main-header">🏥 VERO Facility Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload → Match → Download Golden Records</div>', unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.header("📤 Step 1: Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Option A: Single Excel File")
    excel_file = st.file_uploader(
        "Upload Excel with multiple sheets",
        type=['xlsx', 'xls'],
        help="Excel file should contain sheets: 'Government registry', 'NGO Dataset', 'WhatsApp Dataset', 'Sankey GTP' (optional ground truth)"
    )
    
    if excel_file:
        try:
            excel_sheets = pd.ExcelFile(excel_file).sheet_names
            st.success(f"✅ Found {len(excel_sheets)} sheets: {', '.join(excel_sheets)}")
            
            # Sheet name mapping
            st.subheader("Sheet Name Mapping")
            gov_sheet = st.selectbox("Government Registry Sheet", excel_sheets, index=excel_sheets.index('Government registry') if 'Government registry' in excel_sheets else 0)
            ngo_sheet = st.selectbox("NGO Dataset Sheet", excel_sheets, index=excel_sheets.index('NGO Dataset') if 'NGO Dataset' in excel_sheets else 0)
            wa_sheet = st.selectbox("WhatsApp Dataset Sheet", excel_sheets, index=excel_sheets.index('WhatsApp Dataset') if 'WhatsApp Dataset' in excel_sheets else 0)
            
            has_ground_truth = st.checkbox("Include Ground Truth for Training")
            gt_sheet = None
            if has_ground_truth:
                gt_sheet = st.selectbox("Ground Truth Sheet", excel_sheets, index=excel_sheets.index('Sankey GTP') if 'Sankey GTP' in excel_sheets else 0)
            
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")

with col2:
    st.subheader("Option B: Separate CSV Files")
    gov_csv = st.file_uploader("Government Registry CSV", type=['csv'], key="gov")
    ngo_csv = st.file_uploader("NGO Dataset CSV", type=['csv'], key="ngo")
    wa_csv = st.file_uploader("WhatsApp Dataset CSV", type=['csv'], key="wa")
    gt_csv = st.file_uploader("Ground Truth CSV (optional)", type=['csv'], key="gt")

# ============================================================================
# STEP 2: LOAD AND VALIDATE DATA
# ============================================================================

if excel_file or (gov_csv and ngo_csv and wa_csv):
    st.markdown("---")
    st.header("📊 Step 2: Data Preview")
    
    # Load data based on upload method
    if excel_file:
        gov_df = load_excel_file(excel_file, gov_sheet)
        ngo_df = load_excel_file(excel_file, ngo_sheet)
        wa_df = load_excel_file(excel_file, wa_sheet)
        gt_df = load_excel_file(excel_file, gt_sheet) if has_ground_truth and gt_sheet else None
    else:
        gov_df = pd.read_csv(gov_csv) if gov_csv else None
        ngo_df = pd.read_csv(ngo_csv) if ngo_csv else None
        wa_df = pd.read_csv(wa_csv) if wa_csv else None
        gt_df = pd.read_csv(gt_csv) if gt_csv else None
    
    # Validate data
    valid_gov = validate_dataframe(gov_df, ['RecordID', 'OfficialFacilityName', 'District'], "Government Registry")
    valid_ngo = validate_dataframe(ngo_df, ['RecordID', 'FacilityName', 'District'], "NGO Dataset")
    valid_wa = validate_dataframe(wa_df, ['RecordID', 'RelatedFacility', 'DistrictNote'], "WhatsApp Dataset")
    
    if valid_gov and valid_ngo and valid_wa:
        # Display preview
        tab1, tab2, tab3 = st.tabs(["Government", "NGO", "WhatsApp"])
        
        with tab1:
            st.dataframe(gov_df.head(10), use_container_width=True)
            st.caption(f"Total records: {len(gov_df)}")
        
        with tab2:
            st.dataframe(ngo_df.head(10), use_container_width=True)
            st.caption(f"Total records: {len(ngo_df)}")
        
        with tab3:
            st.dataframe(wa_df.head(10), use_container_width=True)
            st.caption(f"Total records: {len(wa_df)}")
        
        # ============================================================================
        # STEP 3: RUN MATCHING
        # ============================================================================
        
        st.markdown("---")
        st.header("🚀 Step 3: Run Entity Matching")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Government Records", len(gov_df))
        with col2:
            st.metric("NGO Records", len(ngo_df))
        with col3:
            st.metric("WhatsApp Records", len(wa_df))
        
        if st.button("🎯 Start Matching", type="primary", use_container_width=True):
            # Store record counts for summary view later
            st.session_state.gov_count = len(gov_df)
            st.session_state.ngo_count = len(ngo_df)
            st.session_state.wa_count = len(wa_df)
            
            st.session_state.processing = True
            
            with st.spinner("Running VERO matching pipeline..."):
                try:
                    results = run_vero_pipeline(
                        gov_df=gov_df,
                        ngo_df=ngo_df,
                        whatsapp_df=wa_df,
                        ground_truth_df=gt_df
                    )
                    
                    st.session_state.results = results
                    st.session_state.processing = False
                    st.success("✅ Matching complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during matching: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.processing = False

# ============================================================================
# STEP 4: DISPLAY RESULTS
# ============================================================================

if st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    st.header("📈 Step 4: Results & Clustering Overview")
    
    # ---------------------------------------------------------------------
    # 4.1 Summary metrics
    # ---------------------------------------------------------------------
    gov_count = st.session_state.get("gov_count", 0)
    ngo_count = st.session_state.get("ngo_count", 0)
    wa_count  = st.session_state.get("wa_count", 0)
    
    clusters_df = results.get("clusters", pd.DataFrame())
    golden_df   = results.get("golden", pd.DataFrame())
    matched_df  = results.get("matched_pairs", pd.DataFrame())
    metrics     = results.get("metrics", {})
    
    total_clusters = clusters_df["ClusterID"].nunique() if "ClusterID" in clusters_df.columns and len(clusters_df) > 0 else 0
    
    st.subheader("🔢 Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gov Records", gov_count)
    with col2:
        st.metric("NGO Records", ngo_count)
    with col3:
        st.metric("WhatsApp Records", wa_count)
    with col4:
        st.metric("Clusters Found", total_clusters)
    
    # ---------------------------------------------------------------------
    # 4.2 Golden facilities table (Table 1)
    # ---------------------------------------------------------------------
    st.markdown("### 🏥 Table 1 – Golden Facilities")
    
    if len(golden_df) > 0:
        # Pick only key columns if they exist
        display_cols = []
        for c in ["GoldenID", "OfficialName", "District", "Sources", "RecordCount"]:
            if c in golden_df.columns:
                display_cols.append(c)
        
        if display_cols:
            st.dataframe(golden_df[display_cols], use_container_width=True)
        else:
            st.dataframe(golden_df, use_container_width=True)
        
        st.caption(f"Total golden facility records: {len(golden_df)}")
    else:
        st.info("No golden records generated yet. Check your VERO pipeline output.")
    
    # ---------------------------------------------------------------------
    # 4.3 One sample cluster expanded (Table 2)
    # ---------------------------------------------------------------------
    st.markdown("### 🧬 Table 2 – Sample Cluster (How Clustering Works)")
    
    if len(clusters_df) > 0 and "ClusterID" in clusters_df.columns:
        # Let user pick a cluster to inspect
        cluster_ids = clusters_df["ClusterID"].unique()
        
        # Simple heuristic: preselect the largest cluster
        cluster_sizes = clusters_df.groupby("ClusterID").size().sort_values(ascending=False)
        default_cluster = cluster_sizes.index[0]
        
        selected_cluster = st.selectbox(
            "Select a ClusterID to inspect",
            options=list(cluster_ids),
            index=list(cluster_ids).index(default_cluster) if default_cluster in cluster_ids else 0
        )
        
        sample_cluster = clusters_df[clusters_df["ClusterID"] == selected_cluster].copy()
        
        st.write(f"Showing all records in **{selected_cluster}** "
                 f"({len(sample_cluster)} records):")
        
        # Try to show intuitive columns
        sample_cols = []
        for c in ["RecordID", "Source", "Name", "AltName", "District", "Phone"]:
            if c in sample_cluster.columns:
                sample_cols.append(c)
        
        if sample_cols:
            st.dataframe(sample_cluster[sample_cols], use_container_width=True)
        else:
            st.dataframe(sample_cluster, use_container_width=True)
        
        st.caption("Each row above is an original record from Gov/NGO/WhatsApp that VERO "
                   "decided belongs to the same real-world facility.")
    else:
        st.info("No cluster data available to display. Make sure your pipeline returns a 'clusters' DataFrame.")
    
    # ---------------------------------------------------------------------
    # 4.4 Download buttons
    # ---------------------------------------------------------------------
    st.markdown("### 📥 Step 5: Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Golden Facilities (CSV)**")
        if len(golden_df) > 0:
            st.download_button(
                "⬇️ Download Golden Facilities",
                golden_df.to_csv(index=False).encode("utf-8"),
                "golden_facilities.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.caption("No golden records to download.")
    
    with col2:
        st.markdown("**Master Entity Table (Clusters) (CSV)**")
        if len(clusters_df) > 0:
            st.download_button(
                "⬇️ Download Master Entity Table",
                clusters_df.to_csv(index=False).encode("utf-8"),
                "master_entity_table.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.caption("No clusters to download.")
    
    with col3:
        st.markdown("**Matched Pairs (CSV)**")
        if len(matched_df) > 0:
            st.download_button(
                "⬇️ Download Matched Pairs",
                matched_df.to_csv(index=False).encode("utf-8"),
                "matched_pairs.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.caption("No matched pairs to download.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>VERO - Entity Resolution Platform</strong></p>
    <p>Powered by AI | Built with Streamlit | © 2025</p>
</div>
""", unsafe_allow_html=True)
```

---

## 📄 **File 2: vero_engine.py** (No changes - already correct)

The vero_engine.py from the previous artifact is still correct and doesn't need any updates. It already returns the proper structure with `GoldenID`, `OfficialName`, `Sources`, `RecordCount`, etc.

---

## 📄 **File 3: requirements.txt** (No changes)
```
pandas
openpyxl
plotly
sentence-transformers
rapidfuzz
scikit-learn
networkx
torch
