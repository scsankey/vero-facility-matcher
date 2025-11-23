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
    st.header("📈 Step 4: Results & Insights")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Golden Records", len(results['golden']))
    with col2:
        st.metric("Matched Pairs", len(results['matched_pairs']))
    with col3:
        st.metric("Clusters", results['clusters']['ClusterID'].nunique() if len(results['clusters']) > 0 else 0)
    with col4:
        st.metric("Model ROC-AUC", f"{results['metrics'].get('roc_auc', 0):.3f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Matched Pairs", "🏥 Golden Records", "📁 Download"])
    
    with tab1:
        st.subheader("Matching Overview")
        
        if len(results['matched_pairs']) > 0:
            # Match probability distribution
            fig = px.histogram(
                results['matched_pairs'],
                x='match_prob',
                nbins=20,
                title="Match Probability Distribution",
                labels={'match_prob': 'Match Probability', 'count': 'Number of Pairs'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Source distribution
            if len(results['clusters']) > 0:
                source_dist = results['clusters']['Source'].value_counts()
                fig = px.pie(
                    values=source_dist.values,
                    names=source_dist.index,
                    title="Records by Source"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("High-Confidence Matches")
        if len(results['matched_pairs']) > 0:
            display_cols = ['record_A', 'record_B', 'name_A', 'name_B', 'source_A', 'source_B', 'match_prob']
            available_cols = [col for col in display_cols if col in results['matched_pairs'].columns]
            st.dataframe(
                results['matched_pairs'][available_cols].head(50),
                use_container_width=True
            )
        else:
            st.info("No matched pairs found")
    
    with tab3:
        st.subheader("Golden Facility Records")
        if len(results['golden']) > 0:
            st.dataframe(results['golden'], use_container_width=True)
            
            # Cluster size distribution
            if len(results['clusters']) > 0:
                cluster_sizes = results['clusters'].groupby('ClusterID').size()
                fig = px.histogram(
                    x=cluster_sizes.values,
                    nbins=10,
                    title="Cluster Size Distribution",
                    labels={'x': 'Records per Cluster', 'y': 'Number of Clusters'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No golden records generated")
    
    with tab4:
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 Download All Results (Excel)")
            
            if st.button("Generate Excel File", use_container_width=True):
                excel_data = to_excel({
                    'Golden Records': results['golden'],
                    'Matched Pairs': results['matched_pairs'],
                    'All Clusters': results['clusters'],
                    'Unified Dataset': results['unified']
                })
                
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_data,
                    file_name="vero_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("### 📥 Download Individual Files (CSV)")
            
            if len(results['golden']) > 0:
                st.download_button(
                    "Golden Records CSV",
                    results['golden'].to_csv(index=False),
                    "golden_records.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            if len(results['matched_pairs']) > 0:
                st.download_button(
                    "Matched Pairs CSV",
                    results['matched_pairs'].to_csv(index=False),
                    "matched_pairs.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            if len(results['clusters']) > 0:
                st.download_button(
                    "All Clusters CSV",
                    results['clusters'].to_csv(index=False),
                    "clusters.csv",
                    "text/csv",
                    use_container_width=True
                )

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
