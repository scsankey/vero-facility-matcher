
"""
app.py
VERO - Data Entity Matching Platform
Upload → Match → Canonical Entities → LLM Chat → Download
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import requests

# Import the VERO engine
from vero_engine import run_vero_pipeline

st.set_page_config(
    page_title="VERO - Entity Resolution",
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

def call_llm_free(prompt: str) -> str:
    """
    Call a free/local LLM endpoint
    
    Configure via Streamlit secrets:
      - LLM_API_URL: endpoint URL
      - LLM_API_KEY: optional API key
    """
    api_url = st.secrets.get("LLM_API_URL", None)
    api_key = st.secrets.get("LLM_API_KEY", None)

    if not api_url:
        return (
            "❌ LLM backend not configured. "
            "Set LLM_API_URL (and optionally LLM_API_KEY) in Streamlit secrets."
        )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"prompt": prompt, "max_tokens": 512}

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "text" in data:
            return data["text"]
        elif "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "").strip()
        else:
            return str(data)
    except Exception as e:
        return f"❌ Error calling LLM: {e}"

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    st.subheader("Matching Thresholds")
    high_threshold = st.slider(
        "High Confidence", 0.7, 1.0, 0.90, 0.05,
        help="Matches above this are auto-accepted"
    )
    
    medium_threshold = st.slider(
        "Medium Confidence", 0.6, 0.9, 0.75, 0.05,
        help="Medium confidence requires strong name match"
    )
    
    st.markdown("---")
    
    st.subheader("Blocking Settings")
    district_threshold = st.slider(
        "District Match", 0.5, 1.0, 0.75, 0.05,
        help="Higher = stricter district matching"
    )
    
    st.markdown("---")
    
    st.subheader("About VERO")
    st.info("""
    **VERO Entity Resolution**
    
    Match facility records across sources and create a canonical identity fabric for LLMs and analytics.
    
    📊 Upload data  
    🤖 AI matching  
    🧩 Canonical entities  
    💬 LLM chat interface  
    """)

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<div class="main-header">🏥 VERO Entity Resolution</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Canonical Identity Fabric for LLMs & Analytics</div>', unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

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
        help="Should contain: Government registry, NGO Dataset, WhatsApp Dataset"
    )
    
    if excel_file:
        try:
            excel_sheets = pd.ExcelFile(excel_file).sheet_names
            st.success(f"✅ Found {len(excel_sheets)} sheets")
            
            gov_sheet = st.selectbox("Government Registry", excel_sheets, 
                                    index=excel_sheets.index('Government registry') if 'Government registry' in excel_sheets else 0)
            ngo_sheet = st.selectbox("NGO Dataset", excel_sheets,
                                    index=excel_sheets.index('NGO Dataset') if 'NGO Dataset' in excel_sheets else 0)
            wa_sheet = st.selectbox("WhatsApp Dataset", excel_sheets,
                                   index=excel_sheets.index('WhatsApp Dataset') if 'WhatsApp Dataset' in excel_sheets else 0)
            
            has_ground_truth = st.checkbox("Include Ground Truth for Training")
            gt_sheet = None
            if has_ground_truth:
                gt_sheet = st.selectbox("Ground Truth", excel_sheets,
                                       index=excel_sheets.index('Sankey GTP') if 'Sankey GTP' in excel_sheets else 0)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Option B: Separate CSV Files")
    gov_csv = st.file_uploader("Government APIs, IoT", type=['csv'], key="gov")
    ngo_csv = st.file_uploader("NGO pdf, email,etc", type=['csv'], key="ngo")
    wa_csv = st.file_uploader("WhatsApp etc", type=['csv'], key="wa")
    gt_csv = st.file_uploader("Ground Truth CSV (optional)", type=['csv'], key="gt")

# ============================================================================
# STEP 2: LOAD, VALIDATE & PREVIEW
# ============================================================================

if excel_file or (gov_csv and ngo_csv and wa_csv):
    st.markdown("---")
    st.header("📊 Step 2: Data Preview")
    
    # Load data
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
    
    # Validate
    valid_gov = validate_dataframe(gov_df, ['RecordID', 'OfficialFacilityName', 'District'], "Government")
    valid_ngo = validate_dataframe(ngo_df, ['RecordID', 'FacilityName', 'District'], "NGO")
    valid_wa = validate_dataframe(wa_df, ['RecordID', 'RelatedFacility', 'DistrictNote'], "WhatsApp")
    
    if valid_gov and valid_ngo and valid_wa:
        # Preview tabs
        t1, t2, t3 = st.tabs(["Government", "NGO", "WhatsApp"])
        
        with t1:
            st.dataframe(gov_df.head(10), use_container_width=True)
            st.caption(f"Total: {len(gov_df)} records")
        
        with t2:
            st.dataframe(ngo_df.head(10), use_container_width=True)
            st.caption(f"Total: {len(ngo_df)} records")
        
        with t3:
            st.dataframe(wa_df.head(10), use_container_width=True)
            st.caption(f"Total: {len(wa_df)} records")
        
        # ============================================================================
        # STEP 3: RUN MATCHING
        # ============================================================================
        
        st.markdown("---")
        st.header("🚀 Step 3: Run Entity Matching")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Government", len(gov_df))
        with col2:
            st.metric("NGO", len(ngo_df))
        with col3:
            st.metric("WhatsApp", len(wa_df))
        
        if st.button("🎯 Start Matching", type="primary", use_container_width=True):
            with st.spinner("Running VERO pipeline..."):
                try:
                    results = run_vero_pipeline(
                        gov_df=gov_df,
                        ngo_df=ngo_df,
                        whatsapp_df=wa_df,
                        ground_truth_df=gt_df
                    )
                    
                    st.session_state.results = results
                    st.success("✅ Matching complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# STEP 4: DISPLAY RESULTS
# ============================================================================

if st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    st.header("📈 Step 4: Results & Insights")
    
    # Extract data
    canonical = results.get("canonical_entities", pd.DataFrame())
    matched = results.get("matched_pairs", pd.DataFrame())
    clusters = results.get("clusters", pd.DataFrame())
    unified = results.get("unified", pd.DataFrame())
    metrics = results.get("metrics", {})

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Canonical Entities", len(canonical))
    with col2:
        st.metric("Matched Pairs", len(matched))
    with col3:
        st.metric("Clusters", clusters["ClusterID"].nunique() if len(clusters) > 0 else 0)
    with col4:
        st.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🎯 Matched Pairs",
        "🧩 Canonical Entities",
        "📁 Download",
        "💬 LLM Chat",
        "🧪 Simulations & APIs"
    ])
    
    # ----------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # ----------------------------------------------------------------------
    with tab1:
        st.subheader("Matching Overview")
        
        if len(matched) > 0 and "match_prob" in matched.columns:
            fig = px.histogram(
                matched, x='match_prob', nbins=20,
                title="Match Probability Distribution",
                labels={'match_prob': 'Probability', 'count': 'Pairs'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if len(clusters) > 0 and "Source" in clusters.columns:
            source_dist = clusters["Source"].value_counts()
            fig = px.pie(
                values=source_dist.values,
                names=source_dist.index,
                title="Records by Source"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------------
    # TAB 2: MATCHED PAIRS
    # ----------------------------------------------------------------------
    with tab2:
        st.subheader("High-Confidence Matched Pairs")
        if len(matched) > 0:
            cols = ['record_A', 'record_B', 'name_A', 'name_B', 'source_A', 'source_B', 'match_prob']
            display_cols = [c for c in cols if c in matched.columns]
            st.dataframe(matched[display_cols].head(200), use_container_width=True)
        else:
            st.info("No matched pairs found")

    # ----------------------------------------------------------------------
    # TAB 3: CANONICAL ENTITIES (ONLY TABLE WE SHOW)
    # ----------------------------------------------------------------------
    with tab3:
        st.subheader("🧩 Canonical Entities (Intel. Services-Ready Identity Table)")
        st.caption("One row per real-world entity, deduplicated across all sources")
        
        if len(canonical) > 0:
            st.dataframe(canonical, use_container_width=True)
            st.caption(f"Total canonical entities: {len(canonical)}")
            
            # Entity type breakdown
            if "EntityType" in canonical.columns:
                type_counts = canonical["EntityType"].value_counts()
                st.markdown("**Entity Types:**")
                for entity_type, count in type_counts.items():
                    st.write(f"- {entity_type}: {count}")
        else:
            st.info("No canonical entities. Run matching first.")

    # ----------------------------------------------------------------------
    # TAB 4: DOWNLOAD
    # ----------------------------------------------------------------------
    with tab4:
        st.subheader("📥 Download Results")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### Excel (All Sheets)")
            if st.button("Generate Excel", use_container_width=True):
                excel_data = to_excel({
                    'Canonical Entities': canonical,
                    'Matched Pairs': matched,
                    'All Clusters': clusters,
                    'Unified Dataset': unified,
                })
                st.download_button(
                    "⬇️ Download Excel",
                    excel_data,
                    "vero_results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col_right:
            st.markdown("### Individual CSVs")
            
            if len(canonical) > 0:
                st.download_button(
                    "Canonical Entities",
                    canonical.to_csv(index=False),
                    "canonical_entities.csv",
                    use_container_width=True
                )
            
            if len(matched) > 0:
                st.download_button(
                    "Matched Pairs",
                    matched.to_csv(index=False),
                    "matched_pairs.csv",
                    use_container_width=True
                )
            
            if len(clusters) > 0:
                st.download_button(
                    "All Clusters",
                    clusters.to_csv(index=False),
                    "clusters.csv",
                    use_container_width=True
                )

    # ----------------------------------------------------------------------
    # TAB 5: LLM CHAT
    # ----------------------------------------------------------------------
    with tab5:
        st.subheader("💬 LLM Chat on Canonical Entities")

        if len(canonical) == 0:
            st.info("No canonical entities. Run matching first.")
        else:
            user_question = st.text_input("Ask about a facility, person, or district:")

            if user_question:
                # Search canonical entities
                match_mask = (
                    canonical["CanonicalName"].fillna("").str.contains(user_question, case=False) |
                    canonical["Aliases"].fillna("").str.contains(user_question, case=False)
                )
                candidates = canonical[match_mask]

                if len(candidates) == 0:
                    st.warning("No match found. Try different spelling.")
                else:
                    st.markdown("### 🎯 Select Entity")

                    options = [
                        f"{row['GoldenID']} | {row['CanonicalName']} ({row['EntityType']}, {row['MainDistrict']})"
                        for _, row in candidates.iterrows()
                    ]
                    selection = st.selectbox("Found these matches:", options)

                    selected = candidates.iloc[options.index(selection)]
                    st.markdown(f"**Selected:** `{selected['CanonicalName']}` ({selected['EntityType']})")
                    st.caption(
                        f"ID: {selected['GoldenID']} | District: {selected['MainDistrict']} | "
                        f"Sources: {selected['SourcesRepresented']}"
                    )

                    # Get underlying records
                    if len(clusters) > 0 and "ClusterID" in clusters.columns:
                        cluster_id = selected["ClusterID"]
                        raw = clusters[clusters["ClusterID"] == cluster_id]

                        st.markdown("#### Context Records")
                        show_cols = [c for c in ["RecordID", "Source", "Name", "District", "Phone"] if c in raw.columns]
                        st.dataframe(raw[show_cols] if show_cols else raw, use_container_width=True)

                        # Build context
                        ctx_lines = [
                            f"Entity: {selected['CanonicalName']} (type={selected['EntityType']}, district={selected['MainDistrict']})",
                            f"Sources: {selected['SourcesRepresented']}",
                            f"Total records: {int(selected['RecordCount'])}"
                        ]
                        
                        for _, r in raw.head(10).iterrows():
                            name = r.get("Name") or r.get("AltName", "")
                            dist = r.get("District", "")
                            src = r.get("Source", "")
                            ctx_lines.append(f"- [{src}] {name}, {dist}")

                        context = "\n".join(ctx_lines)

                        st.markdown("#### LLM Context")
                        st.code(context)

                        prompt = f"""You are a health data assistant.

Given this canonical entity and records, answer the user's question factually using ONLY this context.

CONTEXT:
{context}

QUESTION:
{user_question}

ANSWER (2-4 paragraphs):
"""

                        if st.button("🧠 Ask LLM", type="primary"):
                            with st.spinner("Querying LLM..."):
                                answer = call_llm_free(prompt)
                            st.markdown("#### 🧠 Answer")
                            st.write(answer)

    # ----------------------------------------------------------------------
    # TAB 6: SIMULATIONS & APIS (PLACEHOLDER)
    # ----------------------------------------------------------------------
    with tab6:
        st.subheader("🧪 Simulations & External APIs")
        st.info("Placeholder for future integrations: system dynamics, impact metrics, external APIs etc")

        col_sim, col_api = st.columns(2)

        with col_sim:
            st.markdown("### 📊 Simulations")
            st.button("Run Scenario Simulation", disabled=True)
            st.button("Stress Test: Facility Overload", disabled=True)
            st.button("Forecast Stockouts", disabled=True)

        with col_api:
            st.markdown("### 🌐 External APIs")
            st.button("Sync External Registry", disabled=True)
            st.button("Pull Ground-Truth Data", disabled=True)
            st.button("Push to Partner System", disabled=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>VERO - Entity Resolution Platform</strong></p>
    <p>Canonical Identity Fabric for LLMs, Analytics & RWA</p>
    <p>© 2025</p>
</div>
""", unsafe_allow_html=True)
