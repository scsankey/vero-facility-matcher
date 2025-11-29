"""
Debug Page for VERO Streamlit App  
Tests Google AI (Gemini) API integration
"""

import streamlit as st
import json
from datetime import datetime


def render_debug_page():
    """Render the Google AI API debug page"""
    
    st.title("🔍 Google AI (Gemini) API Debugger")
    st.markdown("---")
    
    st.info("""
    **This page will help diagnose issues with Google AI (Gemini) API integration.**
    
    It will test:
    1. ✅ API key validation
    2. 🔍 Model availability
    3. 🔍 Text generation
    4. 📊 Your current configuration
    """)
    
    # Get credentials from secrets
    try:
        google_api_key = st.secrets["google_ai"]["api_key"]
        google_model = st.secrets["google_ai"]["model"]
        
        st.success("✅ Secrets loaded successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Key Length", f"{len(google_api_key)} chars")
        with col2:
            st.metric("Model", google_model)
            
    except Exception as e:
        st.error(f"❌ **Error loading secrets:** {e}")
        st.warning("Make sure your `.streamlit/secrets.toml` has the `[google_ai]` section!")
        st.code("""[google_ai]
api_key = "YOUR_GOOGLE_AI_STUDIO_KEY"
model = "gemini-2.0-flash-exp"
""", language="toml")
        return
    
    st.markdown("---")
    
    # Run tests button
    if st.button("🚀 Run All Diagnostic Tests", type="primary"):
        run_all_tests(google_api_key, google_model)


def run_all_tests(google_api_key, google_model):
    """Run all diagnostic tests"""
    
    results = {
        'key_valid': False,
        'model_works': False,
        'generation_works': False,
    }
    
    # Test 1: API Key Format
    with st.expander("📋 **TEST 1: API Key Format Check**", expanded=True):
        results['key_valid'] = test_api_key_format(google_api_key)
    
    if not results['key_valid']:
        st.error("🔴 **API key format issue! Check your key.**")
        return
    
    # Test 2: Model Availability
    with st.expander("🔍 **TEST 2: Model Availability**", expanded=True):
        results['model_works'] = test_model_availability(google_api_key, google_model)
    
    # Test 3: Text Generation
    with st.expander("🔍 **TEST 3: Text Generation Test**", expanded=True):
        results['generation_works'] = test_text_generation(google_api_key, google_model)
    
    # Summary and Recommendations
    st.markdown("---")
    show_summary_and_recommendations(results, google_api_key, google_model)


def test_api_key_format(google_api_key):
    """Test 1: Check API key format"""
    
    st.write("🔍 **Checking API key format...**")
    
    # Check key format
    if not google_api_key or len(google_api_key) < 30:
        st.error(f"❌ API key seems too short ({len(google_api_key)} chars)")
        st.warning("Google AI Studio API keys are typically 39 characters long")
        return False
    
    if " " in google_api_key or "\n" in google_api_key:
        st.error("❌ API key contains spaces or newlines!")
        return False
    
    st.success(f"✅ API key format looks valid ({len(google_api_key)} characters)")
    return True


def test_model_availability(google_api_key, google_model):
    """Test 2: Check if model is available"""
    
    st.write(f"🔍 **Testing model: {google_model}**")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=google_api_key)
        
        # Try to create model instance
        model = genai.GenerativeModel(google_model)
        
        st.success(f"✅ **Model {google_model} is available!**")
        return True
        
    except ImportError:
        st.error("❌ **google-generativeai library not installed**")
        st.code("pip install google-generativeai")
        return False
        
    except Exception as e:
        st.error(f"❌ **Error:** {type(e).__name__}")
        st.code(f"Details: {str(e)}")
        
        if "API_KEY_INVALID" in str(e) or "invalid" in str(e).lower():
            st.warning("""
**Your API key appears to be invalid.**

**Steps to fix:**
1. Go to https://aistudio.google.com/apikey
2. Create a new API key
3. Copy it carefully (no spaces!)
4. Update your Streamlit secrets
""")
        
        return False


def test_text_generation(google_api_key, google_model):
    """Test 3: Test text generation"""
    
    st.write(f"🔍 **Testing text generation with {google_model}...**")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(google_model)
        
        # Simple test prompt
        test_prompt = "Say 'Hello, I am working!' in a friendly way."
        
        st.write("Sending test prompt...")
        
        response = model.generate_content(
            test_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 100,
            }
        )
        
        if response.text:
            st.success("✅ **Text generation WORKS!**")
            st.code(f"Response: {response.text}")
            return True
        else:
            st.warning("⚠️ **Got empty response**")
            return False
            
    except Exception as e:
        st.error(f"❌ **Error:** {type(e).__name__}")
        st.code(f"Details: {str(e)}")
        
        # Provide specific guidance
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            st.warning("""
**Rate limit or quota exceeded.**

**Free tier limits:**
- 15 requests per minute
- 1 million tokens per day

**Wait a few minutes and try again.**
""")
        elif "SAFETY" in str(e):
            st.warning("Response blocked by safety filters (normal for some prompts)")
        
        return False


def show_summary_and_recommendations(results, google_api_key, google_model):
    """Show test summary and recommendations"""
    
    st.header("📊 Test Results Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if results['key_valid']:
            st.success("✅ API Key Valid")
        else:
            st.error("❌ API Key Invalid")
    
    with col2:
        if results['model_works']:
            st.success("✅ Model Available")
        else:
            st.error("❌ Model Unavailable")
    
    with col3:
        if results['generation_works']:
            st.success("✅ Generation Works")
        else:
            st.error("❌ Generation Fails")
    
    st.markdown("---")
    st.header("🎯 Recommendations")
    
    if not results['key_valid']:
        st.error("🔴 **CRITICAL: Your API key is invalid!**")
        st.markdown("""
**Action Items:**
1. Go to: https://aistudio.google.com/apikey
2. Click "Create API key"
3. Copy the key (should be ~39 characters)
4. Update Streamlit secrets:

```toml
[google_ai]
api_key = "YOUR_KEY_HERE"
model = "gemini-2.0-flash-exp"
```
""")
        return
    
    if results['key_valid'] and results['model_works'] and results['generation_works']:
        st.success("✅ **Everything is working perfectly!**")
        st.markdown(f"""
**Your Google AI configuration:**

```toml
[google_ai]
api_key = "{google_api_key[:10]}...{google_api_key[-5:]}"
model = "{google_model}"
```

**You're ready to use AI chat and storytelling!**

Go to the **VAS** tab and try:
- Deep Dive Chat: Ask about your crop data
- AI Storytelling: Generate value chain stories
""")
        return
    
    if results['model_works'] and not results['generation_works']:
        st.warning("⚠️ **Model available but generation failed**")
        st.markdown("""
**Possible causes:**
1. Rate limit exceeded (wait a few minutes)
2. Safety filters blocking response
3. Network issues

**Try:**
- Wait 5 minutes and test again
- Check your internet connection
- Try a different model: `gemini-2.0-flash-exp`
""")
        return
    
    st.error("🔴 **Configuration issues detected**")
    st.markdown("""
**Recommended actions:**
1. Verify API key is correct
2. Check model name is valid
3. Ensure internet connectivity
4. Try creating a new API key
""")


# For testing standalone
if __name__ == "__main__":
    render_debug_page()
