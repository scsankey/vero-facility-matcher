"""
Debug Page for VERO Streamlit App
Add this as a new tab to diagnose HuggingFace API issues
"""

import streamlit as st
import requests
import json
from datetime import datetime


def render_debug_page():
    """Render the HuggingFace API debug page"""
    
    st.title("🔍 HuggingFace API Debugger")
    st.markdown("---")
    
    st.info("""
    **This page will help diagnose issues with HuggingFace API integration.**
    
    It will test:
    1. ✅ Token validation
    2. 🔍 Old Inference API (api-inference.huggingface.co)
    3. 🔍 Router API (router.huggingface.co)
    4. 🔍 InferenceClient library
    5. 📊 Model accessibility
    """)
    
    # Get token from secrets
    try:
        hf_token = st.secrets["huggingface"]["token"]
        hf_model = st.secrets["huggingface"]["model"]
        hf_api_url = st.secrets["huggingface"].get("api_url", "N/A - Using InferenceClient")  # Optional now
        
        st.success("✅ Secrets loaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Token Length", f"{len(hf_token)} chars")
        with col2:
            st.metric("Model", hf_model)
        with col3:
            token_preview = f"{hf_token[:10]}...{hf_token[-5:]}" if len(hf_token) > 15 else hf_token
            st.metric("Token Preview", token_preview)
            
    except Exception as e:
        st.error(f"❌ **Error loading secrets:** {e}")
        st.warning("Make sure your `.streamlit/secrets.toml` is configured!")
        return
    
    st.markdown("---")
    
    # Run tests button
    if st.button("🚀 Run All Diagnostic Tests", type="primary"):
        run_all_tests(hf_token, hf_model, hf_api_url)


def run_all_tests(hf_token, hf_model, hf_api_url):
    """Run all diagnostic tests"""
    
    results = {
        'token_valid': False,
        'old_api_works': False,
        'router_works': False,
        'client_works': False,
    }
    
    # Test 1: Token Validation
    with st.expander("📋 **TEST 1: Token Validation**", expanded=True):
        results['token_valid'] = test_token_validation(hf_token)
    
    if not results['token_valid']:
        st.error("🔴 **Token is invalid! Fix this first before running other tests.**")
        return
    
    # Test 2: Old Inference API
    with st.expander("🔍 **TEST 2: Old Inference API** (api-inference.huggingface.co)", expanded=True):
        results['old_api_works'] = test_old_inference_api(hf_token)
    
    # Test 3: Router API
    with st.expander("🔍 **TEST 3: Router API** (router.huggingface.co)", expanded=True):
        results['router_works'] = test_router_api(hf_token)
    
    # Test 4: InferenceClient
    with st.expander("🔍 **TEST 4: InferenceClient Library**", expanded=True):
        results['client_works'] = test_inference_client(hf_token)
    
    # Test 5: Current Configuration
    with st.expander("⚙️ **TEST 5: Your Current Configuration**", expanded=True):
        test_current_config(hf_token, hf_api_url)
    
    # Summary and Recommendations
    st.markdown("---")
    show_summary_and_recommendations(results, hf_token)


def test_token_validation(hf_token):
    """Test 1: Validate HuggingFace token"""
    
    st.write("🔍 **Checking token format...**")
    
    # Format checks
    issues = []
    
    if not hf_token.startswith("hf_"):
        issues.append("⚠️ Token should start with 'hf_'")
    else:
        st.success("✅ Token starts with 'hf_'")
    
    if len(hf_token) < 30:
        issues.append(f"⚠️ Token seems short ({len(hf_token)} chars)")
    else:
        st.success(f"✅ Token length: {len(hf_token)} characters")
    
    if " " in hf_token or "\n" in hf_token:
        issues.append("❌ Token contains spaces/newlines!")
        for issue in issues:
            st.error(issue)
        return False
    
    if issues:
        for issue in issues:
            st.warning(issue)
    
    # Test with API
    st.write("🔍 **Testing token with HuggingFace API...**")
    
    url = "https://huggingface.co/api/whoami-v2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ **TOKEN IS VALID!**")
            
            # Show details
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Token Details:**")
                st.json({
                    "type": data.get('type', 'unknown'),
                    "name": data.get('name', 'unknown'),
                })
            
            with col2:
                auth_info = data.get('auth', {})
                if auth_info:
                    token_info = auth_info.get('accessToken', {})
                    st.write("**Permissions:**")
                    st.json({
                        "role": token_info.get('role', 'unknown'),
                        "display_name": token_info.get('displayName', 'unknown'),
                    })
            
            return True
            
        elif response.status_code == 401:
            st.error("❌ **Invalid token or unauthorized**")
            st.code(response.text)
            return False
            
        else:
            st.warning(f"⚠️ Unexpected status: {response.status_code}")
            st.code(response.text)
            return False
            
    except Exception as e:
        st.error(f"❌ **Error:** {e}")
        return False


def test_old_inference_api(hf_token):
    """Test 2: Old Inference API"""
    
    models = ["distilgpt2", "gpt2"]
    
    for model in models:
        st.write(f"🧪 **Testing: {model}**")
        
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": "Hello",
            "parameters": {"max_new_tokens": 10}
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            st.code(f"URL: {url}\nStatus: {response.status_code}")
            
            if response.status_code == 200:
                st.success("✅ **SUCCESS! This API works!**")
                result = response.json()
                st.json(result)
                return True
                
            elif response.status_code == 503:
                st.info("⏳ **Model is loading** (this means endpoint WORKS!)")
                st.code(response.text)
                return True
                
            elif response.status_code == 410:
                st.error("❌ **410 DEPRECATED**")
                st.code(response.text)
                if "router" in response.text:
                    st.info("💡 Response suggests using router.huggingface.co")
                    
            elif response.status_code == 401:
                st.error("❌ **UNAUTHORIZED**")
                st.code(response.text)
                
            else:
                st.warning(f"⚠️ Error {response.status_code}")
                st.code(response.text)
                
        except Exception as e:
            st.error(f"❌ Exception: {e}")
    
    return False


def test_router_api(hf_token):
    """Test 3: Router API"""
    
    models = ["distilgpt2", "gpt2"]
    
    for model in models:
        st.write(f"🧪 **Testing: {model}**")
        
        url = f"https://router.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": "Hello",
            "parameters": {"max_new_tokens": 10}
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            st.code(f"URL: {url}\nStatus: {response.status_code}")
            
            if response.status_code == 200:
                st.success("✅ **SUCCESS! Router API works!**")
                result = response.json()
                st.json(result)
                return True
                
            elif response.status_code == 503:
                st.info("⏳ **Model is loading** (endpoint works!)")
                st.code(response.text)
                return True
                
            elif response.status_code == 404:
                st.error("❌ **404 NOT FOUND**")
                st.code(response.text)
                
            elif response.status_code == 401:
                st.error("❌ **UNAUTHORIZED**")
                st.code(response.text)
                if "Invalid username or password" in response.text:
                    st.warning("💡 This is the error you mentioned!")
                    
            else:
                st.warning(f"⚠️ Error {response.status_code}")
                st.code(response.text)
                
        except Exception as e:
            st.error(f"❌ Exception: {e}")
    
    return False


def test_inference_client(hf_token):
    """Test 4: InferenceClient"""
    
    try:
        from huggingface_hub import InferenceClient
        st.success("✅ huggingface_hub library is installed")
    except ImportError:
        st.error("❌ huggingface_hub not installed")
        st.code("pip install huggingface_hub")
        return False
    
    st.write("🧪 **Testing InferenceClient...**")
    
    try:
        client = InferenceClient(token=hf_token)
        st.success("✅ Client created successfully")
        
        st.write("🧪 **Attempting text generation...**")
        
        response = client.text_generation(
            "Hello, how are you?",
            model="distilgpt2",
            max_new_tokens=20,
        )
        
        st.success("✅ **SUCCESS! InferenceClient works!**")
        st.code(response)
        return True
        
    except Exception as e:
        st.error(f"❌ **Error:** {e}")
        st.code(f"Error type: {type(e).__name__}")
        return False


def test_current_config(hf_token, hf_api_url):
    """Test 5: Test current configuration"""
    
    if hf_api_url == "N/A - Using InferenceClient":
        st.info("ℹ️ **No api_url configured** - Your app uses InferenceClient (recommended)")
        st.write("InferenceClient handles routing automatically, no URL needed!")
        
        # Test InferenceClient directly
        st.write("🧪 **Testing InferenceClient with your configuration...**")
        
        try:
            from huggingface_hub import InferenceClient
            
            # Get model from secrets
            hf_model = st.secrets["huggingface"]["model"]
            
            client = InferenceClient(token=hf_token)
            
            response = client.text_generation(
                "Hello",
                model=hf_model,
                max_new_tokens=10,
            )
            
            st.success("✅ **Your configuration WORKS!**")
            st.code(f"Response: {response}")
            
        except Exception as e:
            st.error(f"❌ **Error:** {e}")
        
        return
    
    st.write(f"🧪 **Testing your current API URL:**")
    st.code(hf_api_url)
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": "Hello",
        "parameters": {"max_new_tokens": 10}
    }
    
    try:
        response = requests.post(hf_api_url, headers=headers, json=payload, timeout=30)
        
        st.write(f"**Status Code:** {response.status_code}")
        
        if response.status_code == 200:
            st.success("✅ **Your current configuration WORKS!**")
            result = response.json()
            st.json(result)
            
        elif response.status_code == 503:
            st.info("⏳ **Model is loading** (your config works, just wait 20 sec)")
            st.code(response.text)
            
        else:
            st.error(f"❌ **Your current configuration FAILS** (Status {response.status_code})")
            st.code(response.text)
            
    except Exception as e:
        st.error(f"❌ **Exception:** {e}")


def show_summary_and_recommendations(results, hf_token):
    """Show test summary and recommendations"""
    
    st.header("📊 Test Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if results['token_valid']:
            st.success("✅ Token Valid")
        else:
            st.error("❌ Token Invalid")
    
    with col2:
        if results['old_api_works']:
            st.success("✅ Old API Works")
        else:
            st.error("❌ Old API Fails")
    
    with col3:
        if results['router_works']:
            st.success("✅ Router Works")
        else:
            st.error("❌ Router Fails")
    
    with col4:
        if results['client_works']:
            st.success("✅ Client Works")
        else:
            st.error("❌ Client Fails")
    
    st.markdown("---")
    st.header("🎯 Recommendations")
    
    if not results['token_valid']:
        st.error("🔴 **CRITICAL: Your token is invalid!**")
        st.markdown("""
        **Action Items:**
        1. Go to: https://huggingface.co/settings/tokens
        2. Create a NEW token with 'Read' permission
        3. Update Streamlit secrets with new token
        4. Restart app and run this debug again
        """)
        return
    
    if results['old_api_works']:
        st.success("✅ **Old API Works!**")
        st.markdown("**Update your Streamlit secrets to:**")
        st.code(f"""[huggingface]
token = "{hf_token[:10]}...{hf_token[-5:]}"
model = "distilgpt2"
api_url = "https://api-inference.huggingface.co/models/distilgpt2"
""", language="toml")
        return
    
    if results['router_works']:
        st.success("✅ **Router API Works!**")
        st.markdown("**Update your Streamlit secrets to:**")
        st.code(f"""[huggingface]
token = "{hf_token[:10]}...{hf_token[-5:]}"
model = "distilgpt2"
api_url = "https://router.huggingface.co/models/distilgpt2"
""", language="toml")
        return
    
    if results['client_works']:
        st.success("✅ **InferenceClient Works!**")
        st.markdown("""
        **Your app needs code changes to use InferenceClient.**
        
        Would you like me to provide the updated code?
        """)
        return
    
    st.error("🔴 **NONE of the APIs are working!**")
    st.markdown("""
    **Recommended Solutions:**
    
    1. **Use Fallback Responses** (Already in your app!)
       - Your app has curated responses that work without any API
       - Perfect for demos and testing
       - Just disable the LLM features temporarily
    
    2. **Try a Different Model**
       - Some models require accepting terms first
       - Visit model page and click "Agree and access"
    
    3. **Use Different Provider**
       - OpenAI API (costs ~$0.01 per query but very reliable)
       - Anthropic Claude API
       - Google Gemini API
    
    4. **Deploy Local Model**
       - Run model on your own server
       - I can help set this up
    """)


# For testing standalone
if __name__ == "__main__":
    render_debug_page()
