import os
import requests
import io
import fitz  
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from PIL import Image
import cv2
import numpy as np
from googletrans import Translator
import time
from datetime import datetime
import json

# Load API keys
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

# Configure settings
REQUEST_TIMEOUT = 60  # Increased timeout
MAX_RETRIES = 3

# LLM provider configs - Fixed URLs and endpoints
LLM_PROVIDERS = {
    "huggingface": {
        "url": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        "headers": {"Authorization": f"Bearer {HF_TOKEN}"}
    },
    "gemini": {
        "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
        "headers": {"Content-Type": "application/json"}
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
    },
    "ollama": {
        "url": f"{OLLAMA_API_URL}/api/generate",
        "headers": {"Content-Type": "application/json"}
    }
}

# Enhanced Translation with caching
class InsuranceTranslator:
    def __init__(self):
        self.translator = Translator()
        self.cache = {}
        
    def translate(self, text, dest="hi"):
        if not text or dest == "en":
            return text
            
        cache_key = f"{dest}:{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            translated = self.translator.translate(text, dest=dest).text
            self.cache[cache_key] = translated
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return text

translator = InsuranceTranslator()

# Image Processing Functions
def enhance_image(image):
    try:
        img = np.array(image.convert('RGB'))
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return Image.fromarray(img)
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return image

def analyze_image(img):
    try:
        gray = np.array(img.convert('L'))
        edges = cv2.Canny(gray, 50, 150)
        density = edges.sum() / edges.size
        
        if density > 0.15: level, mult = 'severe', 2.5
        elif density > 0.08: level, mult = 'moderate', 1.5
        elif density > 0.03: level, mult = 'minor', 1.0
        else: level, mult = 'minimal', 0.5
            
        brightness = gray.mean()
        quality = min(100, max(0, 100 - abs(brightness - 128)/1.28))
        
        return {
            'damage_level': level,
            'cost_multiplier': mult,
            'edge_density': f"{density*100:.2f}%",
            'quality_score': f"{quality:.1f}/100"
        }
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return {
            'damage_level': 'unknown',
            'cost_multiplier': 1.0,
            'edge_density': "0%",
            'quality_score': "0/100"
        }

# Cost Estimation
CLAIM_TYPE_BASES = {
    "Vehicle": {"severe": (15000,40000), "moderate": (7000,15000), "minor": (1000,5000)},
    "Health": {"severe": (100000,500000), "moderate": (20000,50000), "minor": (5000,20000)},
    "Home": {"severe": (100000,500000), "moderate": (30000,100000), "minor": (5000,30000)}
}

def estimate_repair_cost(desc, analysis, claim_type):
    base = CLAIM_TYPE_BASES.get(claim_type, CLAIM_TYPE_BASES["Vehicle"])
    severity = next((k for k in ["severe","moderate","minor"] 
                   if k in desc.lower() or analysis["damage_level"]==k), "moderate")
    low, high = base.get(severity, base["moderate"])
    return (int(low * analysis["cost_multiplier"]), int(high * analysis["cost_multiplier"]))

# Health Document Processor
class HealthAgent:
    def __init__(self, lang):
        self.lang = lang
        
    def extract_text(self, file_bytes):
        try:
            with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            return f"PDF error: {e}"
            
    def identify_issues(self, text):
        issues = {
            "cash": "Cash payment detected",
            "duplicate": "Possible duplicate bill",
            "missing": "Missing information found",
            "expired": "Expired document detected"
        }
        return [issues[k] for k in issues if k in text.lower()]
        
    def generate_report(self, file_bytes, scenario):
        text = self.extract_text(file_bytes)
        flags = self.identify_issues(text)
        summary = text[:500] + ("..." if len(text) > 500 else "")
        
        report = (
            f"📖 Scenario: {scenario}\n\n"
            f"📄 Document Summary: {summary or 'No text found in document'}\n\n"
            f"🚩 Potential Issues Found: {', '.join(flags) if flags else 'No issues detected'}\n\n"
            f"✅ Status: {'Requires review' if flags else 'Document appears complete'}"
        )
        return translator.translate(report, self.lang)

# Enhanced Default Responses - More detailed and human-friendly
DEFAULT_RESPONSES = {
    "Vehicle": {
        "en": """For your vehicle insurance claim, here's what you need to do step by step:

📋 **Required Documents:**
• **FIR Copy** - File a police complaint within 24 hours of the accident
• **RC Book** - Registration certificate of your vehicle
• **Driving License** - Valid license of the person driving
• **Insurance Policy** - Your current policy document
• **Repair Estimates** - Get quotes from authorized service centers
• **Damage Photos** - Take clear pictures from multiple angles

🔄 **Claim Process:**
1. **Immediate Steps:** Ensure safety, call police if needed, take photos
2. **Contact Insurer:** Call your insurance company's helpline immediately
3. **Submit Documents:** Provide all required documents within 7 days
4. **Vehicle Inspection:** Insurance surveyor will inspect the damage
5. **Approval & Repair:** Once approved, get repairs done at authorized garage
6. **Settlement:** Cashless or reimbursement based on your policy

⏰ **Important Timelines:**
- Report accident: Within 24-48 hours
- Submit documents: Within 7 days
- Claim settlement: Usually 15-30 days after document submission

💡 **Pro Tips:**
- Keep original receipts safe
- Don't repair before inspection unless emergency
- Follow up regularly with your insurer""",
        
        "hi": """आपके वाहन बीमा दावे के लिए, यहाँ आपको चरणबद्ध जानकारी दी गई है:

📋 **आवश्यक दस्तावेज:**
• **FIR कॉपी** - दुर्घटना के 24 घंटे के भीतर पुलिस शिकायत दर्ज करें
• **RC बुक** - वाहन का पंजीकरण प्रमाणपत्र
• **ड्राइविंग लाइसेंस** - चालक का वैध लाइसेंस
• **बीमा पॉलिसी** - आपका वर्तमान पॉलिसी दस्तावेज
• **मरम्मत अनुमान** - अधिकृत सेवा केंद्रों से कोटेशन लें
• **क्षति की तस्वीरें** - विभिन्न कोणों से स्पष्ट तस्वीरें लें

🔄 **दावा प्रक्रिया:**
1. **तत्काल कदम:** सुरक्षा सुनिश्चित करें, आवश्यक हो तो पुलिस को कॉल करें
2. **बीमाकर्ता से संपर्क:** तुरंत अपनी बीमा कंपनी की हेल्पलाइन पर कॉल करें
3. **दस्तावेज जमा करें:** 7 दिनों के भीतर सभी आवश्यक दस्तावेज प्रदान करें
4. **वाहन निरीक्षण:** बीमा सर्वेयर क्षति का निरीक्षण करेगा
5. **अनुमोदन और मरम्मत:** अनुमोदन के बाद, अधिकृत गैरेज में मरम्मत कराएं"""
    },
    
    "Health": {
        "en": """For your health insurance claim, here's your complete guide:

📋 **Essential Documents Checklist:**
• **Hospital Bills** - Original bills and receipts for all treatments
• **Discharge Summary** - Complete medical summary from hospital
• **Doctor's Prescriptions** - All prescriptions during treatment
• **Diagnostic Reports** - Lab tests, X-rays, MRI, CT scans etc.
• **Policy Document** - Your health insurance policy copy
• **ID Proof** - Aadhar card, PAN card or passport
• **Claim Form** - Properly filled and signed claim form

🏥 **Two Types of Claims:**
**Cashless Treatment:**
- Show your health card at network hospitals
- Hospital directly settles with insurance company
- You pay only non-covered expenses

**Reimbursement Claims:**
- Pay hospital bills upfront
- Submit documents to insurance company
- Get money back after claim approval

⏰ **Important Deadlines:**
- Inform insurer: Within 24 hours for planned treatments
- Submit documents: Within 15-30 days of discharge
- Claim processing: Usually 15-30 days

💡 **Quick Tips:**
- Always inform your insurer before planned surgeries
- Keep all original bills and reports
- Check if hospital is in your network for cashless facility
- Maintain a health file with all medical records""",
        
        "hi": """आपके स्वास्थ्य बीमा दावे के लिए, यहाँ आपकी पूरी गाइड है:

📋 **आवश्यक दस्तावेजों की सूची:**
• **अस्पताल के बिल** - सभी उपचारों के मूल बिल और रसीदें
• **डिस्चार्ज समरी** - अस्पताल से पूरी चिकित्सा रिपोर्ट
• **डॉक्टर के नुस्खे** - उपचार के दौरान सभी दवाओं के नुस्खे
• **जांच रिपोर्ट** - लैब टेस्ट, एक्स-रे, MRI, CT स्कैन आदि
• **पॉलिसी दस्तावेज** - आपकी स्वास्थ्य बीमा पॉलिसी की कॉपी
• **पहचान प्रमाण** - आधार कार्ड, पैन कार्ड या पासपोर्ट"""
    },
    
    "Home": {
        "en": """For your home insurance claim, here's everything you need to know:

📋 **Required Documentation:**
• **Claim Form** - Properly filled insurance claim form
• **Policy Document** - Your current home insurance policy
• **Damage Photos** - Clear pictures of all damaged items/areas
• **Purchase Bills** - Original bills for damaged items (if available)
• **Repair Estimates** - Quotes from contractors for repair work
• **Police Report** - Required for theft, burglary, or vandalism claims
• **Fire Brigade Report** - For fire-related damage claims

🏠 **Types of Home Insurance Claims:**
**Property Damage:** Structure damage due to fire, earthquake, floods
**Contents Claim:** Damage to furniture, electronics, personal belongings
**Theft Claims:** Burglary or theft of items from your home
**Liability Claims:** If someone gets injured on your property

📞 **Claim Process Steps:**
1. **Immediate Action:** Ensure safety, prevent further damage
2. **Contact Insurer:** Report claim within 24-48 hours
3. **Document Everything:** Take photos, make inventory of damages
4. **File Police Report:** If required (theft, vandalism cases)
5. **Surveyor Visit:** Insurance company will send assessor
6. **Submit Documents:** Provide all required paperwork
7. **Claim Settlement:** Processing usually takes 15-45 days

💡 **Important Tips:**
- Don't throw away damaged items until surveyor inspection
- Keep all receipts and warranties of valuable items
- Maintain home inventory with photos and values
- Review your policy coverage limits annually""",
        
        "hi": """आपके गृह बीमा दावे के लिए, यहाँ आपको जानने योग्य सब कुछ है:

📋 **आवश्यक दस्तावेज:**
• **दावा फॉर्म** - सही तरीके से भरा गया बीमा दावा फॉर्म
• **पॉलिसी दस्तावेज** - आपकी वर्तमान गृह बीमा पॉलिसी
• **क्षति की तस्वीरें** - सभी क्षतिग्रस्त वस्तुओं/क्षेत्रों की स्पष्ट तस्वीरें
• **खरीदारी के बिल** - क्षतिग्रस्त वस्तुओं के मूल बिल (यदि उपलब्ध हो)"""
    }
}

# Enhanced LLM Query with better error handling and debugging
def query_llm(prompt, provider, lang="en", claim_type=None, context=None):
    system_messages = {
        "Vehicle": {
            "en": "You are a helpful vehicle insurance expert. Provide detailed, accurate information about insurance claims, required documents, and step-by-step guidance. Be empathetic and understanding as people dealing with vehicle accidents are often stressed. Always provide practical, actionable advice.",
            "hi": "आप एक सहायक वाहन बीमा विशेषज्ञ हैं। बीमा दावों, आवश्यक दस्तावेजों और चरणबद्ध मार्गदर्शन के बारे में विस्तृत, सटीक जानकारी प्रदान करें। सहानुभूतिपूर्ण और समझदार बनें क्योंकि वाहन दुर्घटना से निपटने वाले लोग अक्सर तनावग्रस्त होते हैं।"
        },
        "Health": {
            "en": "You are a compassionate health insurance specialist. Help people navigate medical insurance claims with clear explanations. Be sensitive to their health concerns and provide step-by-step guidance for claim procedures and required documents.",
            "hi": "आप एक दयालु स्वास्थ्य बीमा विशेषज्ञ हैं। लोगों को स्पष्ट व्याख्या के साथ चिकित्सा बीमा दावों में मार्गदर्शन करने में मदद करें। उनकी स्वास्थ्य चिंताओं के प्रति संवेदनशील रहें।"
        },
        "Home": {
            "en": "You are a knowledgeable home insurance consultant. Help people understand property insurance claims, required documentation, and the claims process. Be supportive as property damage can be very stressful for families.",
            "hi": "आप एक जानकार गृह बीमा सलाहकार हैं। लोगों को संपत्ति बीमा दावों, आवश्यक दस्तावेजों और दावा प्रक्रिया को समझने में मदद करें।"
        }
    }

    system_msg = system_messages.get(claim_type, system_messages["Vehicle"])[lang]
    full_prompt = f"{system_msg}\n\nContext: {context}\nQuestion: {prompt}\nPlease provide a helpful, detailed response:"

    print(f"Attempting to query {provider} with prompt length: {len(full_prompt)}")
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1} with {provider}")
            
            if provider == "ollama":
                # Check if Ollama is running
                try:
                    health_response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
                    if not health_response.ok:
                        print("Ollama server not responding to health check")
                        continue
                except:
                    print("Cannot connect to Ollama server")
                    continue
                    
                payload = {
                    "model": "phi3",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                }
                
                print(f"Sending request to: {LLM_PROVIDERS['ollama']['url']}")
                response = requests.post(
                    LLM_PROVIDERS["ollama"]["url"], 
                    headers=LLM_PROVIDERS["ollama"]["headers"],
                    json=payload, 
                    timeout=REQUEST_TIMEOUT
                )
                
                print(f"Ollama response status: {response.status_code}")
                if response.ok:
                    result = response.json()
                    print(f"Ollama response keys: {result.keys()}")
                    return result.get("response", "").strip()
                else:
                    print(f"Ollama error: {response.text}")

            elif provider == "groq" and GROQ_API_KEY:
                payload = {
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
                
                print("Sending request to Groq...")
                response = requests.post(
                    LLM_PROVIDERS["groq"]["url"],
                    headers=LLM_PROVIDERS["groq"]["headers"],
                    json=payload, 
                    timeout=REQUEST_TIMEOUT
                )
                
                print(f"Groq response status: {response.status_code}")
                if response.ok:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print(f"Groq error: {response.text}")

            elif provider == "gemini" and GEMINI_API_KEY:
                payload = {
                    "contents": [{
                        "parts": [{"text": full_prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 500
                    }
                }
                
                print("Sending request to Gemini...")
                response = requests.post(
                    LLM_PROVIDERS["gemini"]["url"],
                    headers=LLM_PROVIDERS["gemini"]["headers"],
                    json=payload, 
                    timeout=REQUEST_TIMEOUT
                )
                
                print(f"Gemini response status: {response.status_code}")
                if response.ok:
                    result = response.json()
                    return result["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    print(f"Gemini error: {response.text}")

            elif provider == "huggingface" and HF_TOKEN:
                payload = {
                    "inputs": full_prompt,
                    "parameters": {
                        "return_full_text": False,
                        "max_new_tokens": 500,
                        "temperature": 0.7,
                        "do_sample": True
                    },
                    "options": {"wait_for_model": True}
                }
                
                print("Sending request to HuggingFace...")
                response = requests.post(
                    LLM_PROVIDERS["huggingface"]["url"],
                    headers=LLM_PROVIDERS["huggingface"]["headers"],
                    json=payload, 
                    timeout=REQUEST_TIMEOUT
                )
                
                print(f"HuggingFace response status: {response.status_code}")
                if response.ok:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]["generated_text"].strip()
                else:
                    print(f"HuggingFace error: {response.text}")

        except requests.exceptions.Timeout:
            print(f"Timeout error on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}")
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
        
        if attempt < MAX_RETRIES - 1:
            print(f"Retrying in 2 seconds...")
            time.sleep(2)

    # Return comprehensive default response if all LLMs fail
    print(f"All LLM attempts failed, returning default response for {claim_type}")
    default_key = claim_type if claim_type in DEFAULT_RESPONSES else "Vehicle"
    return translator.translate(DEFAULT_RESPONSES[default_key][lang], lang)

# UI Configuration
def setup_ui():
    st.set_page_config(
        page_title="InsuranceSaathi",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for black theme
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e !important;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        background: linear-gradient(90deg, #121212 0%, #333333 100%);
        color: white;
        font-weight: bold;
        border: 1px solid #444;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background: #333333;
        color: white;
    }
    .ai-message {
        background: #424242;
        color: white;
    }
    .stMetric {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
    }
    .stMetric label {
        color: #aaaaaa !important;
    }
    .stMetric div {
        color: white !important;
    }
    .stAlert {
        background-color: #2d2d2d !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    setup_ui()
    
    # Initialize session state
    if "chat" not in st.session_state:
        st.session_state.chat = []
    
    # Debug panel
    with st.expander("🔧 Debug Information"):
        st.write("**API Keys Status:**")
        st.write(f"- HuggingFace: {'✅ Set' if HF_TOKEN else '❌ Missing'}")
        st.write(f"- Gemini: {'✅ Set' if GEMINI_API_KEY else '❌ Missing'}")
        st.write(f"- Groq: {'✅ Set' if GROQ_API_KEY else '❌ Missing'}")
        st.write(f"- Ollama URL: {OLLAMA_API_URL}")
        
        # Test Ollama connection
        if st.button("Test Ollama Connection"):
            try:
                response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
                if response.ok:
                    models = response.json().get('models', [])
                    st.success(f"✅ Ollama connected! Available models: {[m['name'] for m in models]}")
                else:
                    st.error(f"❌ Ollama connection failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Cannot connect to Ollama: {str(e)}")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        lang = st.selectbox("Language", ["en", "hi"], 
                          format_func=lambda x: "English" if x == "en" else "हिंदी")
        provider = st.selectbox("AI Provider", ["ollama", "groq", "gemini", "huggingface"])
        claim_type = st.selectbox("Claim Type", ["Vehicle", "Health", "Home"])
        
        st.markdown("---")
        st.subheader("📋 Scenario Details")
        scenario = st.text_area(translator.translate("Describe your situation", lang),
                              value=translator.translate("I had an accident and need help with my insurance claim...", lang),
                              height=100)
        
        st.markdown("---")
        st.subheader("📁 Upload Files")
        pdf_file = st.file_uploader(translator.translate("Health Documents (PDF)", lang), 
                                  type=["pdf"])
        image_file = st.file_uploader(translator.translate("Damage Photos", lang), 
                                    type=["jpg", "png", "jpeg"])
    
    # Main Content
    st.title(f"🛡️ {translator.translate('InsuranceSaathi - Your Claim Assistant', lang)}")
    st.markdown(translator.translate("Get instant help with your insurance claims - we're here to guide you through every step!", lang))
    
    # Show quick help based on claim type
    with st.expander(f"📖 Quick Guide for {claim_type} Claims"):
        default_response = DEFAULT_RESPONSES.get(claim_type, DEFAULT_RESPONSES["Vehicle"])
        st.markdown(translator.translate(default_response[lang], lang))
    
    # Image Processing Section
    if image_file:
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(translator.translate("Original Image", lang))
                original_img = Image.open(image_file)
                st.image(original_img, use_container_width=True)
            
            with col2:
                st.subheader(translator.translate("Enhanced Image", lang))
                enhanced_img = enhance_image(original_img)
                st.image(enhanced_img, use_container_width=True)
            
            analysis = analyze_image(enhanced_img)
            
            st.subheader(translator.translate("Damage Analysis", lang))
            cols = st.columns(4)
            with cols[0]:
                st.metric(translator.translate("Severity", lang), 
                         translator.translate(analysis['damage_level'].capitalize(), lang))
            with cols[1]:
                st.metric(translator.translate("Quality Score", lang), analysis['quality_score'])
            with cols[2]:
                st.metric(translator.translate("Edge Density", lang), analysis['edge_density'])
            
            cost_range = estimate_repair_cost(scenario, analysis, claim_type)
            st.success(f"💵 {translator.translate('Estimated Repair Cost', lang)}: ₹{cost_range[0]:,} - ₹{cost_range[1]:,}")
            
        except Exception as e:
            st.error(translator.translate(f"Error processing image: {e}", lang))
    
    # PDF Processing Section
    if pdf_file and st.button(translator.translate("Analyze Health Documents", lang)):
        with st.spinner(translator.translate("Processing documents...", lang)):
            try:
                agent = HealthAgent(lang)
                report = agent.generate_report(pdf_file.read(), scenario)
                st.subheader(translator.translate("Document Analysis Report", lang))
                st.text_area(label="", value=report, height=200)
            except Exception as e:
                st.error(translator.translate(f"Error processing PDF: {e}", lang))
    
    # Chat Interface
    st.markdown("---")
    st.subheader(f"💬 {translator.translate('Chat with InsuranceSaathi', lang)}")
    
    user_input = st.text_input(translator.translate("Ask your insurance question...", lang), key="user_input")
    
    if st.button(translator.translate("Submit", lang)) and user_input:
        # More flexible keyword checking
        insurance_keywords = ["claim", "insurance", "document", "required", "policy", "coverage", "premium", "accident", "damage", "repair", "hospital", "medical", "bill", "reimbursement", "cashless", "settlement"]
        
        if not any(kw in user_input.lower() for kw in insurance_keywords):
            st.warning(translator.translate("Please ask insurance-related questions. I'm here to help with your insurance claims!", lang))
        else:
            with st.spinner(translator.translate("Generating response...", lang)):
                context = f"{claim_type} claim scenario: {scenario}"
                if image_file:
                    context += f". Image analysis shows {analysis['damage_level']} damage level."
                
                response = query_llm(
                    prompt=user_input,
                    provider=provider,
                    lang=lang,
                    claim_type=claim_type,
                    context=context
                )
                
                st.session_state.chat.append((user_input, response))
                st.rerun()
    
    # Display Chat History
    if st.session_state.chat:
        st.subheader(translator.translate("Conversation History", lang))
        for i, (user_msg, ai_msg) in enumerate(st.session_state.chat):
            message(user_msg, is_user=True, key=f"user_{i}")
            message(ai_msg, key=f"ai_{i}")
    
    # Quick Action Buttons
    st.markdown("---")
    st.subheader(translator.translate("Quick Actions", lang))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(translator.translate("📋 Required Documents", lang)):
            context = f"User needs to know required documents for {claim_type} claim"
            response = query_llm(
                prompt=f"What documents are required for {claim_type} insurance claim?",
                provider=provider,
                lang=lang,
                claim_type=claim_type,
                context=context
            )
            st.session_state.chat.append((f"What documents do I need for {claim_type} claim?", response))
            st.rerun()
    
    with col2:
        if st.button(translator.translate("⏰ Claim Timeline", lang)):
            context = f"User wants to know the timeline for {claim_type} claim processing"
            response = query_llm(
                prompt=f"What is the typical timeline for {claim_type} insurance claim processing?",
                provider=provider,
                lang=lang,
                claim_type=claim_type,
                context=context
            )
            st.session_state.chat.append((f"How long does {claim_type} claim take?", response))
            st.rerun()
    
    with col3:
        if st.button(translator.translate("📞 Next Steps", lang)):
            context = f"User wants to know next steps for {claim_type} claim with scenario: {scenario}"
            response = query_llm(
                prompt=f"What should I do next for my {claim_type} insurance claim?",
                provider=provider,
                lang=lang,
                claim_type=claim_type,
                context=context
            )
            st.session_state.chat.append(("What are my next steps?", response))
            st.rerun()
    
    # Clear chat button
    if st.session_state.chat and st.button(translator.translate("🗑️ Clear Chat History", lang)):
        st.session_state.chat = []
        st.rerun()

if __name__ == "__main__":
    main()
