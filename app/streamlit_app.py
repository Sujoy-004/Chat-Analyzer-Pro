import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import json
import re
import requests
import warnings
import sys
import os
from typing import List, Dict, Any, Tuple
import zipfile
import uuid
import logging
from PIL import Image, ImageFile

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Chat Analyzer Pro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for modules
if 'modules_loaded' not in st.session_state:
    st.session_state.modules_loaded = False
    st.session_state.ingestion_available = False
    st.session_state.whatsapp_parser_available = False
    st.session_state.telegram_parser_available = False
    st.session_state.relationship_health_available = False
    st.session_state.executed_modules = None

@st.cache_data
def load_github_modules():
    """Load analysis modules from GitHub"""
    modules = {}
    module_urls = {
        "ingestion": "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/src/ingest/ingestion.py",
        "whatsapp_parser": "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/src/parser/whatsapp_parser.py",
        "telegram_parser": "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/src/parser/telegram_parser.py",
        "relationship_health": "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/src/analysis/relationship_health.py"
    }
    
    success_count = 0
    for name, url in module_urls.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                modules[name] = response.text
                success_count += 1
        except Exception as e:
            st.warning(f"Could not load {name} module: {str(e)[:100]}...")
            modules[name] = None
    
    st.session_state.modules_loaded = True
    st.session_state.ingestion_available = modules.get("ingestion") is not None
    st.session_state.whatsapp_parser_available = modules.get("whatsapp_parser") is not None
    st.session_state.telegram_parser_available = modules.get("telegram_parser") is not None
    st.session_state.relationship_health_available = modules.get("relationship_health") is not None
    
    return modules, success_count

# Load modules and handle execution
def load_and_execute_modules():
    """Load and execute modules, handling caching properly"""
    if 'executed_modules' not in st.session_state or st.session_state.executed_modules is None:
        with st.spinner('🔄 Loading analysis modules...'):
            modules, success_count = load_github_modules()
        
        # Execute modules
        executed_modules = {}
        for name, code in modules.items():
            if code:
                try:
                    namespace = {}
                    exec(code, namespace)
                    # Store only serializable references
                    executed_modules[name] = {
                        'loaded': True,
                        'functions': list(namespace.keys())
                    }
                    # Store the actual namespace in session state (not cached)
                    st.session_state[f'module_{name}'] = namespace
                except Exception as e:
                    st.error(f"Error executing {name}: {str(e)[:100]}...")
                    executed_modules[name] = {'loaded': False, 'error': str(e)}
                    st.session_state[f'module_{name}'] = None
            else:
                executed_modules[name] = {'loaded': False, 'error': 'Failed to download'}
                st.session_state[f'module_{name}'] = None
        
        # Update session state flags based on successful module loading
        st.session_state.ingestion_available = executed_modules.get('ingestion', {}).get('loaded', False)
        st.session_state.whatsapp_parser_available = executed_modules.get('whatsapp_parser', {}).get('loaded', False)
        st.session_state.telegram_parser_available = executed_modules.get('telegram_parser', {}).get('loaded', False)
        st.session_state.relationship_health_available = executed_modules.get('relationship_health', {}).get('loaded', False)
        
        st.session_state.executed_modules = executed_modules
        st.session_state.modules_success_count = success_count
        return executed_modules, success_count
    else:
        return st.session_state.executed_modules, st.session_state.get('modules_success_count', 0)

def get_module_namespace(module_name):
    """Get the actual module namespace from session state"""
    return st.session_state.get(f'module_{module_name}', None)

# Load custom CSS
def load_css():
    """Load custom CSS"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .health-score {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .excellent { color: #28a745; }
    .good { color: #17a2b8; }
    .fair { color: #ffc107; }
    .poor { color: #dc3545; }
    .file-info {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .media-info {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .module-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .module-success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
    .module-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
    .module-error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    </style>
    """, unsafe_allow_html=True)

# Fallback parsing functions
def fallback_whatsapp_parser(content):
    """Fallback WhatsApp parser when main module is unavailable"""
    messages = []
    lines = content.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        pattern = r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*-?\s*([^:]+):\s*(.+)'
        match = re.match(pattern, line)
        
        if match:
            date_str, time_str, sender, message = match.groups()
            
            try:
                if len(date_str.split('/')[2]) == 2:
                    date_obj = datetime.strptime(date_str, '%d/%m/%y')
                else:
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                
                try:
                    time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
                except:
                    time_obj = datetime.strptime(time_str, '%H:%M').time()
                
                full_datetime = datetime.combine(date_obj.date(), time_obj)
                
                messages.append({
                    'datetime': full_datetime,
                    'sender': sender.strip(),
                    'message': message.strip(),
                    'date': full_datetime.date().strftime('%Y-%m-%d'),
                    'time': full_datetime.time().strftime('%H:%M'),
                    'hour': full_datetime.hour,
                    'message_length': len(message.strip()),
                    'source': 'whatsapp_fallback'
                })
            except Exception:
                continue
    
    return messages

def fallback_telegram_parser(json_data):
    """Fallback Telegram parser when main module is unavailable"""
    messages = []
    
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        chat_messages = data.get('messages', [])
        
        for msg in chat_messages:
            if msg.get('type') == 'message' and 'text' in msg:
                try:
                    datetime_obj = datetime.fromisoformat(msg['date'].replace('Z', '+00:00'))
                    sender = msg.get('from', 'Unknown')
                    text_content = msg['text']
                    
                    if isinstance(text_content, list):
                        text_content = ''.join([item if isinstance(item, str) else '' for item in text_content])
                    
                    messages.append({
                        'datetime': datetime_obj,
                        'sender': sender,
                        'message': text_content,
                        'date': datetime_obj.date().strftime('%Y-%m-%d'),
                        'time': datetime_obj.time().strftime('%H:%M'),
                        'hour': datetime_obj.hour,
                        'message_length': len(text_content),
                        'source': 'telegram_fallback'
                    })
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Error parsing Telegram data: {e}")
    
    return messages

# File processing with both advanced and fallback methods
def process_uploaded_file(uploaded_file, executed_modules):
    """Process uploaded file using available modules or fallbacks"""
    filename = uploaded_file.name
    file_type = filename.lower().split('.')[-1]
    
    # Try advanced ingestion first
    ingestion_module = get_module_namespace("ingestion")
    if ingestion_module and st.session_state.ingestion_available:
        try:
            # Use the advanced ingestion module
            if "process_uploaded_file" in ingestion_module:
                messages, media_ocr = ingestion_module["process_uploaded_file"](uploaded_file)
                return convert_normalized_messages_to_df(messages), media_ocr, "advanced"
        except Exception as e:
            st.warning(f"Advanced processing failed, falling back to basic mode: {str(e)[:100]}...")
    
    # Fallback processing
    content = uploaded_file.read()
    messages = []
    media_ocr = []
    
    if file_type == 'txt':
        try:
            text = content.decode('utf-8')
            messages = fallback_whatsapp_parser(text)
        except Exception as e:
            st.error(f"Error processing TXT file: {e}")
    
    elif file_type == 'json':
        try:
            text = content.decode('utf-8')
            data = json.loads(text)
            messages = fallback_telegram_parser(data)
        except Exception as e:
            st.error(f"Error processing JSON file: {e}")
    
    elif file_type in ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tiff']:
        media_ocr.append({
            "file": filename,
            "note": f"Image file detected. OCR text extraction requires advanced mode."
        })
    
    elif file_type == 'pdf':
        media_ocr.append({
            "file": filename,
            "note": f"PDF file detected. Text extraction requires advanced mode."
        })
    
    elif file_type == 'zip':
        media_ocr.append({
            "file": filename,
            "note": f"ZIP archive detected. Archive extraction requires advanced mode."
        })
    
    elif file_type in ['opus', 'mp4', 'avi', 'mov', 'm4a', 'wav', 'mp3']:
        media_ocr.append({
            "file": filename,
            "note": f"Media file detected. Metadata extraction requires advanced mode."
        })
    
    else:
        media_ocr.append({
            "file": filename,
            "note": f"File type '{file_type}' not supported in fallback mode. Advanced mode required."
        })
    
    df = pd.DataFrame(messages) if messages else pd.DataFrame()
    return df, media_ocr, "fallback"

def convert_normalized_messages_to_df(messages):
    """Convert normalized messages from ingestion module to DataFrame"""
    if not messages:
        return pd.DataFrame()
    
    df_data = []
    for msg in messages:
        try:
            # Parse datetime
            if msg.get('date') and msg.get('time'):
                try:
                    datetime_str = f"{msg['date']} {msg['time']}"
                    full_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
                except:
                    full_datetime = datetime.now()
            else:
                full_datetime = datetime.now()
            
            df_data.append({
                'datetime': full_datetime,
                'sender': msg.get('author', 'Unknown'),
                'message': msg.get('text', ''),
                'date': msg.get('date', full_datetime.strftime('%Y-%m-%d')),
                'time': msg.get('time', full_datetime.strftime('%H:%M')),
                'hour': full_datetime.hour,
                'message_length': len(msg.get('text', '')),
                'source': msg.get('source', 'unknown'),
                'uid': msg.get('uid', str(uuid.uuid4()))
            })
        except Exception:
            continue
    
    return pd.DataFrame(df_data)

# Enhanced relationship health calculation
def calculate_relationship_health(df, executed_modules):
    """Calculate relationship health using available modules or fallback"""
    if df.empty:
        return None
    
    # Try advanced relationship health analysis first
    rh_module = get_module_namespace("relationship_health")
    if rh_module and st.session_state.relationship_health_available:
        try:
            if "analyze_relationship_health" in rh_module:
                # Prepare data for the advanced module
                df_prepared = df.copy()
                if 'datetime' not in df_prepared.columns:
                    df_prepared['datetime'] = pd.to_datetime(df_prepared['date'] + ' ' + df_prepared['time'])
                
                results = rh_module["analyze_relationship_health"](df_prepared)
                return {
                    'method': 'advanced',
                    'results': results,
                    'total_score': results['health_score']['overall_health_score'] * 100,
                    'grade': results['health_score']['grade'],
                    'component_scores': results['health_score']['component_scores'],
                    'strengths': results['health_score']['strengths'],
                    'improvements': results['health_score']['areas_for_improvement']
                }
        except Exception as e:
            st.warning(f"Advanced health analysis failed, using fallback: {str(e)[:100]}...")
    
    # Fallback health calculation
    return calculate_basic_health_score(df)

def calculate_basic_health_score(df):
    """Basic health score calculation as fallback"""
    if len(df) < 2:
        return None
    
    # Calculate conversation initiators
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    conversation_gap = timedelta(minutes=30)
    conversation_starters = []
    previous_time = None

    for idx, row in df_sorted.iterrows():
        current_time = row['datetime']
        if previous_time is None or (current_time - previous_time) > conversation_gap:
            conversation_starters.append(row['sender'])
        previous_time = current_time

    initiators_df = pd.DataFrame({'initiator': conversation_starters})
    
    # Calculate response times
    response_data = []
    for idx in range(1, len(df_sorted)):
        current_msg = df_sorted.iloc[idx]
        previous_msg = df_sorted.iloc[idx-1]
        if current_msg['sender'] != previous_msg['sender']:
            response_time = current_msg['datetime'] - previous_msg['datetime']
            response_minutes = response_time.total_seconds() / 60
            response_data.append({
                'responder': current_msg['sender'],
                'response_time_minutes': response_minutes
            })
    
    response_df = pd.DataFrame(response_data)
    message_counts = df['sender'].value_counts()
    
    # Calculate health score components
    # 1. Communication Balance (25 points)
    participants = list(message_counts.index)
    if len(participants) >= 2:
        main_participant_pct = (message_counts.iloc[0] / len(df)) * 100
        balance_score = 100 - abs(main_participant_pct - 50)
        balance_points = (balance_score / 100) * 25
    else:
        balance_points = 12.5

    # 2. Initiation Balance (20 points)
    if len(initiators_df) > 0:
        init_counts = initiators_df['initiator'].value_counts()
        if len(init_counts) >= 2:
            main_init_pct = (init_counts.iloc[0] / len(initiators_df)) * 100
            init_balance = 100 - abs(main_init_pct - 50)
            init_points = (init_balance / 100) * 20
        else:
            init_points = 10
    else:
        init_points = 10

    # 3. Response Quality (25 points)
    if len(response_df) > 0:
        avg_response_time = response_df['response_time_minutes'].mean()
        if avg_response_time <= 30:
            response_points = 25
        elif avg_response_time <= 120:
            response_points = 25 - ((avg_response_time - 30) / 90) * 15
        else:
            response_points = max(5, 10 - ((avg_response_time - 120) / 60) * 2)
    else:
        response_points = 15

    # 4. Conversation Consistency (15 points)
    date_range = (df['datetime'].max() - df['datetime'].min()).days + 1
    conversations_per_day = len(initiators_df) / date_range if date_range > 0 else 0
    if conversations_per_day >= 2:
        consistency_points = 15
    elif conversations_per_day >= 1:
        consistency_points = conversations_per_day * 7.5
    else:
        consistency_points = conversations_per_day * 15

    # 5. Engagement Quality (15 points)
    avg_msg_length = df['message_length'].mean()
    if avg_msg_length >= 40:
        engagement_points = 15
    elif avg_msg_length >= 20:
        engagement_points = (avg_msg_length - 20) / 20 * 15
    else:
        engagement_points = avg_msg_length / 20 * 7.5

    total_score = balance_points + init_points + response_points + consistency_points + engagement_points
    
    # Determine grade
    if total_score >= 85:
        grade = "EXCELLENT"
    elif total_score >= 70:
        grade = "GOOD"
    elif total_score >= 55:
        grade = "FAIR"
    else:
        grade = "NEEDS IMPROVEMENT"
    
    return {
        'method': 'basic',
        'total_score': total_score,
        'grade': grade,
        'balance_points': balance_points,
        'init_points': init_points,
        'response_points': response_points,
        'consistency_points': consistency_points,
        'engagement_points': engagement_points,
        'message_counts': message_counts,
        'response_df': response_df,
        'initiators_df': initiators_df
    }

def display_media_results(media_ocr):
    """Display media OCR results"""
    if not media_ocr:
        return
    
    st.subheader("📸 Media & File Analysis")
    
    for item in media_ocr:
        filename = item.get('file', 'Unknown file')
        ocr_text = item.get('ocr', '')
        note = item.get('note', '')
        
        st.markdown(f'<div class="media-info"><strong>📁 {filename}</strong></div>', unsafe_allow_html=True)
        
        if ocr_text:
            with st.expander(f"View extracted text from {filename}"):
                st.text_area("Extracted Text", ocr_text, height=150, key=f"ocr_{filename}")
        
        if note:
            st.info(f"ℹ️ {note}")

def show_module_status(executed_modules, success_count):
    """Show module loading status"""
    st.sidebar.markdown("### 🔧 Module Status")
    
    total_modules = len(executed_modules)
    successful_modules = sum(1 for m in executed_modules.values() if m.get('loaded', False))
    
    if successful_modules == total_modules:
        st.sidebar.markdown('<div class="module-status module-success">✅ All modules loaded successfully</div>', unsafe_allow_html=True)
        st.sidebar.success(f"Advanced mode enabled - all {successful_modules}/{total_modules} modules active")
    elif successful_modules > 0:
        st.sidebar.markdown(f'<div class="module-status module-warning">⚠️ Partial functionality - {successful_modules}/{total_modules} modules loaded</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="module-status module-error">❌ Basic mode only - no advanced modules loaded</div>', unsafe_allow_html=True)
    
    # Show individual module status
    with st.sidebar.expander("Module Details"):
        module_names = {
            "ingestion": "🔄 File Ingestion",
            "whatsapp_parser": "💬 WhatsApp Parser", 
            "telegram_parser": "📱 Telegram Parser",
            "relationship_health": "❤️ Health Analysis"
        }
        
        for key, name in module_names.items():
            module_info = executed_modules.get(key, {})
            if module_info.get('loaded', False):
                st.write(f"✅ {name}")
            else:
                error_msg = module_info.get('error', 'Unknown error')
                st.write(f"❌ {name}")
                if error_msg != 'Failed to download':
                    st.caption(f"Error: {error_msg[:50]}...")


# Main Streamlit App
def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">💬 Chat Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced chat analysis with multi-format support, OCR, and AI-powered insights")
    
    # Load modules
    executed_modules, success_count = load_and_execute_modules()
    
    # Show module status with debug info
    show_module_status(executed_modules, success_count)
    
    # Debug info in sidebar
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("**Debug Information:**")
        st.sidebar.write(f"ingestion_available: {st.session_state.ingestion_available}")
        st.sidebar.write(f"Module namespaces available:")
        for module in ['ingestion', 'whatsapp_parser', 'telegram_parser', 'relationship_health']:
            namespace = get_module_namespace(module)
            if namespace:
                functions = [k for k in namespace.keys() if not k.startswith('_')][:5]  # Show first 5 functions
                st.sidebar.write(f"- {module}: {functions}")
            else:
                st.sidebar.write(f"- {module}: None")
    
    # Sidebar
    st.sidebar.title("📁 Upload Your Files")
    
    # Dynamic file upload based on available modules
    if st.session_state.ingestion_available:
        accepted_types = ['txt', 'json', 'zip', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'pdf', 'opus', 'mp4', 'avi', 'mov', 'gif', 'tiff', 'tif', 'm4a', 'wav', 'mp3', 'aac', 'ogg', 'flac', 'mkv', 'webm', 'flv', '3gp']
        help_text = "All file types supported: Chat exports, images (OCR), PDFs, ZIP archives, media files"
    else:
        accepted_types = ['txt', 'json', 'zip']
        help_text = "Basic mode: TXT (WhatsApp), JSON (Telegram), and ZIP files"
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose files to analyze",
        type=accepted_types,
        help=help_text
    )
    
    # Display capabilities
    if st.session_state.ingestion_available:
        st.sidebar.markdown("""
        **✨ Advanced Features Available:**
        - 📦 **ZIP archives**: Extract and process multiple files
        - 🖼️ **Images**: PNG, JPG, JPEG, WebP, BMP, GIF, TIFF (OCR)
        - 📄 **PDFs**: Text extraction + OCR
        - 🎵 **Media files**: OPUS, MP4, AVI, MOV (metadata extraction)
        - 🔄 **Multi-format**: Automatic format detection
        """)
    else:
        st.sidebar.markdown("""
        **📝 Basic Mode:**
        - WhatsApp TXT exports
        - Telegram JSON exports
        """)
    
    if uploaded_file is not None:
        try:
            with st.spinner('🔍 Processing your file...'):
                df, media_ocr, processing_method = process_uploaded_file(uploaded_file, executed_modules)
            
            if df.empty and not media_ocr:
                st.error("❌ No messages could be extracted from the uploaded file. Please check the format.")
                return
            
            # Show processing summary
            st.sidebar.success(f"✅ File processed ({processing_method} mode)")
            
            if not df.empty:
                sources = df['source'].value_counts() if 'source' in df.columns else pd.Series(['unknown'])
                unique_senders = df['sender'].nunique()
                date_range_days = (df['datetime'].max() - df['datetime'].min()).days + 1
                
                st.sidebar.markdown(f"""
                <div class="file-info">
                <strong>📊 Processing Results:</strong><br>
                • {len(df)} messages extracted<br>
                • {unique_senders} participants<br>
                • {date_range_days} days of chat<br>
                • {len(media_ocr)} media items processed<br>
                • Sources: {', '.join(sources.index.tolist())}
                </div>
                """, unsafe_allow_html=True)
                
                # Display media results
                if media_ocr:
                    display_media_results(media_ocr)
                
                # Main analysis
                st.markdown(f"""
                ### 📊 Analysis Summary
                **File**: {uploaded_file.name}  
                **Messages**: {len(df)}  
                **Participants**: {unique_senders}  
                **Date Range**: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}  
                **Processing Method**: {processing_method.title()}
                """)
                
                # Relationship Health Analysis
                with st.spinner('🔍 Analyzing relationship health...'):
                    health_results = calculate_relationship_health(df, executed_modules)
                
                if health_results:
                    # Display health score
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        score = health_results['total_score']
                        grade = health_results['grade']
                        
                        # Modern gradient colors for health score
                        if score >= 85:
                            color = "excellent"
                            bg_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                        elif score >= 70:
                            color = "good"  
                            bg_color = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                        elif score >= 55:
                            color = "fair"
                            bg_color = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
                        else:
                            color = "poor"
                            bg_color = "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"
                        
                        st.markdown(f'''
                        <div style="
                            background: {bg_color};
                            border-radius: 20px;
                            padding: 2rem;
                            text-align: center;
                            color: white;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                            margin: 1rem 0;
                        ">
                            <div style="font-size: 4rem; font-weight: bold; margin-bottom: 0.5rem;">
                                {score:.1f}/100
                            </div>
                            <div style="font-size: 1.2rem; opacity: 0.9;">
                                Relationship Health: {grade}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Filter out OCR messages for accurate participant count
                    chat_only_df = df[~df['source'].str.contains('ocr|pdf', case=False, na=False)]
                    
                    with col1:
                        st.metric("📊 Messages", len(df))
                    
                    with col2:
                        actual_participants = chat_only_df['sender'].nunique() if not chat_only_df.empty else df['sender'].nunique()
                        st.metric("👥 Participants", actual_participants)
                    
                    with col3:
                        st.metric("📅 Days", date_range_days)
                    
                    with col4:
                        avg_response = health_results.get('response_df', pd.DataFrame())
                        if not avg_response.empty and 'response_time_minutes' in avg_response.columns:
                            avg_time = avg_response['response_time_minutes'].mean()
                            st.metric("⏱️ Avg Response", f"{avg_time:.1f} min")
                        else:
                            st.metric("⏱️ Avg Response", "N/A")
                    
                    # Charts section
                    st.subheader("📈 Analysis Dashboard")
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Message distribution with beautiful colors - exclude OCR/unknown
                        chat_message_counts = chat_only_df[chat_only_df['sender'] != 'unknown']['sender'].value_counts() if not chat_only_df.empty else df['sender'].value_counts()
                        
                        # Modern color palette
                        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#43e97b']
                        
                        fig_pie = px.pie(
                            values=chat_message_counts.values,
                            names=chat_message_counts.index,
                            title="💬 Message Distribution",
                            color_discrete_sequence=colors
                        )
                        fig_pie.update_layout(
                            font=dict(size=14),
                            title_font_size=16,
                            showlegend=True,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        fig_pie.update_traces(
                            textposition='inside', 
                            textinfo='percent+label',
                            hovertemplate='<b>%{label}</b><br>Messages: %{value}<br>Percentage: %{percent}<extra></extra>',
                            marker=dict(line=dict(color='#FFFFFF', width=2))
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with chart_col2:
                        # Health score breakdown with gradient colors
                        if health_results['method'] == 'advanced' and 'component_scores' in health_results:
                            comp_scores = health_results['component_scores']
                            categories = list(comp_scores.keys())
                            values = [comp_scores[cat] * 100 for cat in categories]
                        else:
                            categories = ['Balance', 'Initiation', 'Response', 'Consistency', 'Engagement']
                            values = [
                                health_results.get('balance_points', 0),
                                health_results.get('init_points', 0),
                                health_results.get('response_points', 0),
                                health_results.get('consistency_points', 0),
                                health_results.get('engagement_points', 0)
                            ]
                        
                        fig_bar = px.bar(
                            x=categories, 
                            y=values,
                            title='🎯 Health Score Breakdown',
                            color=values,
                            color_continuous_scale=['#ff6b6b', '#feca57', '#48dbfb', '#0abde3', '#00d2d3']
                        )
                        fig_bar.update_layout(
                            font=dict(size=12),
                            title_font_size=16,
                            xaxis_title="",
                            yaxis_title="Score",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            showlegend=False
                        )
                        fig_bar.update_traces(
                            hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>',
                            marker_line_color='rgba(255,255,255,0.6)',
                            marker_line_width=1.5
                        )
                        fig_bar.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Timeline analysis with modern styling
                    daily_activity = df.groupby('date').size().reset_index(name='messages')
                    fig_line = px.line(
                        daily_activity, 
                        x='date', 
                        y='messages', 
                        title='📅 Daily Message Activity',
                        markers=True,
                        line_shape='spline'
                    )
                    fig_line.update_layout(
                        font=dict(size=12),
                        title_font_size=16,
                        xaxis_title="Date",
                        yaxis_title="Messages",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode='x unified'
                    )
                    fig_line.update_traces(
                        line_color='#667eea',
                        line_width=3,
                        marker_color='#764ba2',
                        marker_size=8,
                        hovertemplate='<b>%{x}</b><br>Messages: %{y}<extra></extra>'
                    )
                    fig_line.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                    # Additional visualizations
                    st.subheader("📊 Advanced Analytics")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Hourly activity heatmap
                        if 'hour' in df.columns:
                            hourly_activity = df.groupby('hour').size().reset_index(name='messages')
                            fig_hourly = px.bar(
                                hourly_activity,
                                x='hour',
                                y='messages',
                                title='⏰ Messages by Hour of Day',
                                color='messages',
                                color_continuous_scale=['#667eea', '#764ba2', '#f093fb']
                            )
                            fig_hourly.update_layout(
                                font=dict(size=12),
                                title_font_size=16,
                                xaxis_title="Hour",
                                yaxis_title="Message Count",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=False
                            )
                            fig_hourly.update_traces(
                                hovertemplate='<b>%{x}:00</b><br>Messages: %{y}<extra></extra>'
                            )
                            st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    with viz_col2:
                        # Message length distribution
                        if 'message_length' in df.columns:
                            # Create bins for message lengths
                            bins = [0, 20, 50, 100, 200, 500, float('inf')]
                            labels = ['0-20', '21-50', '51-100', '101-200', '201-500', '500+']
                            df['length_category'] = pd.cut(df['message_length'], bins=bins, labels=labels)
                            
                            length_dist = df['length_category'].value_counts().sort_index()
                            fig_length = px.bar(
                                x=length_dist.index,
                                y=length_dist.values,
                                title='📏 Message Length Distribution',
                                color=length_dist.values,
                                color_continuous_scale=['#43e97b', '#38f9d7', '#4facfe']
                            )
                            fig_length.update_layout(
                                font=dict(size=12),
                                title_font_size=16,
                                xaxis_title="Characters",
                                yaxis_title="Message Count",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=False
                            )
                            fig_length.update_traces(
                                hovertemplate='<b>%{x} chars</b><br>Messages: %{y}<extra></extra>'
                            )
                            st.plotly_chart(fig_length, use_container_width=True)
                    
                    # Week day analysis
                    viz_col3, viz_col4 = st.columns(2)
                    
                    with viz_col3:
                        # Day of week activity
                        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.day_name()
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_counts = df['day_of_week'].value_counts().reindex(day_order, fill_value=0)
                        
                        fig_weekday = px.bar(
                            x=day_counts.index,
                            y=day_counts.values,
                            title='📆 Activity by Day of Week',
                            color=day_counts.values,
                            color_continuous_scale=['#f093fb', '#f5576c', '#ff6b6b']
                        )
                        fig_weekday.update_layout(
                            font=dict(size=12),
                            title_font_size=16,
                            xaxis_title="Day",
                            yaxis_title="Messages",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            showlegend=False
                        )
                        fig_weekday.update_xaxes(tickangle=45)
                        fig_weekday.update_traces(
                            hovertemplate='<b>%{x}</b><br>Messages: %{y}<extra></extra>'
                        )
                        st.plotly_chart(fig_weekday, use_container_width=True)
                    
                    with viz_col4:
                        # Message count per participant over time
                        if not chat_only_df.empty:
                            actual_senders = chat_only_df[chat_only_df['sender'] != 'unknown']['sender'].unique()
                            if len(actual_senders) > 0:
                                participant_timeline = chat_only_df[chat_only_df['sender'].isin(actual_senders)].groupby(['date', 'sender']).size().reset_index(name='messages')
                                
                                fig_timeline = px.line(
                                    participant_timeline,
                                    x='date',
                                    y='messages',
                                    color='sender',
                                    title='👥 Participant Activity Over Time',
                                    markers=True,
                                    color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#f5576c']
                                )
                                fig_timeline.update_layout(
                                    font=dict(size=12),
                                    title_font_size=16,
                                    xaxis_title="Date",
                                    yaxis_title="Messages",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    legend=dict(title="Participant")
                                )
                                fig_timeline.update_traces(
                                    line_width=2.5,
                                    marker_size=6
                                )
                                st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Advanced insights for advanced mode
                    if health_results['method'] == 'advanced':
                        st.subheader("🔍 Advanced Insights")
                        
                        insight_col1, insight_col2 = st.columns(2)
                        
                        with insight_col1:
                            st.write("**💪 Strengths:**")
                            for strength in health_results.get('strengths', []):
                                st.write(strength)
                        
                        with insight_col2:
                            st.write("**⚠️ Areas for Improvement:**")
                            for improvement in health_results.get('improvements', []):
                                st.write(improvement)
                    
                    else:
                        # Basic insights
                        st.subheader("🔍 Basic Insights")
                        
                        insight_col1, insight_col2 = st.columns(2)
                        
                        with insight_col1:
                            st.write("**💪 Strengths:**")
                            if health_results.get('balance_points', 0) >= 20:
                                st.write("✅ Well-balanced message distribution")
                            if health_results.get('init_points', 0) >= 16:
                                st.write("✅ Good conversation initiation balance")
                            if health_results.get('response_points', 0) >= 20:
                                st.write("✅ Responsive communication")
                            if health_results.get('consistency_points', 0) >= 12:
                                st.write("✅ Consistent communication pattern")
                        
                        with insight_col2:
                            st.write("**⚠️ Areas for Improvement:**")
                            if health_results.get('response_points', 0) < 20:
                                st.write("🔄 Could improve response times")
                            if health_results.get('balance_points', 0) < 20:
                                st.write("⚖️ Could balance message distribution")
                            if health_results.get('init_points', 0) < 16:
                                st.write("🚀 Could balance conversation initiation")
                    
                    # Source analysis if available
                    if 'source' in df.columns and len(df['source'].unique()) > 1:
                        st.subheader("📋 Source Analysis")
                        source_counts = df['source'].value_counts()
                        
                        # Custom colors for different source types
                        source_colors = {
                            'whatsapp_txt': '#25D366',  # WhatsApp green
                            'telegram_json': '#0088cc',  # Telegram blue
                            'json': '#ff6b6b',
                            'ocr_from_zip': '#feca57',  # Yellow for OCR
                            'ocr_image': '#ff9ff3',     # Pink for image OCR
                            'pdf': '#54a0ff',           # Blue for PDF
                            'unknown': '#95a5a6'        # Gray for unknown
                        }
                        
                        colors_list = [source_colors.get(source, '#667eea') for source in source_counts.index]
                        
                        fig_source = px.bar(
                            x=source_counts.index, 
                            y=source_counts.values, 
                            title="📊 Messages by Source Type",
                            color=source_counts.index,
                            color_discrete_map=source_colors
                        )
                        fig_source.update_layout(
                            font=dict(size=12),
                            title_font_size=16,
                            xaxis_title="Source Type",
                            yaxis_title="Message Count",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            showlegend=False
                        )
                        fig_source.update_traces(
                            hovertemplate='<b>%{x}</b><br>Messages: %{y}<extra></extra>',
                            marker_line_color='rgba(255,255,255,0.6)',
                            marker_line_width=1.5
                        )
                        fig_source.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_source, use_container_width=True)
                    
                    # Raw data preview
                    if st.checkbox("📋 Show Raw Data"):
                        st.subheader("Raw Message Data")
                        display_df = df.copy()
                        # Truncate long messages for display
                        if 'message' in display_df.columns:
                            display_df['message'] = display_df['message'].apply(
                                lambda x: x[:100] + "..." if len(str(x)) > 100 else x
                            )
                        st.dataframe(display_df.head(50), use_container_width=True)
                        
                        if len(df) > 50:
                            st.info(f"Showing first 50 rows of {len(df)} total messages")
                    
                    # Optional: Show media file analysis (collapsed by default)
                    if media_ocr and len(media_ocr) > 0:
                        with st.expander("🔍 View Detailed File Processing Log (Advanced)"):
                            st.caption("Technical details about processed files - primarily for debugging")
                            for item in media_ocr:
                                filename = item.get('file', 'Unknown file')
                                ocr_text = item.get('ocr', '')
                                note = item.get('note', '')
                                metadata = item.get('metadata', {})
                                
                                st.markdown(f"**📁 {filename}**")
                                
                                if note:
                                    st.caption(f"ℹ️ {note}")
                                
                                if metadata:
                                    st.json(metadata)
                                
                                if ocr_text and len(ocr_text.strip()) > 0:
                                    with st.expander(f"View extracted text"):
                                        st.text_area("Extracted Text", ocr_text[:500], height=100, disabled=True, key=f"ocr_{filename}")
                                        if len(ocr_text) > 500:
                                            st.caption(f"Showing first 500 of {len(ocr_text)} characters")
                                
                                st.markdown("---")
                
                else:
                    st.error("❌ Could not calculate relationship health metrics")
            
            else:
                st.info("ℹ️ No chat messages found, but processed media files:")
                display_media_results(media_ocr)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.exception(e)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="border: 2px dashed #1f77b4; border-radius: 10px; padding: 2rem; text-align: center; margin: 1rem 0;">
            <h3>👋 Welcome to Chat Analyzer Pro!</h3>
            <p>Upload your chat files to start analyzing communication patterns and relationship health.</p>
            <p>Supports multiple formats and provides AI-powered insights!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        st.subheader("🚀 Enhanced Features")
        
        # Dynamic feature display based on loaded modules
        if st.session_state.ingestion_available:
            feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
            
            with feature_col1:
                st.write("""
                **📊 Advanced Analytics**
                - Message statistics & patterns
                - Communication flow analysis  
                - Response time tracking
                - Engagement quality metrics
                """)
            
            with feature_col2:
                st.write("""
                **❤️ Relationship Health**
                - AI-powered health scoring
                - Balance & dominance analysis
                - Conversation quality assessment
                - Personalized recommendations
                """)
            
            with feature_col3:
                st.write("""
                **📈 Rich Visualizations**
                - Interactive charts & graphs
                - Timeline analysis
                - Distribution breakdowns
                - Health score dashboard
                """)
            
            with feature_col4:
                st.write("""
                **🔧 Multi-Format Support**
                - ZIP archive processing
                - Image OCR (PNG, JPG, WebP, etc.)
                - PDF text analysis
                - Media file metadata
                - Automatic format detection
                """)
            
            st.success("✨ **Pro Tip**: Upload a ZIP file containing multiple chat exports, screenshots, PDFs, or media files for comprehensive analysis!")
        
        else:
            feature_col1, feature_col2, feature_col3 = st.columns(3)
            
            with feature_col1:
                st.write("""
                **📊 Basic Analytics**
                - Message counting
                - Sender distribution
                - Timeline visualization
                """)
            
            with feature_col2:
                st.write("""
                **❤️ Health Scoring**
                - Communication balance
                - Response time analysis
                - Engagement metrics
                """)
            
            with feature_col3:
                st.write("""
                **📈 Visualizations**
                - Charts and graphs
                - Activity timelines
                - Distribution plots
                """)
        
        # Instructions
        st.subheader("📖 How to Use")
        
        with st.expander("💬 WhatsApp Chat Export"):
            st.markdown("""
            1. Open WhatsApp on your phone
            2. Go to the chat you want to analyze
            3. Tap the three dots menu → More → Export chat
            4. Choose "Without Media" for faster processing
            5. Upload the exported .txt file here
            """)
        
        with st.expander("📱 Telegram Chat Export"):
            st.markdown("""
            1. Open Telegram Desktop
            2. Go to Settings → Advanced → Export Telegram data
            3. Select the chat(s) you want to export
            4. Choose JSON format
            5. Upload the exported .json file here
            """)
        
        if st.session_state.ingestion_available:
            with st.expander("🔧 Advanced Features"):
                st.markdown("""
                - **ZIP Archives**: Upload multiple files at once
                - **Images**: Screenshots, photos with OCR text extraction
                - **PDFs**: Document text extraction and analysis
                - **Media Files**: Audio/video metadata extraction
                - **Mixed Content**: Process different file types together
                """)
        
        # Sample data information
        st.subheader("📝 Privacy & Security")
        st.info("""
        🔒 **Your data is safe**: All processing happens in your browser session. 
        No chat data is stored or transmitted to external servers. 
        Files are processed temporarily and discarded after analysis.
        """)

        # Footer with module status
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏗️ Built with:**")
            st.markdown("- Streamlit for the web interface")
            st.markdown("- Pandas & NumPy for data processing")
            st.markdown("- Plotly for interactive visualizations")
            if st.session_state.ingestion_available:
                st.markdown("- PIL, PyTesseract for OCR")
                st.markdown("- PDFPlumber for PDF processing")
        
        with col2:
            st.markdown("**📊 Current Status:**")
            total_modules = len(executed_modules)
            successful_modules = sum(1 for m in executed_modules.values() if m.get('loaded', False))
            st.markdown(f"- Modules loaded: {successful_modules}/{total_modules}")
            if st.session_state.ingestion_available:
                st.markdown("- ✅ Advanced file processing")
            else:
                st.markdown("- ⚠️ Basic file processing only")
            
            if st.session_state.relationship_health_available:
                st.markdown("- ✅ Advanced health analysis")
            else:
                st.markdown("- ⚠️ Basic health analysis only")


if __name__ == "__main__":
    main()
