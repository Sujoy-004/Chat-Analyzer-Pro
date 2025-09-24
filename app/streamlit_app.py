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
warnings.filterwarnings('ignore')

# Install required packages for advanced ingestion
def setup_advanced_features():
    """Setup advanced features with automatic installation and configuration"""
    global ADVANCED_FEATURES
    ADVANCED_FEATURES = False
    
    # Try to import and configure required packages
    try:
        # Install packages if missing
        import subprocess
        import sys
        
        # Check and install Python packages
        required_packages = ['pillow', 'pytesseract', 'pdfplumber', 'pdf2image']
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                st.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        # Now import the packages
        from PIL import Image
        import pytesseract
        import pdfplumber
        from pdf2image import convert_from_bytes
        
        # Configure Tesseract path for Windows
        import platform
        if platform.system() == "Windows":
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Tesseract-OCR\tesseract.exe'
            ]
            
            tesseract_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    tesseract_found = True
                    break
            
            if not tesseract_found:
                st.warning("Tesseract not found. OCR features will be limited.")
                return False
        
        ADVANCED_FEATURES = True
        return True
        
    except Exception as e:
        st.error(f"Could not setup advanced features: {e}")
        return False

# Simple OCR fallback using basic text extraction
def simple_ocr_fallback(image_bytes):
    """Basic image text extraction without tesseract"""
    try:
        from PIL import Image
        import io
        
        # Just return basic info about the image
        img = Image.open(io.BytesIO(image_bytes))
        return f"[Image detected: {img.size[0]}x{img.size[1]} pixels, format: {img.format}]"
    except:
        return "[Image file detected but could not process]"

# Setup advanced features
ADVANCED_FEATURES = setup_advanced_features()

# Configure page
st.set_page_config(
    page_title="Chat Analyzer Pro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS from GitHub"""
    try:
        css_url = "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/app/assets/style.css"
        response = requests.get(css_url)
        if response.status_code == 200:
            st.markdown(f"<style>{response.text}</style>", unsafe_allow_html=True)
    except:
        # Fallback basic styling
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
        .ocr-result {
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

# Robust ingestion function with fallbacks
def process_uploaded_file_with_fallbacks(uploaded_file):
    """
    Process uploaded file with graceful fallbacks when advanced features aren't available
    """
    filename = uploaded_file.name
    content = uploaded_file.getvalue()
    file_type = filename.lower().split('.')[-1]
    
    messages = []
    media_ocr = []
    
    try:
        # ZIP files - basic extraction
        if file_type == 'zip':
            import zipfile
            import io
            
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    for member in z.namelist():
                        if member.endswith('/'):
                            continue
                        
                        try:
                            with z.open(member) as f:
                                file_content = f.read()
                                
                            # Process based on file extension
                            member_ext = member.lower().split('.')[-1]
                            
                            if member_ext == 'txt':
                                # WhatsApp text file
                                text_content = file_content.decode('utf-8', errors='ignore')
                                df = parse_whatsapp_simple(text_content)
                                
                                for _, row in df.iterrows():
                                    messages.append({
                                        'author': row['sender'],
                                        'text': row['message'],
                                        'date': row['date'],
                                        'time': row['time']
                                    })
                            
                            elif member_ext == 'json':
                                # Telegram JSON file
                                json_content = file_content.decode('utf-8', errors='ignore')
                                df = parse_telegram_simple(json_content)
                                
                                for _, row in df.iterrows():
                                    messages.append({
                                        'author': row['sender'],
                                        'text': row['message'],
                                        'date': row['date'],
                                        'time': row['time']
                                    })
                            
                            elif member_ext in ['png', 'jpg', 'jpeg']:
                                # Image file - use simple fallback
                                ocr_result = simple_ocr_fallback(file_content)
                                media_ocr.append({
                                    'file': member,
                                    'ocr': ocr_result,
                                    'note': 'Basic image detection (install Tesseract for OCR)'
                                })
                                
                            else:
                                media_ocr.append({
                                    'file': member,
                                    'note': f'File detected but not processed ({member_ext})'
                                })
                                
                        except Exception as e:
                            media_ocr.append({
                                'file': member,
                                'note': f'Error processing: {e}'
                            })
                            
            except zipfile.BadZipFile:
                raise Exception("Invalid ZIP file")
                
        # Image files - basic handling
        elif file_type in ['png', 'jpg', 'jpeg']:
            if ADVANCED_FEATURES:
                # Use tesseract if available
                try:
                    import pytesseract
                    from PIL import Image
                    import io
                    
                    img = Image.open(io.BytesIO(content))
                    ocr_text = pytesseract.image_to_string(img)
                    
                    media_ocr.append({
                        'file': filename,
                        'ocr': ocr_text
                    })
                    
                    if ocr_text.strip():
                        messages.append({
                            'author': 'OCR_Extract',
                            'text': ocr_text,
                            'date': '',
                            'time': ''
                        })
                        
                except Exception as e:
                    ocr_result = simple_ocr_fallback(content)
                    media_ocr.append({
                        'file': filename,
                        'ocr': ocr_result,
                        'note': f'OCR failed: {e}'
                    })
            else:
                # Basic image info
                ocr_result = simple_ocr_fallback(content)
                media_ocr.append({
                    'file': filename,
                    'ocr': ocr_result,
                    'note': 'Install Tesseract for text extraction'
                })
        
        # PDF files - basic handling
        elif file_type == 'pdf':
            if ADVANCED_FEATURES:
                try:
                    import pdfplumber
                    import io
                    
                    pdf_text = ""
                    with pdfplumber.open(io.BytesIO(content)) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                pdf_text += text + "\n"
                    
                    if pdf_text.strip():
                        messages.append({
                            'author': 'PDF_Extract',
                            'text': pdf_text,
                            'date': '',
                            'time': ''
                        })
                        
                except Exception as e:
                    media_ocr.append({
                        'file': filename,
                        'note': f'PDF processing failed: {e}'
                    })
            else:
                media_ocr.append({
                    'file': filename,
                    'note': 'PDF detected - install pdfplumber for text extraction'
                })
        
        # Regular text/json files
        elif file_type == 'txt':
            text_content = content.decode('utf-8', errors='ignore')
            df = parse_whatsapp_simple(text_content)
            
            for _, row in df.iterrows():
                messages.append({
                    'author': row['sender'],
                    'text': row['message'],
                    'date': row['date'],
                    'time': row['time']
                })
                
        elif file_type == 'json':
            json_content = content.decode('utf-8', errors='ignore')
            df = parse_telegram_simple(json_content)
            
            for _, row in df.iterrows():
                messages.append({
                    'author': row['sender'],
                    'text': row['message'],
                    'date': row['date'],
                    'time': row['time']
                })
        
        else:
            raise Exception(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        raise Exception(f"Error processing {filename}: {e}")
    
    return messages, media_ocr

# Load all analysis modules
@st.cache_data
def load_analysis_modules():
    """Load analysis modules from GitHub"""
    modules = {}
    module_urls = {
        "whatsapp_parser": "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/src/parser/whatsapp_parser.py",
        "telegram_parser": "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/src/parser/telegram_parser.py",
        "relationship_health": "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/src/analysis/relationship_health.py",
        "pdf_generator": "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/src/reporting/pdf_report.py"
    }
    
    for name, url in module_urls.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                modules[name] = response.text
        except Exception as e:
            st.error(f"Failed to load {name}: {e}")
    
    return modules

# Fallback parsers for when ingestion module isn't available
def parse_whatsapp_simple(content):
    """Simple WhatsApp parser for Streamlit"""
    messages = []
    lines = content.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        # Match WhatsApp format: [DD/MM/YY, HH:MM:SS] Name: Message
        pattern = r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*-?\s*([^:]+):\s*(.+)'
        match = re.match(pattern, line)
        
        if match:
            date_str, time_str, sender, message = match.groups()
            
            try:
                # Parse date
                if len(date_str.split('/')[2]) == 2:
                    date_obj = datetime.strptime(date_str, '%d/%m/%y')
                else:
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                
                # Parse time
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
                    'time': full_datetime.time().strftime('%H:%M:%S'),
                    'hour': full_datetime.hour,
                    'message_length': len(message.strip())
                })
            except Exception as e:
                continue
    
    return pd.DataFrame(messages)

def parse_telegram_simple(json_data):
    """Simple Telegram parser for Streamlit"""
    messages = []
    
    try:
        data = json.loads(json_data)
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
                        'time': datetime_obj.time().strftime('%H:%M:%S'),
                        'hour': datetime_obj.hour,
                        'message_length': len(text_content)
                    })
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Error parsing Telegram data: {e}")
    
    return pd.DataFrame(messages)

# Calculate relationship health (Day 4 method)
def calculate_health_score(df):
    """Calculate relationship health score using Day 4 methodology"""
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
    
    # Calculate health score components (Day 4 method)
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
    
    return {
        'total_score': total_score,
        'balance_points': balance_points,
        'init_points': init_points,
        'response_points': response_points,
        'consistency_points': consistency_points,
        'engagement_points': engagement_points,
        'message_counts': message_counts,
        'response_df': response_df,
        'initiators_df': initiators_df
    }

def convert_messages_to_dataframe(messages):
    """Convert normalized messages from ingestion to DataFrame"""
    df_data = []
    for msg in messages:
        try:
            # Handle datetime parsing
            if msg.get('date') and msg.get('time'):
                datetime_str = f"{msg['date']} {msg['time']}"
                try:
                    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                except:
                    # Try alternative parsing
                    try:
                        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    except:
                        dt = datetime.now()  # fallback
            else:
                dt = datetime.now()  # fallback
            
            df_data.append({
                'datetime': dt,
                'sender': msg.get('author', 'Unknown'),
                'message': msg.get('text', ''),
                'date': msg.get('date', dt.date().strftime('%Y-%m-%d')),
                'time': msg.get('time', dt.time().strftime('%H:%M:%S')),
                'hour': dt.hour,
                'message_length': len(msg.get('text', ''))
            })
        except Exception:
            # Skip malformed messages
            continue
    
    return pd.DataFrame(df_data)

def display_ocr_results(media_ocr):
    """Display OCR results in an organized way"""
    if not media_ocr:
        return
    
    st.subheader("📄 OCR & Media Processing Results")
    
    # Group by type
    images = [item for item in media_ocr if 'ocr' in item]
    errors = [item for item in media_ocr if 'note' in item and 'error' in item.get('note', '').lower()]
    other = [item for item in media_ocr if item not in images and item not in errors]
    
    if images:
        st.write("**🖼️ Image OCR Results:**")
        for i, item in enumerate(images[:5], 1):  # Show first 5
            with st.expander(f"📷 {item.get('file', f'Image {i}')}"):
                ocr_text = item.get('ocr', '')
                if ocr_text.strip():
                    st.markdown(f'<div class="ocr-result">{ocr_text[:1000]}{"..." if len(ocr_text) > 1000 else ""}</div>', 
                              unsafe_allow_html=True)
                    if len(ocr_text) > 1000:
                        if st.button(f"Show full text for Image {i}"):
                            st.text_area("Full extracted text:", ocr_text, height=200, key=f"full_ocr_{i}")
                else:
                    st.info("No text detected in this image")
    
    if errors:
        st.write("**⚠️ Processing Issues:**")
        for item in errors:
            st.warning(f"📁 {item.get('file', 'Unknown file')}: {item.get('note', 'Unknown error')}")
    
    if other:
        st.write("**📁 Other Files Processed:**")
        for item in other:
            st.info(f"📄 {item.get('file', 'Unknown file')}: {item.get('note', 'Processed')}")

# Main Streamlit App
def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">💬 Chat Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze your WhatsApp, Telegram conversations, images, and documents with AI-powered insights")
    
    # Sidebar
    st.sidebar.title("📁 Upload Your Chat")
    
    # File upload with expanded format support
    file_types = ['txt', 'json', 'zip', 'png', 'jpg', 'jpeg', 'pdf']
    help_text = "Upload chat exports, ZIP files, screenshots, or PDF documents"
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=file_types,
        help=help_text
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            df = None
            media_ocr = []
            
            # Use our robust processing function
            with st.spinner('🔍 Processing file...'):
                messages, media_ocr = process_uploaded_file_with_fallbacks(uploaded_file)
            
            if messages:
                df = convert_messages_to_dataframe(messages)
                
                # Success message based on file type
                if file_type == 'zip':
                    st.sidebar.success(f"✅ ZIP processed: {len(messages)} messages found")
                elif file_type in ['png', 'jpg', 'jpeg']:
                    st.sidebar.success(f"✅ Image processed")
                    if ADVANCED_FEATURES:
                        st.sidebar.info("📝 OCR completed")
                    else:
                        st.sidebar.info("📝 Basic image detection (install Tesseract for OCR)")
                elif file_type == 'pdf':
                    st.sidebar.success(f"✅ PDF processed")
                    if not ADVANCED_FEATURES:
                        st.sidebar.info("📄 Install pdfplumber for better PDF processing")
                else:
                    st.sidebar.success(f"✅ File processed: {len(messages)} messages")
            
            if not messages:
                st.error("❌ No readable content found in the uploaded file.")
                return
            
            st.sidebar.success(f"📊 Analyzed {len(df)} messages from {len(df['sender'].unique())} participants")
            
            # Analysis
            with st.spinner('🔍 Analyzing your data...'):
                health_results = calculate_health_score(df)
            
            if health_results:
                # Display results
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Health Score
                    score = health_results['total_score']
                    if score >= 85:
                        grade = "Excellent"
                        color = "excellent"
                    elif score >= 70:
                        grade = "Good"
                        color = "good"
                    elif score >= 55:
                        grade = "Fair"
                        color = "fair"
                    else:
                        grade = "Needs Improvement"
                        color = "poor"
                    
                    st.markdown(f'<div class="health-score {color}">{score:.1f}/100</div>', unsafe_allow_html=True)
                    st.markdown(f'<h3 style="text-align: center; color: gray;">Relationship Health: {grade}</h3>', unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total Messages", len(df))
                
                with col2:
                    st.metric("👥 Participants", len(df['sender'].unique()))
                
                with col3:
                    date_range = (df['datetime'].max() - df['datetime'].min()).days + 1
                    st.metric("📅 Days Analyzed", date_range)
                
                with col4:
                    avg_response = health_results['response_df']['response_time_minutes'].mean() if len(health_results['response_df']) > 0 else 0
                    st.metric("⏱️ Avg Response", f"{avg_response:.1f} min")
                
                # Charts
                st.subheader("📈 Analysis Dashboard")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Message distribution
                    fig_pie = px.pie(
                        values=health_results['message_counts'].values,
                        names=health_results['message_counts'].index,
                        title="💬 Message Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with chart_col2:
                    # Score breakdown
                    categories = ['Communication\nBalance', 'Initiation\nBalance', 'Response\nQuality', 'Consistency', 'Engagement']
                    scores = [
                        health_results['balance_points'],
                        health_results['init_points'],
                        health_results['response_points'],
                        health_results['consistency_points'],
                        health_results['engagement_points']
                    ]
                    max_scores = [25, 20, 25, 15, 15]
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(x=categories, y=scores, name='Score', marker_color='#1f77b4'))
                    fig_bar.add_trace(go.Bar(x=categories, y=max_scores, name='Max Score', marker_color='lightgray', opacity=0.5))
                    fig_bar.update_layout(title='🎯 Health Score Breakdown', barmode='overlay')
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Daily activity
                daily_activity = df.groupby('date').size().reset_index(name='messages')
                fig_line = px.line(daily_activity, x='date', y='messages', title='📅 Daily Message Activity', markers=True)
                st.plotly_chart(fig_line, use_container_width=True)
                
                # OCR Results Display
                if media_ocr:
                    display_ocr_results(media_ocr)
                
                # Detailed insights
                st.subheader("🔍 Detailed Insights")
                
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.write("**💪 Strengths:**")
                    if health_results['balance_points'] >= 20:
                        st.write("✅ Well-balanced message distribution")
                    if health_results['init_points'] >= 16:
                        st.write("✅ Good conversation initiation balance")
                    if health_results['response_points'] >= 20:
                        st.write("✅ Responsive communication")
                    if health_results['consistency_points'] >= 12:
                        st.write("✅ Consistent communication pattern")
                
                with insight_col2:
                    st.write("**⚠️ Areas for Improvement:**")
                    if health_results['response_points'] < 20:
                        st.write("🔄 Could improve response times")
                    if health_results['balance_points'] < 20:
                        st.write("⚖️ Could balance message distribution")
                    if health_results['init_points'] < 16:
                        st.write("🚀 Could balance conversation initiation")
                
                # Raw data preview
                if st.checkbox("📋 Show Raw Data"):
                    st.dataframe(df.head(20))
            
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.exception(e)  # Show full traceback for debugging
    
    else:
        # Welcome message
        st.markdown("""
        <div style="border: 2px dashed #1f77b4; border-radius: 10px; padding: 2rem; text-align: center; margin: 1rem 0;">
            <h3>👋 Welcome to Chat Analyzer Pro!</h3>
            <p>Upload your chat files to get comprehensive communication insights!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.subheader("🚀 Features")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.write("""
            **📊 Analytics**
            - Message statistics
            - Communication patterns
            - Response time analysis
            """)
        
        with feature_col2:
            st.write("""
            **📁 File Support**
            - WhatsApp (.txt)
            - Telegram (.json)
            - ZIP archives
            - Screenshots (OCR)
            - PDF documents
            """)
        
        with feature_col3:
            st.write("""
            **🏥 Health Score**
            - Relationship assessment
            - Balance metrics  
            - Engagement quality
            """)
        
        # Installation instructions if advanced features not available
        if not ADVANCED_FEATURES:
            st.info("""
            **📦 Enable Advanced Features:**
            Install additional packages for ZIP, OCR, and PDF support:
            ```bash
            pip install pillow pytesseract pdfplumber pdf2image
            ```
            Also install system packages: `tesseract-ocr` and `poppler-utils`
            """)

if __name__ == "__main__":
    main()
