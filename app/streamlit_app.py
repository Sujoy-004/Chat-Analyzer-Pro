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
        </style>
        """, unsafe_allow_html=True)

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

# WhatsApp parser function
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

# Telegram parser function
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

# Main Streamlit App
def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">💬 Chat Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze your WhatsApp and Telegram conversations with AI-powered insights")
    
    # Sidebar
    st.sidebar.title("📁 Upload Your Chat")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a chat file",
        type=['txt', 'json'],
        help="Upload WhatsApp exported .txt file or Telegram .json export"
    )
    
    if uploaded_file is not None:
        # Determine file type and parse
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_type == 'txt':
                # WhatsApp file
                content = uploaded_file.getvalue().decode('utf-8')
                st.sidebar.success("✅ WhatsApp file detected")
                df = parse_whatsapp_simple(content)
                
            elif file_type == 'json':
                # Telegram file
                content = uploaded_file.getvalue().decode('utf-8')
                st.sidebar.success("✅ Telegram file detected")
                df = parse_telegram_simple(content)
            
            if df.empty:
                st.error("❌ Could not parse the uploaded file. Please check the format.")
                return
            
            st.sidebar.success(f"📊 Parsed {len(df)} messages from {len(df['sender'].unique())} participants")
            
            # Analysis
            with st.spinner('🔍 Analyzing your chat...'):
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
    
    else:
        # Welcome message
        st.markdown("""
        <div style="border: 2px dashed #1f77b4; border-radius: 10px; padding: 2rem; text-align: center; margin: 1rem 0;">
            <h3>👋 Welcome to Chat Analyzer Pro!</h3>
            <p>Upload your WhatsApp (.txt) or Telegram (.json) chat export to get started.</p>
            <p>Get insights into your communication patterns, relationship health, and more!</p>
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
            **🏥 Health Score**
            - Relationship assessment
            - Balance metrics
            - Engagement quality
            """)
        
        with feature_col3:
            st.write("""
            **📈 Visualizations**
            - Interactive charts
            - Daily activity timeline
            - Distribution analysis
            """)

if __name__ == "__main__":
    main()
  
