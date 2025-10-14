import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modular components
from src.ingest.ingestion import process_uploaded_file
from src.parser.whatsapp_parser import WhatsAppParser
from src.parser.telegram_parser import parse_telegram_chat
from src.analysis.eda import ChatEDA
from src.analysis.sentiment import analyze_sentiment

# Configure page
st.set_page_config(
    page_title="Chat Analyzer Pro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styling"""
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
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

def calculate_health_score(df):
    """Calculate relationship health score"""
    if len(df) < 2:
        return None
    
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
    participants = list(message_counts.index)
    if len(participants) >= 2:
        main_participant_pct = (message_counts.iloc[0] / len(df)) * 100
        balance_score = 100 - abs(main_participant_pct - 50)
        balance_points = (balance_score / 100) * 25
    else:
        balance_points = 12.5

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

    date_range = (df['datetime'].max() - df['datetime'].min()).days + 1
    conversations_per_day = len(initiators_df) / date_range if date_range > 0 else 0
    if conversations_per_day >= 2:
        consistency_points = 15
    elif conversations_per_day >= 1:
        consistency_points = conversations_per_day * 7.5
    else:
        consistency_points = conversations_per_day * 15

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

def display_health_dashboard(health_results, df):
    """Display health score dashboard"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
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

def display_charts(health_results, df):
    """Display analysis charts"""
    st.subheader("📈 Analysis Dashboard")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_pie = px.pie(
            values=health_results['message_counts'].values,
            names=health_results['message_counts'].index,
            title="💬 Message Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col2:
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
    df['date_str'] = df['datetime'].dt.date.astype(str)
    daily_activity = df.groupby('date_str').size().reset_index(name='messages')
    fig_line = px.line(daily_activity, x='date_str', y='messages', title='📅 Daily Message Activity', markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

def display_insights(health_results):
    """Display detailed insights"""
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

def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">💬 Chat Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze your WhatsApp and Telegram conversations with AI-powered insights")
    
    # Sidebar
    st.sidebar.title("📁 Upload Your Chat")
    
    # File upload - supports multiple formats
    uploaded_file = st.sidebar.file_uploader(
        "Choose a chat file",
        type=['txt', 'json', 'zip', 'pdf', 'png', 'jpg', 'jpeg'],
        help="Upload WhatsApp .txt, Telegram .json, or ZIP/PDF/Images for OCR"
    )
    
    if uploaded_file is not None:
        try:
            # Use ingestion module for unified parsing
            with st.spinner('🔍 Processing your file...'):
                messages, media_ocr = process_uploaded_file(uploaded_file)
            
            if not messages:
                st.error("❌ No messages found in the file. Please check the format.")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(messages)
            
            # Ensure datetime column
            if 'datetime' not in df.columns and 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
            elif 'datetime' not in df.columns:
                st.error("❌ Could not parse datetime information from messages.")
                return
            
            # Rename author/sender to consistent 'sender'
            if 'author' in df.columns and 'sender' not in df.columns:
                df['sender'] = df['author']
            
            # Rename text to message if needed
            if 'text' in df.columns and 'message' not in df.columns:
                df['message'] = df['text']
            
            # Add message_length if not present
            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].str.len()
            
            # Drop rows with invalid datetime
            df = df.dropna(subset=['datetime'])
            
            if df.empty:
                st.error("❌ Could not parse valid messages with timestamps.")
                return
            
            st.sidebar.success(f"✅ Parsed {len(df)} messages from {len(df['sender'].unique())} participants")
            
            # Show media OCR results if any
            if media_ocr:
                with st.sidebar.expander("📷 Media/OCR Results"):
                    for item in media_ocr[:5]:  # Show first 5
                        st.write(f"**{item.get('file', 'Unknown')}**")
                        if 'ocr' in item:
                            st.text(item['ocr'][:100] + "..." if len(item['ocr']) > 100 else item['ocr'])
                        if 'note' in item:
                            st.caption(item['note'])
            
            # Analysis
            with st.spinner('🔍 Analyzing your chat...'):
                health_results = calculate_health_score(df)
            
            if health_results:
                # Display health dashboard
                display_health_dashboard(health_results, df)
                
                # Display charts
                display_charts(health_results, df)
                
                # Display insights
                display_insights(health_results)
                
                # Raw data preview
                if st.checkbox("📋 Show Raw Data"):
                    st.dataframe(df[['datetime', 'sender', 'message', 'message_length']].head(50))
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            import traceback
            with st.expander("🔧 Debug Info"):
                st.code(traceback.format_exc())
    
    else:
        # Welcome message
        st.markdown("""
        <div style="border: 2px dashed #1f77b4; border-radius: 10px; padding: 2rem; text-align: center; margin: 1rem 0;">
            <h3>👋 Welcome to Chat Analyzer Pro!</h3>
            <p>Upload your WhatsApp (.txt), Telegram (.json), ZIP, PDF, or Images to get started.</p>
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
        
        st.subheader("📁 Supported Formats")
        st.write("""
        - **WhatsApp**: Export chat as `.txt` file
        - **Telegram**: Export chat as `.json` file
        - **ZIP Archives**: Upload multiple files at once
        - **PDF/Images**: OCR extraction for scanned chats
        """)

if __name__ == "__main__":
    main()
