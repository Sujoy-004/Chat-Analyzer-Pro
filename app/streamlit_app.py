import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ALL modular components
from src.ingest.ingestion import process_uploaded_file
from src.parser.whatsapp_parser import WhatsAppParser
from src.parser.telegram_parser import parse_telegram_chat
from src.analysis.eda import ChatEDA
from src.analysis.sentiment import analyze_sentiment
from src.reporting.pdf_report import generate_chat_analysis_pdf

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
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
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
        'initiators_df': initiators_df,
        'conversations_per_day': conversations_per_day,
        'avg_msg_length': avg_msg_length,
        'date_range': date_range
    }

def create_word_frequency_chart(df):
    """Create word frequency visualization"""
    from collections import Counter
    import re
    
    all_text = ' '.join(df['message'].astype(str))
    words = re.findall(r'\b\w+\b', all_text.lower())
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'it', 'this', 'that'}
    words = [w for w in words if len(w) > 3 and w not in stop_words]
    
    word_counts = Counter(words).most_common(15)
    
    if word_counts:
        words, counts = zip(*word_counts)
        fig = px.bar(x=list(words), y=list(counts), title="📝 Top 15 Most Used Words",
                     labels={'x': 'Words', 'y': 'Frequency'})
        fig.update_traces(marker_color='#3498db')
        return fig
    return None

def create_hourly_heatmap(df):
    """Create hourly activity heatmap"""
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    hourly_activity = df.groupby(['date', 'hour']).size().reset_index(name='messages')
    pivot_table = hourly_activity.pivot(index='hour', columns='date', values='messages').fillna(0)
    
    fig = px.imshow(pivot_table, 
                    labels=dict(x="Date", y="Hour of Day", color="Messages"),
                    title="🕐 Hourly Activity Heatmap",
                    color_continuous_scale='Blues')
    fig.update_xaxes(side="bottom")
    return fig

def create_response_time_distribution(response_df):
    """Create response time distribution chart"""
    if len(response_df) == 0:
        return None
    
    # Cap at 120 minutes for better visualization
    capped_times = response_df['response_time_minutes'].clip(upper=120)
    
    fig = px.histogram(capped_times, nbins=30, 
                       title="⏱️ Response Time Distribution",
                       labels={'value': 'Response Time (minutes)', 'count': 'Frequency'})
    fig.update_traces(marker_color='#e74c3c')
    return fig

def create_conversation_flow(df):
    """Create conversation flow visualization"""
    df_sorted = df.sort_values('datetime')
    
    # Get sender transitions
    transitions = []
    for i in range(len(df_sorted) - 1):
        sender1 = df_sorted.iloc[i]['sender']
        sender2 = df_sorted.iloc[i + 1]['sender']
        if sender1 != sender2:
            transitions.append((sender1, sender2))
    
    if not transitions:
        return None
    
    transition_counts = pd.DataFrame(transitions, columns=['From', 'To'])
    transition_matrix = transition_counts.groupby(['From', 'To']).size().reset_index(name='Count')
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = list(set(transition_matrix['From']) | set(transition_matrix['To'])),
            color = "blue"
        ),
        link = dict(
            source = [list(set(transition_matrix['From']) | set(transition_matrix['To'])).index(x) for x in transition_matrix['From']],
            target = [list(set(transition_matrix['From']) | set(transition_matrix['To'])).index(x) for x in transition_matrix['To']],
            value = transition_matrix['Count']
        ))])
    
    fig.update_layout(title_text="🔄 Conversation Flow Diagram", font_size=10)
    return fig

def display_eda_analysis(df):
    """Display comprehensive EDA using the EDA module"""
    st.subheader("📊 Exploratory Data Analysis")
    
    try:
        eda = ChatEDA(df)
        summary = eda.generate_comprehensive_summary()
        
        # Display dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", summary['dataset_info']['total_messages'])
        with col2:
            st.metric("Duration (days)", summary['dataset_info']['duration_days'])
        with col3:
            st.metric("Peak Hour", summary['activity_patterns']['peak_hour'])
        with col4:
            st.metric("Active Period", summary['activity_patterns']['most_active_period'])
        
        # Activity patterns
        st.write("**Activity Insights:**")
        col1, col2 = st.columns(2)
        
        with col1:
            # Word frequency
            word_freq_fig = create_word_frequency_chart(df)
            if word_freq_fig:
                st.plotly_chart(word_freq_fig, use_container_width=True)
        
        with col2:
            # Hourly heatmap
            heatmap_fig = create_hourly_heatmap(df)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"EDA analysis error: {e}")

def display_sentiment_analysis(df):
    """Display sentiment analysis"""
    st.subheader("😊 Sentiment Analysis")
    
    try:
        # Run sentiment analysis
        with st.spinner("Analyzing sentiment..."):
            df_sentiment = analyze_sentiment(df.copy(), message_col='message')
        
        if 'sentiment_label' in df_sentiment.columns:
            # Sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = df_sentiment['sentiment_label'].value_counts()
                fig_pie = px.pie(values=sentiment_counts.values, 
                                names=sentiment_counts.index,
                                title="Overall Sentiment Distribution",
                                color_discrete_map={'Positive': '#2ecc71', 
                                                   'Neutral': '#f39c12', 
                                                   'Negative': '#e74c3c'})
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Sentiment by sender
                if 'sender' in df_sentiment.columns:
                    sentiment_by_sender = df_sentiment.groupby(['sender', 'sentiment_label']).size().reset_index(name='count')
                    fig_bar = px.bar(sentiment_by_sender, x='sender', y='count', color='sentiment_label',
                                    title="Sentiment Distribution by Sender",
                                    color_discrete_map={'Positive': '#2ecc71', 
                                                       'Neutral': '#f39c12', 
                                                       'Negative': '#e74c3c'})
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Sentiment timeline
            if 'datetime' in df_sentiment.columns and 'sentiment_score' in df_sentiment.columns:
                df_sentiment['date'] = df_sentiment['datetime'].dt.date
                daily_sentiment = df_sentiment.groupby('date')['sentiment_score'].mean().reset_index()
                
                fig_line = px.line(daily_sentiment, x='date', y='sentiment_score',
                                  title="📈 Sentiment Timeline",
                                  labels={'sentiment_score': 'Average Sentiment Score'})
                fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_line, use_container_width=True)
            
            # Sentiment metrics
            st.write("**Sentiment Metrics:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_pct = (sentiment_counts.get('Positive', 0) / len(df_sentiment)) * 100
                st.metric("Positive Messages", f"{positive_pct:.1f}%")
            
            with col2:
                neutral_pct = (sentiment_counts.get('Neutral', 0) / len(df_sentiment)) * 100
                st.metric("Neutral Messages", f"{neutral_pct:.1f}%")
            
            with col3:
                negative_pct = (sentiment_counts.get('Negative', 0) / len(df_sentiment)) * 100
                st.metric("Negative Messages", f"{negative_pct:.1f}%")
        
        return df_sentiment
    
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")
        return df

def generate_pdf_report_data(df, health_results):
    """Generate data structure for PDF report"""
    date_range_str = f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}"
    
    analysis_data = {
        'conversation_stats': {
            'total_messages': len(df),
            'unique_senders': len(df['sender'].unique()),
            'date_range': date_range_str,
            'total_conversations': len(health_results['initiators_df']) if 'initiators_df' in health_results else 0,
            'avg_response_time': health_results['response_df']['response_time_minutes'].mean() if len(health_results.get('response_df', [])) > 0 else 0
        },
        'health_score': {
            'overall_health_score': health_results['total_score'] / 100,
            'grade': 'EXCELLENT' if health_results['total_score'] >= 85 else 'GOOD' if health_results['total_score'] >= 70 else 'FAIR',
            'description': 'Strong relationship with balanced communication',
            'component_scores': {
                'initiation_balance': health_results['init_points'] / 20,
                'responsiveness': health_results['response_points'] / 25,
                'response_balance': health_results['response_points'] / 25,
                'dominance_balance': health_results['balance_points'] / 25
            },
            'strengths': [],
            'areas_for_improvement': []
        },
        'initiator_analysis': {
            'initiator_counts': health_results['initiators_df']['initiator'].value_counts().to_dict() if 'initiators_df' in health_results else {},
            'balance_score': health_results['init_points'] / 20,
            'interpretation': 'Good conversation balance'
        },
        'response_analysis': {
            'response_stats': {'mean': {}},
            'total_responses_analyzed': len(health_results.get('response_df', [])),
            'overall_avg_response_minutes': health_results['response_df']['response_time_minutes'].mean() if len(health_results.get('response_df', [])) > 0 else 0,
            'responsiveness_score': health_results['response_points'] / 25,
            'response_balance_score': health_results['response_points'] / 25
        },
        'dominance_analysis': {
            'message_distribution': health_results['message_counts'].to_dict(),
            'composite_dominance_score': health_results['balance_points'] / 25,
            'interpretation': 'Balanced participation',
            'message_count_balance': health_results['balance_points'] / 25,
            'message_length_balance': health_results['engagement_points'] / 15,
            'conversation_control_balance': health_results['init_points'] / 20
        }
    }
    
    return analysis_data

def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">💬 Chat Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive chat analysis with AI-powered insights, sentiment analysis, and PDF reports")
    
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
                    for item in media_ocr[:5]:
                        st.write(f"**{item.get('file', 'Unknown')}**")
                        if 'ocr' in item:
                            st.text(item['ocr'][:100] + "..." if len(item['ocr']) > 100 else item['ocr'])
                        if 'note' in item:
                            st.caption(item['note'])
            
            # Calculate health score
            with st.spinner('🔍 Analyzing your chat...'):
                health_results = calculate_health_score(df)
            
            if health_results:
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Overview", 
                    "📈 EDA", 
                    "😊 Sentiment", 
                    "🔍 Advanced", 
                    "📄 Report"
                ])
                
                with tab1:
                    # Health Score Overview
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
                        st.metric("📅 Days Analyzed", health_results['date_range'])
                    
                    with col4:
                        avg_response = health_results['response_df']['response_time_minutes'].mean() if len(health_results['response_df']) > 0 else 0
                        st.metric("⏱️ Avg Response", f"{avg_response:.1f} min")
                    
                    # Charts
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
                
                with tab2:
                    display_eda_analysis(df)
                
                with tab3:
                    df_with_sentiment = display_sentiment_analysis(df)
                
                with tab4:
                    st.subheader("🔍 Advanced Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Response time distribution
                        response_fig = create_response_time_distribution(health_results['response_df'])
                        if response_fig:
                            st.plotly_chart(response_fig, use_container_width=True)
                    
                    with col2:
                        # Conversation flow
                        flow_fig = create_conversation_flow(df)
                        if flow_fig:
                            st.plotly_chart(flow_fig, use_container_width=True)
                    
                    # Detailed insights
                    st.subheader("📋 Detailed Insights")
                    
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
                
                with tab5:
                    st.subheader("📄 Generate PDF Report")
                    st.write("Create a comprehensive PDF report with all your analysis results.")
                    
                    if st.button("🔥 Generate PDF Report", type="primary"):
                        with st.spinner("Generating PDF report..."):
                            try:
                                # Prepare data for PDF
                                pdf_data = generate_pdf_report_data(df, health_results)
                                
                                # Generate PDF
                                pdf_filename = f"chat_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                pdf_path = generate_chat_analysis_pdf(pdf_data, pdf_filename)
                                
                                # Read and offer download
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                    st.success("✅ PDF report generated successfully!")
                                    st.download_button(
                                        label="📥 Download PDF Report",
                                        data=pdf_bytes,
                                        file_name=pdf_filename,
                                        mime="application/pdf"
                                    )
                            
                            except Exception as e:
                                st.error(f"Error generating PDF: {e}")
                                st.info("PDF generation requires all analysis modules to be properly configured.")
                
                # Raw data preview
                if st.checkbox("📋 Show Raw Data"):
                    st.dataframe(df[['datetime', 'sender', 'message', 'message_length']].head(100))
            
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
            <p>Get comprehensive insights into your communication patterns, relationship health, sentiment analysis, and more!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.subheader("🚀 Features")
        
        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
        
        with feature_col1:
            st.write("""
            **📊 Analytics**
            - Message statistics
            - Communication patterns
            - Response time analysis
            - Activity heatmaps
            - Word frequency
            """)
        
        with feature_col2:
            st.write("""
            **🏥 Health Score**
            - Relationship assessment
            - Balance metrics
            - Engagement quality
            - Component breakdown
            - Personalized insights
            """)
        
        with feature_col3:
            st.write("""
            **😊 Sentiment Analysis**
            - VADER sentiment scoring
            - Positive/Negative detection
            - Sentiment timeline
            - Sender-wise analysis
            - Emotional trends
            """)
        
        with feature_col4:
            st.write("""
            **📄 PDF Reports**
            - Comprehensive reports
            - Visual charts
            - Detailed metrics
            - Recommendations
            - Export & share
            """)
        
        st.subheader("📁 Supported Formats")
        
        format_col1, format_col2 = st.columns(2)
        
        with format_col1:
            st.write("""
            **Direct Upload:**
            - **WhatsApp**: Export chat as `.txt` file
            - **Telegram**: Export chat as `.json` file
            - **ZIP Archives**: Upload multiple files at once
            """)
        
        with format_col2:
            st.write("""
            **OCR Support:**
            - **PDF Files**: Extract text from PDF documents
            - **Images**: PNG, JPG, JPEG support with OCR
            - **Scanned Chats**: Automatic text extraction
            """)
        
        # How to export guide
        with st.expander("❓ How to Export Your Chats"):
            st.markdown("""
            ### WhatsApp Export:
            1. Open the chat you want to analyze
            2. Tap on the contact/group name at the top
            3. Scroll down and tap "Export Chat"
            4. Choose "Without Media"
            5. Upload the `.txt` file here
            
            ### Telegram Export:
            1. Open Telegram Desktop
            2. Click on the three dots (⋮) menu
            3. Select "Export Chat History"
            4. Choose JSON format
            5. Upload the `.json` file here
            
            ### OCR Upload:
            - Screenshot your chat conversations
            - Or scan chat printouts as images/PDFs
            - Upload directly for automatic text extraction
            """)
        
        # Sample data option
        st.subheader("🎯 Try with Sample Data")
        st.write("Don't have a chat file? Try our sample datasets:")
        
        sample_col1, sample_col2 = st.columns(2)
        
        with sample_col1:
            if st.button("📱 Load WhatsApp Sample"):
                st.info("Sample WhatsApp data would be loaded here (if available in your repo)")
        
        with sample_col2:
            if st.button("💬 Load Telegram Sample"):
                st.info("Sample Telegram data would be loaded here (if available in your repo)")

if __name__ == "__main__":
    main()
