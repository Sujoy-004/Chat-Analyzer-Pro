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
        response = requests.get(css_url, timeout=5)
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
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
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
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

# WhatsApp parser function
def parse_whatsapp_simple(content):
    """Enhanced WhatsApp parser for Streamlit"""
    messages = []
    lines = content.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        # Multiple WhatsApp format patterns
        patterns = [
            r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*-?\s*([^:]+):\s*(.+)',
            r'(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\s*-\s*([^:]+):\s*(.+)',
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s+([^:]+):\s*(.+)'
        ]
        
        match = None
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                break
                
        if match:
            date_str, time_str, sender, message = match.groups()
            
            try:
                # Parse date with multiple formats
                date_formats = ['%d/%m/%y', '%d/%m/%Y', '%m/%d/%y', '%m/%d/%Y']
                date_obj = None
                for date_format in date_formats:
                    try:
                        date_obj = datetime.strptime(date_str, date_format)
                        break
                    except:
                        continue
                
                if date_obj is None:
                    continue
                
                # Parse time
                try:
                    if 'AM' in time_str.upper() or 'PM' in time_str.upper():
                        time_obj = datetime.strptime(time_str, '%I:%M %p').time()
                    else:
                        try:
                            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
                        except:
                            time_obj = datetime.strptime(time_str, '%H:%M').time()
                except:
                    continue
                
                full_datetime = datetime.combine(date_obj.date(), time_obj)
                
                # Skip system messages
                if sender.lower() in ['system', 'whatsapp', 'messages to this chat']:
                    continue
                
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
    """Enhanced Telegram parser for Streamlit"""
    messages = []
    
    try:
        data = json.loads(json_data)
        chat_messages = data.get('messages', [])
        
        for msg in chat_messages:
            if msg.get('type') == 'message' and 'text' in msg:
                try:
                    # Handle different date formats
                    date_str = msg['date']
                    if 'T' in date_str:
                        datetime_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        datetime_obj = datetime.fromisoformat(date_str)
                    
                    # Get sender name
                    sender = msg.get('from', msg.get('from_id', 'Unknown'))
                    if isinstance(sender, dict):
                        sender = sender.get('first_name', 'Unknown')
                    
                    # Process text content
                    text_content = msg['text']
                    if isinstance(text_content, list):
                        text_content = ''.join([
                            item['text'] if isinstance(item, dict) and 'text' in item 
                            else str(item) if not isinstance(item, dict) 
                            else '' for item in text_content
                        ])
                    
                    # Skip empty messages
                    if not text_content or text_content.strip() == '':
                        continue
                    
                    messages.append({
                        'datetime': datetime_obj,
                        'sender': str(sender).strip(),
                        'message': str(text_content).strip(),
                        'date': datetime_obj.date().strftime('%Y-%m-%d'),
                        'time': datetime_obj.time().strftime('%H:%M:%S'),
                        'hour': datetime_obj.hour,
                        'message_length': len(str(text_content).strip())
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

# Create download link for data
def create_download_link(df, filename):
    """Create download link for processed data"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Processed Data (CSV)</a>'
    return href

# Main Streamlit App
def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">💬 Chat Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze your WhatsApp and Telegram conversations with AI-powered insights")
    
    # Sidebar
    st.sidebar.title("📁 Upload Your Chat")
    st.sidebar.markdown("---")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a chat file",
        type=['txt', 'json'],
        help="Upload WhatsApp exported .txt file or Telegram .json export"
    )
    
    # Instructions
    with st.sidebar.expander("📖 How to export chats"):
        st.markdown("""
        **WhatsApp:**
        1. Open the chat you want to analyze
        2. Tap the three dots menu → More → Export chat
        3. Choose "Without media"
        4. Save the .txt file
        
        **Telegram:**
        1. Open Telegram Desktop
        2. Settings → Advanced → Export Telegram data
        3. Select the chat and export as JSON
        """)
    
    if uploaded_file is not None:
        # Determine file type and parse
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            with st.spinner(f'🔍 Parsing {file_type.upper()} file...'):
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
                st.error("❌ Could not parse the uploaded file. Please check the format and try again.")
                st.info("💡 Make sure your WhatsApp export is in the correct format or your Telegram JSON contains message data.")
                return
                
            # Data validation
            if len(df) < 10:
                st.warning("⚠️ Very few messages detected. Results may not be meaningful.")
            
            st.sidebar.success(f"📊 Parsed {len(df)} messages from {len(df['sender'].unique())} participants")
            
            # Show parsed data sample
            with st.sidebar.expander("👀 Preview data"):
                st.dataframe(df.head(5)[['datetime', 'sender', 'message']])
            
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
                        emoji = "🏆"
                    elif score >= 70:
                        grade = "Good"
                        color = "good"
                        emoji = "👍"
                    elif score >= 55:
                        grade = "Fair"
                        color = "fair"
                        emoji = "⚠️"
                    else:
                        grade = "Needs Improvement"
                        color = "poor"
                        emoji = "📈"
                    
                    st.markdown(f'<div class="health-score {color}">{emoji} {score:.1f}/100</div>', unsafe_allow_html=True)
                    st.markdown(f'<h3 style="text-align: center; color: gray;">Relationship Health: {grade}</h3>', unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total Messages", f"{len(df):,}")
                
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
                        title="💬 Message Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
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
                    fig_bar.add_trace(go.Bar(x=categories, y=scores, name='Your Score', 
                                           marker_color='#667eea', text=[f'{s:.1f}' for s in scores],
                                           textposition='auto'))
                    fig_bar.add_trace(go.Bar(x=categories, y=max_scores, name='Max Score', 
                                           marker_color='lightgray', opacity=0.5))
                    fig_bar.update_layout(title='🎯 Health Score Breakdown', barmode='overlay',
                                         yaxis_title="Points")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Daily activity
                daily_activity = df.groupby('date').size().reset_index(name='messages')
                fig_line = px.line(daily_activity, x='date', y='messages', 
                                 title='📅 Daily Message Activity', markers=True,
                                 line_shape='spline')
                fig_line.update_traces(line_color='#667eea', marker_color='#764ba2')
                st.plotly_chart(fig_line, use_container_width=True)
                
                # Hourly activity heatmap
                if len(df) > 50:  # Only show for larger datasets
                    hourly_activity = df.groupby(['sender', 'hour']).size().unstack(fill_value=0)
                    fig_heatmap = px.imshow(hourly_activity.values, 
                                          x=hourly_activity.columns,
                                          y=hourly_activity.index,
                                          title='🕒 Activity Heatmap by Hour',
                                          labels={'x': 'Hour of Day', 'y': 'Participant'},
                                          aspect='auto')
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Detailed insights
                st.subheader("🔍 Detailed Insights")
                
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.markdown("**💪 Strengths:**")
                    strengths_found = False
                    if health_results['balance_points'] >= 20:
                        st.success("✅ Well-balanced message distribution")
                        strengths_found = True
                    if health_results['init_points'] >= 16:
                        st.success("✅ Good conversation initiation balance")
                        strengths_found = True
                    if health_results['response_points'] >= 20:
                        st.success("✅ Responsive communication")
                        strengths_found = True
                    if health_results['consistency_points'] >= 12:
                        st.success("✅ Consistent communication pattern")
                        strengths_found = True
                    if not strengths_found:
                        st.info("💡 Focus on the improvement areas below")
                
                with insight_col2:
                    st.markdown("**⚠️ Areas for Improvement:**")
                    improvements_found = False
                    if health_results['response_points'] < 20:
                        st.warning("🔄 Could improve response times")
                        improvements_found = True
                    if health_results['balance_points'] < 20:
                        st.warning("⚖️ Could balance message distribution")
                        improvements_found = True
                    if health_results['init_points'] < 16:
                        st.warning("🚀 Could balance conversation initiation")
                        improvements_found = True
                    if health_results['consistency_points'] < 12:
                        st.warning("📅 Could improve communication consistency")
                        improvements_found = True
                    if not improvements_found:
                        st.success("🎉 Great communication patterns!")
                
                # Download processed data
                st.subheader("📥 Export Data")
                st.markdown(create_download_link(df, f"chat_analysis_{datetime.now().strftime('%Y%m%d')}.csv"), 
                           unsafe_allow_html=True)
                
                # Raw data preview
                if st.checkbox("📋 Show Raw Data"):
                    st.dataframe(df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please check your file format and try again. Make sure it's a valid WhatsApp or Telegram export.")
    
    else:
        # Welcome message
        st.markdown("""
        <div style="border: 2px dashed #667eea; border-radius: 15px; padding: 2rem; text-align: center; margin: 1rem 0; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));">
            <h3>👋 Welcome to Chat Analyzer Pro!</h3>
            <p>Upload your WhatsApp (.txt) or Telegram (.json) chat export to get started.</p>
            <p>Get insights into your communication patterns, relationship health, and more!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.subheader("🚀 Features")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
            **📊 Analytics**
            - Message statistics and trends
            - Communication patterns analysis
            - Response time evaluation
            - Activity heatmaps
            """)
        
        with feature_col2:
            st.markdown("""
            **🏥 Health Score**
            - Comprehensive relationship assessment
            - Balance metrics calculation
            - Engagement quality analysis
            - Personalized insights
            """)
        
        with feature_col3:
            st.markdown("""
            **📈 Visualizations**
            - Interactive charts and graphs
            - Daily/hourly activity timelines
            - Distribution analysis
            - Export processed data
            """)
        
        # Sample data demonstration
        st.subheader("🎯 Try with Sample Data")
        if st.button("📊 Load Sample Analysis"):
            try:
                sample_url = "https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/data/processed/example_parsed.csv"
                sample_df = pd.read_csv(sample_url)
                sample_df['datetime'] = pd.to_datetime(sample_df['datetime'])
                
                st.success("✅ Sample data loaded!")
                health_results = calculate_health_score(sample_df)
                
                if health_results:
                    score = health_results['total_score']
                    st.metric("Sample Health Score", f"{score:.1f}/100")
                    
                    # Quick sample visualization
                    fig_sample = px.pie(values=health_results['message_counts'].values,
                                       names=health_results['message_counts'].index,
                                       title="Sample: Message Distribution")
                    st.plotly_chart(fig_sample, use_container_width=True)
                    
            except Exception as e:
                st.error("Could not load sample data")

if __name__ == "__main__":
    main()
