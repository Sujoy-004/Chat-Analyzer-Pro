import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
import base64
import numpy as np
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

# Modern dark theme color palette
COLORS = {
    'primary': '#00D9FF',      # Cyan
    'secondary': '#FF6B9D',    # Pink
    'accent': '#C75EFF',       # Purple
    'success': '#00F5A0',      # Mint green
    'warning': '#FFD166',      # Golden yellow
    'danger': '#FF6B6B',       # Coral red
    'info': '#4ECDC4',         # Teal
    'chart1': '#00D9FF',       # Cyan
    'chart2': '#FF6B9D',       # Pink
    'chart3': '#C75EFF',       # Purple
    'chart4': '#00F5A0',       # Mint
    'chart5': '#FFD166',       # Gold
    'chart6': '#4ECDC4',       # Teal
    'bg_dark': '#0E1117',      # Dark background
    'bg_card': '#1E222A',      # Card background
    'text': '#FAFAFA',         # Light text
    'text_muted': '#8B92A6'    # Muted text
}

# Configure page
st.set_page_config(
    page_title="Chat Analyzer Pro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS with modern dark theme
def load_css():
    """Load custom CSS styling optimized for dark background"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main-header {{
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 50%, {COLORS['accent']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
    }}
    
    .subtitle {{
        text-align: center;
        color: {COLORS['text_muted']};
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }}
    
    .health-score {{
        font-size: 5rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 0 0 40px currentColor;
        animation: pulse 2s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    .excellent {{ 
        background: linear-gradient(135deg, {COLORS['success']} 0%, {COLORS['info']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .good {{ 
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['info']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .fair {{ 
        background: linear-gradient(135deg, {COLORS['warning']} 0%, {COLORS['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .poor {{ 
        background: linear-gradient(135deg, {COLORS['danger']} 0%, {COLORS['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['bg_card']} 0%, rgba(30, 34, 42, 0.8) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 217, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.2);
        transform: translateY(-2px);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {COLORS['bg_card']};
        padding: 8px;
        border-radius: 12px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        padding: 12px 24px;
        background: transparent;
        border-radius: 8px;
        color: {COLORS['text_muted']};
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(0, 217, 255, 0.1);
        color: {COLORS['primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%) !important;
        color: white !important;
        box-shadow: 0 4px 16px rgba(0, 217, 255, 0.3);
    }}
    
    .welcome-card {{
        border: 2px dashed {COLORS['primary']};
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.05) 0%, rgba(199, 94, 255, 0.05) 100%);
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }}
    
    .feature-card {{
        background: {COLORS['bg_card']};
        padding: 2rem;
        border-radius: 16px;
        border-left: 4px solid {COLORS['primary']};
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .feature-card:hover {{
        border-left-color: {COLORS['secondary']};
        transform: translateX(5px);
        box-shadow: 0 8px 24px rgba(255, 107, 157, 0.2);
    }}
    
    div[data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
    
    div[data-testid="stMetricLabel"] {{
        color: {COLORS['text_muted']};
        font-weight: 500;
    }}
    
    .insight-box {{
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(199, 94, 255, 0.1) 100%);
        border-left: 4px solid {COLORS['accent']};
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 217, 255, 0.3);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 217, 255, 0.4);
    }}
    </style>
    """, unsafe_allow_html=True)

def create_plotly_theme():
    """Create consistent Plotly theme for all charts"""
    return dict(
        template="plotly_dark",
        paper_bgcolor=COLORS['bg_dark'],
        plot_bgcolor=COLORS['bg_card'],
        font=dict(family="Inter, sans-serif", color=COLORS['text']),
        colorway=[COLORS['chart1'], COLORS['chart2'], COLORS['chart3'], 
                  COLORS['chart4'], COLORS['chart5'], COLORS['chart6']],
        hoverlabel=dict(bgcolor=COLORS['bg_card'], font_size=13),
    )

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

def create_animated_gauge(score):
    """Create animated gauge chart for health score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score", 'font': {'size': 24, 'color': COLORS['text']}},
        delta={'reference': 75, 'increasing': {'color': COLORS['success']}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': COLORS['text_muted']},
            'bar': {'color': COLORS['primary'], 'thickness': 0.75},
            'bgcolor': COLORS['bg_card'],
            'borderwidth': 2,
            'bordercolor': COLORS['text_muted'],
            'steps': [
                {'range': [0, 55], 'color': 'rgba(255, 107, 107, 0.3)'},
                {'range': [55, 70], 'color': 'rgba(255, 209, 102, 0.3)'},
                {'range': [70, 85], 'color': 'rgba(0, 217, 255, 0.3)'},
                {'range': [85, 100], 'color': 'rgba(0, 245, 160, 0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS['danger'], 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(**create_plotly_theme(), height=400)
    return fig

def create_3d_scatter(df):
    """Create 3D scatter plot of messages by time, length, and sender"""
    df_sample = df.sample(n=min(500, len(df)))  # Sample for performance
    df_sample['hour'] = df_sample['datetime'].dt.hour
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df_sample['hour'],
        y=df_sample['message_length'],
        z=df_sample.groupby('sender').cumcount(),
        mode='markers',
        marker=dict(
            size=5,
            color=df_sample['message_length'],
            colorscale=[[0, COLORS['chart1']], [0.5, COLORS['chart3']], [1, COLORS['chart2']]],
            showscale=True,
            colorbar=dict(title="Message Length", titlefont=dict(color=COLORS['text'])),
            line=dict(width=0.5, color=COLORS['primary'])
        ),
        text=df_sample['sender'],
        hovertemplate='<b>%{text}</b><br>Hour: %{x}<br>Length: %{y}<br>Sequence: %{z}<extra></extra>'
    )])
    
    fig.update_layout(
        **create_plotly_theme(),
        title="📊 3D Message Analysis",
        scene=dict(
            xaxis=dict(title='Hour of Day', backgroundcolor=COLORS['bg_card'], gridcolor=COLORS['text_muted']),
            yaxis=dict(title='Message Length', backgroundcolor=COLORS['bg_card'], gridcolor=COLORS['text_muted']),
            zaxis=dict(title='Message Sequence', backgroundcolor=COLORS['bg_card'], gridcolor=COLORS['text_muted']),
            bgcolor=COLORS['bg_dark']
        ),
        height=600
    )
    return fig

def create_sunburst_chart(df):
    """Create sunburst chart of conversation patterns"""
    df['hour'] = df['datetime'].dt.hour
    df['period'] = df['hour'].apply(lambda x: 'Morning' if 5 <= x < 12 else 
                                              'Afternoon' if 12 <= x < 17 else 
                                              'Evening' if 17 <= x < 21 else 'Night')
    
    sunburst_data = df.groupby(['period', 'sender']).size().reset_index(name='count')
    
    fig = px.sunburst(
        sunburst_data,
        path=['period', 'sender'],
        values='count',
        title='🌅 Time Period & Sender Distribution',
        color='count',
        color_continuous_scale=[[0, COLORS['chart1']], [0.5, COLORS['chart3']], [1, COLORS['chart2']]]
    )
    
    fig.update_layout(**create_plotly_theme(), height=500)
    fig.update_traces(textfont=dict(color='white', size=14))
    return fig

def create_timeline_animation(df):
    """Create animated timeline of messages"""
    df['date'] = df['datetime'].dt.date
    daily_data = df.groupby(['date', 'sender']).size().reset_index(name='messages')
    
    fig = px.bar(
        daily_data,
        x='date',
        y='messages',
        color='sender',
        title='📅 Animated Message Timeline',
        animation_frame='date',
        color_discrete_sequence=[COLORS['chart1'], COLORS['chart2'], COLORS['chart3']],
        barmode='group'
    )
    
    fig.update_layout(**create_plotly_theme(), height=500, showlegend=True)
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['text_muted'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['text_muted'])
    return fig

def create_violin_plot(df):
    """Create violin plot for message length distribution"""
    fig = go.Figure()
    
    for sender in df['sender'].unique():
        sender_data = df[df['sender'] == sender]
        fig.add_trace(go.Violin(
            y=sender_data['message_length'],
            name=sender,
            box_visible=True,
            meanline_visible=True,
            fillcolor=COLORS['chart1'] if sender == df['sender'].unique()[0] else COLORS['chart2'],
            opacity=0.6,
            line_color='white'
        ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title='🎻 Message Length Distribution (Violin Plot)',
        yaxis_title='Message Length (characters)',
        height=500
    )
    return fig

def create_radar_chart(health_results):
    """Create radar chart for health components"""
    categories = ['Communication<br>Balance', 'Initiation<br>Balance', 'Response<br>Quality', 
                  'Consistency', 'Engagement']
    
    scores = [
        (health_results['balance_points'] / 25) * 100,
        (health_results['init_points'] / 20) * 100,
        (health_results['response_points'] / 25) * 100,
        (health_results['consistency_points'] / 15) * 100,
        (health_results['engagement_points'] / 15) * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor=COLORS['primary'],
        opacity=0.5,
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=10, color=COLORS['secondary'])
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=COLORS['text_muted'],
                tickfont=dict(color=COLORS['text'])
            ),
            angularaxis=dict(
                gridcolor=COLORS['text_muted'],
                tickfont=dict(color=COLORS['text'], size=12)
            ),
            bgcolor=COLORS['bg_card']
        ),
        title='🎯 Health Component Radar',
        height=500
    )
    return fig

def create_heatmap_calendar(df):
    """Create calendar heatmap of activity"""
    df['date'] = df['datetime'].dt.date
    df['weekday'] = df['datetime'].dt.day_name()
    df['week'] = df['datetime'].dt.isocalendar().week
    
    heatmap_data = df.groupby(['week', 'weekday']).size().reset_index(name='messages')
    pivot = heatmap_data.pivot(index='weekday', columns='week', values='messages').fillna(0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, COLORS['bg_card']], [0.5, COLORS['chart3']], [1, COLORS['chart1']]],
        hoverongaps=False,
        hovertemplate='Week: %{x}<br>Day: %{y}<br>Messages: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title='📆 Weekly Activity Heatmap',
        xaxis_title='Week Number',
        yaxis_title='Day of Week',
        height=400
    )
    return fig

def create_treemap(df):
    """Create treemap of sender and time period distribution"""
    df['hour'] = df['datetime'].dt.hour
    df['period'] = df['hour'].apply(lambda x: 'Morning' if 5 <= x < 12 else 
                                              'Afternoon' if 12 <= x < 17 else 
                                              'Evening' if 17 <= x < 21 else 'Night')
    
    treemap_data = df.groupby(['sender', 'period']).size().reset_index(name='messages')
    
    fig = px.treemap(
        treemap_data,
        path=['sender', 'period'],
        values='messages',
        title='🗺️ Sender & Time Period Treemap',
        color='messages',
        color_continuous_scale=[[0, COLORS['chart1']], [0.5, COLORS['chart3']], [1, COLORS['chart2']]]
    )
    
    fig.update_layout(**create_plotly_theme(), height=500)
    fig.update_traces(textfont=dict(color='white', size=14))
    return fig

def display_eda_analysis(df):
    """Display comprehensive EDA with interactive visualizations"""
    st.subheader("📊 Exploratory Data Analysis")
    
    try:
        eda = ChatEDA(df)
        summary = eda.generate_comprehensive_summary()
        
        # Metrics in modern cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💬 Total Messages", f"{summary['dataset_info']['total_messages']:,}")
        with col2:
            st.metric("📅 Duration", f"{summary['dataset_info']['duration_days']} days")
        with col3:
            st.metric("⏰ Peak Hour", f"{summary['activity_patterns']['peak_hour']}:00")
        with col4:
            st.metric("🌟 Active Period", summary['activity_patterns']['most_active_period'])
        
        st.markdown("---")
        
        # Row 1: 3D Scatter and Sunburst
        col1, col2 = st.columns(2)
        with col1:
            fig_3d = create_3d_scatter(df)
            st.plotly_chart(fig_3d, use_container_width=True)
        with col2:
            fig_sunburst = create_sunburst_chart(df)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Row 2: Treemap and Violin
        col1, col2 = st.columns(2)
        with col1:
            fig_treemap = create_treemap(df)
            st.plotly_chart(fig_treemap, use_container_width=True)
        with col2:
            fig_violin = create_violin_plot(df)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        # Calendar heatmap
        fig_calendar = create_heatmap_calendar(df)
        st.plotly_chart(fig_calendar, use_container_width=True)
        
    except Exception as e:
        st.error(f"EDA analysis error: {e}")

def display_sentiment_analysis(df):
    """Display sentiment analysis with modern visualizations"""
    st.subheader("😊 Sentiment Analysis")
    
    try:
        with st.spinner("Analyzing sentiment..."):
            df_sentiment = analyze_sentiment(df.copy(), message_col='message')
        
        if 'sentiment_label' in df_sentiment.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Animated donut chart
                sentiment_counts = df_sentiment['sentiment_label'].value_counts()
                fig_donut = go.Figure(data=[go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hole=0.5,
                    marker=dict(colors=[COLORS['success'], COLORS['warning'], COLORS['danger']]),
                    textfont=dict(color='white', size=14)
                )])
                fig_donut.update_layout(
                    **create_plotly_theme(),
                    title='Overall Sentiment Distribution',
                    height=400,
                    annotations=[dict(text=f'{len(df_sentiment)}<br>Messages', 
                                    x=0.5, y=0.5, font_size=20, showarrow=False,
                                    font=dict(color=COLORS['text']))]
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            
            with col2:
                # Stacked bar by sender
                if 'sender' in df_sentiment.columns:
                    sentiment_by_sender = df_sentiment.groupby(['sender', 'sentiment_label']).size().reset_index(name='count')
                    fig_stacked = px.bar(
                        sentiment_by_sender, x='sender', y='count', color='sentiment_label',
                        title="Sentiment by Sender",
                        color_discrete_map={'Positive': COLORS['success'], 
                                          'Neutral': COLORS['warning'], 
                                          'Negative': COLORS['danger']},
                        barmode='stack'
                    )
                    fig_stacked.update_layout(**create_plotly_theme(), height=400)
                    st.plotly_chart(fig_stacked, use_container_width=True)
            
            # Animated sentiment timeline
            if 'datetime' in df_sentiment.columns and 'sentiment_score' in df_sentiment.columns:
                df_sentiment['date'] = df_sentiment['datetime'].dt.date
                daily_sentiment = df_sentiment.groupby(['date', 'sender'])['sentiment_score'].mean().reset_index()
                
                fig_timeline = px.line(
                    daily_sentiment, x='date', y='sentiment_score', color='sender',
                    title="📈 Sentiment Timeline by Sender",
                    color_discrete_sequence=[COLORS['chart1'], COLORS['chart2'], COLORS['chart3']]
                )
                fig_timeline.add_hline(y=0, line_dash="dash", line_color=COLORS['text_muted'], 
                                      annotation_text="Neutral", annotation_position="right")
                fig_timeline.update_layout(**create_plotly_theme(), height=400)
                fig_timeline.update_traces(line=dict(width=3), marker=dict(size=8))
                st.plotly_chart(fig_timeline, use_container_width=True)
        
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
    
    # Animated Header
    st.markdown('<h1 class="main-header">💬 Chat Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Unlock the power of your conversations with AI-driven insights</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📁 Upload Your Chat")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a chat file",
        type=['txt', 'json', 'zip', 'pdf', 'png', 'jpg', 'jpeg'],
        help="Upload WhatsApp .txt, Telegram .json, or ZIP/PDF/Images for OCR"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner('🔍 Processing your file...'):
                messages, media_ocr = process_uploaded_file(uploaded_file)
            
            if not messages:
                st.error("❌ No messages found in the file. Please check the format.")
                return
            
            df = pd.DataFrame(messages)
            
            # Data preparation
            if 'datetime' not in df.columns and 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
            elif 'datetime' not in df.columns:
                st.error("❌ Could not parse datetime information from messages.")
                return
            
            if 'author' in df.columns and 'sender' not in df.columns:
                df['sender'] = df['author']
            
            if 'text' in df.columns and 'message' not in df.columns:
                df['message'] = df['text']
            
            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].str.len()
            
            df = df.dropna(subset=['datetime'])
            
            if df.empty:
                st.error("❌ Could not parse valid messages with timestamps.")
                return
            
            # Ensure all required columns exist
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.day_name()
            df['date'] = df['datetime'].dt.date
            
            st.sidebar.success(f"✅ Parsed {len(df)} messages from {len(df['sender'].unique())} participants")
            
            if media_ocr:
                with st.sidebar.expander("📷 Media/OCR Results"):
                    for item in media_ocr[:5]:
                        st.write(f"**{item.get('file', 'Unknown')}**")
                        if 'ocr' in item:
                            st.text(item['ocr'][:100] + "..." if len(item['ocr']) > 100 else item['ocr'])
                        if 'note' in item:
                            st.caption(item['note'])
            
            with st.spinner('🔍 Analyzing your chat...'):
                health_results = calculate_health_score(df)
            
            if health_results:
                # Create tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🏠 Overview", 
                    "📊 Interactive EDA", 
                    "😊 Sentiment", 
                    "🎯 Advanced", 
                    "📄 Export"
                ])
                
                with tab1:
                    # Animated gauge and metrics
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        score = health_results['total_score']
                        fig_gauge = create_animated_gauge(score)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col2:
                        if score >= 85:
                            grade = "Excellent"
                            color = "excellent"
                            emoji = "🌟"
                        elif score >= 70:
                            grade = "Good"
                            color = "good"
                            emoji = "✨"
                        elif score >= 55:
                            grade = "Fair"
                            color = "fair"
                            emoji = "⭐"
                        else:
                            grade = "Needs Improvement"
                            color = "poor"
                            emoji = "💫"
                        
                        st.markdown(f'<div class="health-score {color}">{emoji}<br>{score:.1f}</div>', unsafe_allow_html=True)
                        st.markdown(f'<h2 style="text-align: center; color: {COLORS["text_muted"]};">Grade: {grade}</h2>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💬 Messages", f"{len(df):,}")
                    with col2:
                        st.metric("👥 Participants", len(df['sender'].unique()))
                    with col3:
                        st.metric("📅 Days", health_results['date_range'])
                    with col4:
                        avg_response = health_results['response_df']['response_time_minutes'].mean() if len(health_results['response_df']) > 0 else 0
                        st.metric("⏱️ Avg Response", f"{avg_response:.1f} min")
                    
                    st.markdown("---")
                    
                    # Charts row
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Donut chart
                        fig_donut = go.Figure(data=[go.Pie(
                            labels=health_results['message_counts'].index,
                            values=health_results['message_counts'].values,
                            hole=0.6,
                            marker=dict(colors=[COLORS['chart1'], COLORS['chart2'], COLORS['chart3']]),
                            textfont=dict(color='white', size=14)
                        )])
                        fig_donut.update_layout(
                            **create_plotly_theme(),
                            title='💬 Message Distribution',
                            height=400,
                            annotations=[dict(text='Messages', x=0.5, y=0.5, font_size=16, 
                                            showarrow=False, font=dict(color=COLORS['text']))]
                        )
                        st.plotly_chart(fig_donut, use_container_width=True)
                    
                    with col2:
                        # Radar chart
                        fig_radar = create_radar_chart(health_results)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Timeline
                    df['date_str'] = df['datetime'].dt.date.astype(str)
                    daily_activity = df.groupby(['date_str', 'sender']).size().reset_index(name='messages')
                    
                    fig_area = px.area(
                        daily_activity, x='date_str', y='messages', color='sender',
                        title='📅 Daily Activity Timeline',
                        color_discrete_sequence=[COLORS['chart1'], COLORS['chart2'], COLORS['chart3']]
                    )
                    fig_area.update_layout(**create_plotly_theme(), height=400)
                    fig_area.update_traces(line=dict(width=2))
                    st.plotly_chart(fig_area, use_container_width=True)
                    
                    # Insights
                    st.markdown("### 💡 Key Insights")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                        st.write("**💪 Strengths:**")
                        if health_results['balance_points'] >= 20:
                            st.write("✅ Well-balanced message distribution")
                        if health_results['init_points'] >= 16:
                            st.write("✅ Good conversation initiation balance")
                        if health_results['response_points'] >= 20:
                            st.write("✅ Responsive communication")
                        if health_results['consistency_points'] >= 12:
                            st.write("✅ Consistent communication pattern")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                        st.write("**⚠️ Areas for Improvement:**")
                        if health_results['response_points'] < 20:
                            st.write("🔄 Could improve response times")
                        if health_results['balance_points'] < 20:
                            st.write("⚖️ Could balance message distribution")
                        if health_results['init_points'] < 16:
                            st.write("🚀 Could balance conversation initiation")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    display_eda_analysis(df)
                
                with tab3:
                    df_with_sentiment = display_sentiment_analysis(df)
                
                with tab4:
                    st.subheader("🎯 Advanced Analytics")
                    
                    # Response time distribution with animation
                    if len(health_results['response_df']) > 0:
                        response_df = health_results['response_df'].copy()
                        response_df['response_time_capped'] = response_df['response_time_minutes'].clip(upper=120)
                        
                        fig_hist = px.histogram(
                            response_df, x='response_time_capped', color='responder',
                            title='⏱️ Response Time Distribution',
                            nbins=30,
                            color_discrete_sequence=[COLORS['chart1'], COLORS['chart2']],
                            marginal='violin'
                        )
                        fig_hist.update_layout(**create_plotly_theme(), height=500)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Conversation flow Sankey
                    st.subheader("🔄 Conversation Flow")
                    df_sorted = df.sort_values('datetime')
                    transitions = []
                    for i in range(min(100, len(df_sorted) - 1)):  # Limit for performance
                        sender1 = df_sorted.iloc[i]['sender']
                        sender2 = df_sorted.iloc[i + 1]['sender']
                        if sender1 != sender2:
                            transitions.append((sender1, sender2))
                    
                    if transitions:
                        transition_counts = pd.DataFrame(transitions, columns=['From', 'To'])
                        transition_matrix = transition_counts.groupby(['From', 'To']).size().reset_index(name='Count')
                        
                        all_nodes = list(set(transition_matrix['From']) | set(transition_matrix['To']))
                        node_colors = [COLORS['chart1'], COLORS['chart2'], COLORS['chart3']][:len(all_nodes)]
                        
                        fig_sankey = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color=COLORS['primary'], width=2),
                                label=all_nodes,
                                color=node_colors
                            ),
                            link=dict(
                                source=[all_nodes.index(x) for x in transition_matrix['From']],
                                target=[all_nodes.index(x) for x in transition_matrix['To']],
                                value=transition_matrix['Count'],
                                color='rgba(0, 217, 255, 0.3)'
                            )
                        )])
                        
                        fig_sankey.update_layout(**create_plotly_theme(), height=500, title='Message Flow Between Participants')
                        st.plotly_chart(fig_sankey, use_container_width=True)
                
                with tab5:
                    st.subheader("📄 Export & Reports")
                    st.write("Generate comprehensive reports and export your data.")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔥 Generate PDF Report", type="primary", use_container_width=True):
                            with st.spinner("Generating PDF report..."):
                                try:
                                    pdf_data = generate_pdf_report_data(df, health_results)
                                    pdf_filename = f"chat_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                    pdf_path = generate_chat_analysis_pdf(pdf_data, pdf_filename)
                                    
                                    with open(pdf_path, "rb") as pdf_file:
                                        pdf_bytes = pdf_file.read()
                                        st.success("✅ PDF report generated successfully!")
                                        st.download_button(
                                            label="📥 Download PDF Report",
                                            data=pdf_bytes,
                                            file_name=pdf_filename,
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                
                                except Exception as e:
                                    st.error(f"Error generating PDF: {e}")
                    
                    with col2:
                        # Export CSV
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📊 Download CSV Data",
                            data=csv,
                            file_name=f"chat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                # Raw data viewer
                with st.expander("📋 View Raw Data"):
                    st.dataframe(
                        df[['datetime', 'sender', 'message', 'message_length']].head(100),
                        use_container_width=True,
                        height=400
                    )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            import traceback
            with st.expander("🔧 Debug Info"):
                st.code(traceback.format_exc())
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="welcome-card">
            <h2>👋 Welcome to Chat Analyzer Pro!</h2>
            <p>Upload your WhatsApp (.txt), Telegram (.json), ZIP, PDF, or Images to get started.</p>
            <p>Experience comprehensive insights with beautiful, interactive visualizations!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.markdown("### ✨ Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Interactive EDA")
            st.write("• 3D scatter plots")
            st.write("• Sunburst charts")
            st.write("• Treemaps")
            st.write("• Violin plots")
            st.write("• Calendar heatmaps")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("#### 🏥 Health Score")
            st.markdown("• Animated gauges")
            st.write("• Radar charts")
            st.write("• Component breakdown")
            st.write("• Personalized insights")
            st.write("• Grade system")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("#### 😊 Sentiment")
            st.write("• VADER analysis")
            st.write("• Timeline tracking")
            st.write("• Sender comparison")
            st.write("• Emotional trends")
            st.write("• Donut visualizations")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("#### 📄 Export")
            st.write("• PDF reports")
            st.write("• CSV export")
            st.write("• Professional format")
            st.write("• Charts included")
            st.write("• One-click download")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Supported formats
        st.markdown("### 📁 Supported Formats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Direct Upload:**
            - 📱 WhatsApp: Export as `.txt`
            - 💬 Telegram: Export as `.json`
            - 📦 ZIP: Multiple files at once
            """)
        
        with col2:
            st.info("""
            **OCR Support:**
            - 📄 PDF: Text extraction
            - 🖼️ Images: PNG, JPG, JPEG
            - 📸 Screenshots: Automatic OCR
            """)
        
        # How to guide
        with st.expander("❓ How to Export Your Chats"):
            st.markdown("""
            ### WhatsApp Export:
            1. Open the chat → Tap contact name
            2. Scroll down → "Export Chat"
            3. Choose "Without Media"
            4. Upload the `.txt` file here
            
            ### Telegram Export:
            1. Open Telegram Desktop
            2. Menu (⋮) → "Export Chat History"
            3. Select JSON format
            4. Upload the `.json` file here
            """)

if __name__ == "__main__":
    main()
