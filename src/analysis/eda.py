import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
import re

class ChatEDAWithPlotly:
    """
    Comprehensive Exploratory Data Analysis for Chat Data with Plotly visualizations
    """
    
    def __init__(self, df):
        """Initialize with chat DataFrame"""
        self.df = self.prepare_data(df.copy())
        self.summary = None
    
    def prepare_data(self, df):
        """Prepare and enhance data for analysis"""
        # Convert datetime columns
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Add time-based features
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['month'] = df['datetime'].dt.month_name()
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['weekday'] >= 5
        df['time_period'] = df['hour'].apply(self._categorize_time)
        
        # Add message features
        df['word_count'] = df['message'].str.split().str.len()
        df['is_media'] = df['message'].str.contains('<Media omitted>', case=False, na=False)
        df['has_emoji'] = df['message'].str.contains(r'[😀-🙏🌀-🗿]', regex=True, na=False)
        
        return df
    
    def _categorize_time(self, hour):
        """Categorize hour into time periods"""
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon' 
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    def create_3d_activity_visualization(self):
        """
        Create 3D visualization of chat activity (day, hour, message count)
        with CORRECTED colorbar configuration
        """
        # Aggregate data
        activity = self.df.groupby(['weekday', 'hour']).size().reset_index(name='message_count')
        
        # Create 3D scatter plot with corrected colorbar
        fig = go.Figure(data=[go.Scatter3d(
            x=activity['weekday'],
            y=activity['hour'],
            z=activity['message_count'],
            mode='markers',
            marker=dict(
                size=8,
                color=activity['message_count'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    # CORRECTED: Use nested 'title' object instead of 'titlefont'
                    title=dict(
                        text='Messages',
                        font=dict(size=12, color='black')
                    ),
                    # Use 'tickfont' for tick labels
                    tickfont=dict(size=10, color='black'),
                    thickness=20,
                    len=0.7,
                    x=1.02,
                    xanchor='left'
                ),
                line=dict(color='darkgray', width=0.5)
            ),
            text=[f'Day: {d}<br>Hour: {h}<br>Messages: {m}' 
                  for d, h, m in zip(activity['weekday'], activity['hour'], activity['message_count'])],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        # Update layout
        fig.update_layout(
            title='3D Chat Activity Pattern',
            scene=dict(
                xaxis=dict(
                    title='Day of Week',
                    tickmode='array',
                    tickvals=[0, 1, 2, 3, 4, 5, 6],
                    ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                ),
                yaxis=dict(title='Hour of Day'),
                zaxis=dict(title='Message Count')
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def create_heatmap_with_colorbar(self):
        """
        Create heatmap of hourly activity by day with corrected colorbar
        """
        # Pivot data for heatmap
        pivot_data = self.df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex([d for d in day_order if d in pivot_data.index])
        
        # Create heatmap with corrected colorbar
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='YlOrRd',
            colorbar=dict(
                # CORRECTED: Use nested 'title' object
                title=dict(
                    text='Messages',
                    font=dict(size=12)
                ),
                tickfont=dict(size=10),
                thickness=20,
                len=0.8
            ),
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Messages: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Message Activity Heatmap',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            width=1000,
            height=500
        )
        
        return fig
    
    def create_response_time_3d(self):
        """
        Create 3D visualization of response times with corrected colorbar
        """
        # Calculate response times
        df_sorted = self.df.sort_values('datetime').reset_index(drop=True)
        response_data = []
        
        for i in range(1, len(df_sorted)):
            if df_sorted.iloc[i]['sender'] != df_sorted.iloc[i-1]['sender']:
                time_diff = (df_sorted.iloc[i]['datetime'] - df_sorted.iloc[i-1]['datetime']).total_seconds() / 60
                if time_diff < 1440:  # Less than 24 hours
                    response_data.append({
                        'hour': df_sorted.iloc[i]['hour'],
                        'weekday': df_sorted.iloc[i]['weekday'],
                        'response_time': time_diff,
                        'sender': df_sorted.iloc[i]['sender']
                    })
        
        if not response_data:
            return None
        
        resp_df = pd.DataFrame(response_data)
        
        # Create 3D scatter with corrected colorbar
        fig = go.Figure(data=[go.Scatter3d(
            x=resp_df['weekday'],
            y=resp_df['hour'],
            z=resp_df['response_time'],
            mode='markers',
            marker=dict(
                size=6,
                color=resp_df['response_time'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    # CORRECTED: Proper nested structure
                    title=dict(
                        text='Response<br>Time (min)',
                        font=dict(size=11)
                    ),
                    tickfont=dict(size=9),
                    thickness=15,
                    len=0.6
                )
            ),
            text=[f'Day: {d}<br>Hour: {h}<br>Response: {r:.1f} min' 
                  for d, h, r in zip(resp_df['weekday'], resp_df['hour'], resp_df['response_time'])],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Response Time Patterns (3D)',
            scene=dict(
                xaxis=dict(title='Day of Week', tickvals=[0,1,2,3,4,5,6]),
                yaxis=dict(title='Hour'),
                zaxis=dict(title='Response Time (minutes)')
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def create_comprehensive_dashboard(self):
        """
        Create comprehensive dashboard with multiple visualizations
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Message Volume', 'Time Period Distribution',
                          'Sender Activity', 'Weekend vs Weekday'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 1. Daily message volume
        daily_counts = self.df.groupby('date').size().reset_index(name='count')
        fig.add_trace(
            go.Scatter(x=daily_counts['date'], y=daily_counts['count'],
                      mode='lines+markers', name='Daily Messages'),
            row=1, col=1
        )
        
        # 2. Time period distribution
        time_counts = self.df['time_period'].value_counts()
        fig.add_trace(
            go.Bar(x=time_counts.index, y=time_counts.values, name='Time Periods'),
            row=1, col=2
        )
        
        # 3. Sender activity
        sender_counts = self.df['sender'].value_counts()
        fig.add_trace(
            go.Bar(x=sender_counts.index, y=sender_counts.values, name='By Sender'),
            row=2, col=1
        )
        
        # 4. Weekend vs Weekday
        weekend_counts = self.df.groupby('is_weekend').size()
        fig.add_trace(
            go.Bar(x=['Weekday', 'Weekend'], 
                   y=[weekend_counts.get(False, 0), weekend_counts.get(True, 0)],
                   name='Weekend Split'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text='Chat Analysis Dashboard',
            showlegend=False,
            height=800,
            width=1200
        )
        
        return fig
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        visualizations = {}
        
        try:
            visualizations['3d_activity'] = self.create_3d_activity_visualization()
        except Exception as e:
            print(f"Error creating 3D activity viz: {e}")
        
        try:
            visualizations['heatmap'] = self.create_heatmap_with_colorbar()
        except Exception as e:
            print(f"Error creating heatmap: {e}")
        
        try:
            visualizations['response_time_3d'] = self.create_response_time_3d()
        except Exception as e:
            print(f"Error creating response time viz: {e}")
        
        try:
            visualizations['dashboard'] = self.create_comprehensive_dashboard()
        except Exception as e:
            print(f"Error creating dashboard: {e}")
        
        return visualizations


# Example usage:
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_chat_data.csv')
    
    # Initialize analyzer
    # analyzer = ChatEDAWithPlotly(df)
    
    # Generate visualizations
    # visualizations = analyzer.generate_all_visualizations()
    
    # Show specific visualization
    # visualizations['3d_activity'].show()
    
    pass
