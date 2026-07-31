"""
Weekly Digest Bot - Day 12
Automates weekly chat analysis reports via email and Telegram bot.

This module integrates all analysis components to generate and send
automated weekly digests containing key metrics, insights, and visualizations.
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeeklyDigestBot:
    """
    Automated bot for generating and sending weekly chat analysis digests.
    Supports email and Telegram delivery methods.
    """
    
    def __init__(
        self,
        email_config: Optional[Dict[str, str]] = None,
        telegram_config: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Weekly Digest Bot.
        
        Args:
            email_config: Dict with keys: 'smtp_server', 'smtp_port', 'sender_email', 'sender_password'
            telegram_config: Dict with keys: 'bot_token', 'chat_id'
        """
        self.email_config = email_config or {}
        self.telegram_config = telegram_config or {}
        
    def generate_weekly_summary(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, any]:
        """
        Generate weekly summary statistics from chat data.
        
        Args:
            df: DataFrame with columns ['timestamp', 'sender', 'message', 'sentiment', etc.]
            start_date: Start date for analysis (defaults to 7 days ago)
            end_date: End date for analysis (defaults to today)
            
        Returns:
            Dictionary containing weekly metrics and insights
        """
        # Set default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)
            
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Filter data for the week
        weekly_df = df[
            (df['timestamp'] >= start_date) & 
            (df['timestamp'] <= end_date)
        ].copy()
        
        if weekly_df.empty:
            logger.warning("No data available for the specified date range")
            return self._get_empty_summary(start_date, end_date)
        
        # Calculate metrics
        summary = {
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            },
            'total_messages': len(weekly_df),
            'total_participants': weekly_df['sender'].nunique() if 'sender' in weekly_df.columns else 0,
            'most_active_day': self._get_most_active_day(weekly_df),
            'top_contributors': self._get_top_contributors(weekly_df, top_n=5),
            'message_distribution': self._get_message_distribution(weekly_df),
            'sentiment_summary': self._get_sentiment_summary(weekly_df),
            'activity_patterns': self._get_activity_patterns(weekly_df),
            'engagement_metrics': self._get_engagement_metrics(weekly_df)
        }
        
        return summary
    
    def _get_empty_summary(self, start_date: datetime, end_date: datetime) -> Dict:
        """Return empty summary structure when no data is available."""
        return {
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            },
            'total_messages': 0,
            'total_participants': 0,
            'most_active_day': 'N/A',
            'top_contributors': [],
            'message_distribution': {},
            'sentiment_summary': {},
            'activity_patterns': {},
            'engagement_metrics': {}
        }
    
    def _get_most_active_day(self, df: pd.DataFrame) -> str:
        """Identify the most active day of the week."""
        if 'timestamp' not in df.columns or df.empty:
            return 'N/A'
        
        day_counts = df['timestamp'].dt.day_name().value_counts()
        return day_counts.idxmax() if not day_counts.empty else 'N/A'
    
    def _get_top_contributors(self, df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Get top message contributors."""
        if 'sender' not in df.columns or df.empty:
            return []
        
        contributor_counts = df['sender'].value_counts().head(top_n)
        return [
            {'name': name, 'message_count': int(count)}
            for name, count in contributor_counts.items()
        ]
    
    def _get_message_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate daily message distribution."""
        if 'timestamp' not in df.columns or df.empty:
            return {}
        
        daily_counts = df.groupby(df['timestamp'].dt.date).size()
        return {
            str(date): int(count) 
            for date, count in daily_counts.items()
        }
    
    def _get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate sentiment distribution and trends."""
        if 'sentiment' not in df.columns or df.empty:
            return {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'average_score': 0.0
            }
        
        # Count sentiment categories
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        
        # Calculate average sentiment score if available
        avg_score = 0.0
        if 'sentiment_score' in df.columns:
            avg_score = df['sentiment_score'].mean()
        
        return {
            'positive': int(sentiment_counts.get('positive', 0)),
            'neutral': int(sentiment_counts.get('neutral', 0)),
            'negative': int(sentiment_counts.get('negative', 0)),
            'average_score': float(avg_score)
        }
    
    def _get_activity_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze activity patterns by hour and day."""
        if 'timestamp' not in df.columns or df.empty:
            return {'peak_hour': 'N/A', 'peak_day': 'N/A'}
        
        hourly_activity = df.groupby(df['timestamp'].dt.hour).size()
        daily_activity = df.groupby(df['timestamp'].dt.day_name()).size()
        
        return {
            'peak_hour': f"{hourly_activity.idxmax()}:00" if not hourly_activity.empty else 'N/A',
            'peak_day': daily_activity.idxmax() if not daily_activity.empty else 'N/A',
            'hourly_distribution': hourly_activity.to_dict()
        }
    
    def _get_engagement_metrics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate engagement and interaction metrics."""
        if df.empty:
            return {
                'avg_message_length': 0,
                'total_words': 0,
                'avg_response_time': 'N/A'
            }
        
        # Average message length
        avg_length = 0
        total_words = 0
        if 'message' in df.columns:
            df['message_length'] = df['message'].str.len()
            avg_length = df['message_length'].mean()
            total_words = df['message'].str.split().str.len().sum()
        
        return {
            'avg_message_length': round(float(avg_length), 2),
            'total_words': int(total_words),
            'avg_response_time': 'N/A'  # Can be calculated if response lag data is available
        }
    
    def format_digest_email(self, summary: Dict) -> str:
        """
        Format the weekly summary into an HTML email template.
        
        Args:
            summary: Weekly summary dictionary
            
        Returns:
            HTML string for email body
        """
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: #f7f9fc;
                    border-left: 4px solid #667eea;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                .metric-title {{
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 5px;
                }}
                .contributor {{
                    background: white;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .sentiment-bar {{
                    display: flex;
                    height: 30px;
                    border-radius: 5px;
                    overflow: hidden;
                    margin: 10px 0;
                }}
                .positive {{ background: #4caf50; }}
                .neutral {{ background: #ffc107; }}
                .negative {{ background: #f44336; }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 2px solid #eee;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Weekly Chat Digest</h1>
                <p>{summary['period']['start']} to {summary['period']['end']}</p>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">📈 Total Messages</div>
                <div style="font-size: 24px; font-weight: bold;">{summary['total_messages']:,}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">👥 Active Participants</div>
                <div style="font-size: 24px; font-weight: bold;">{summary['total_participants']}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">🔥 Most Active Day</div>
                <div style="font-size: 20px;">{summary['most_active_day']}</div>
            </div>
            
            <h2 style="color: #667eea; margin-top: 30px;">🏆 Top Contributors</h2>
            {''.join([f'''
            <div class="contributor">
                <strong>{contrib['name']}</strong>: {contrib['message_count']} messages
            </div>
            ''' for contrib in summary['top_contributors']])}
            
            <h2 style="color: #667eea; margin-top: 30px;">😊 Sentiment Analysis</h2>
            <div class="metric-card">
                <div class="sentiment-bar">
                    <div class="positive" style="width: {self._calculate_percentage(summary['sentiment_summary'].get('positive', 0), summary['total_messages'])}%"></div>
                    <div class="neutral" style="width: {self._calculate_percentage(summary['sentiment_summary'].get('neutral', 0), summary['total_messages'])}%"></div>
                    <div class="negative" style="width: {self._calculate_percentage(summary['sentiment_summary'].get('negative', 0), summary['total_messages'])}%"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <span>✅ Positive: {summary['sentiment_summary'].get('positive', 0)}</span>
                    <span>😐 Neutral: {summary['sentiment_summary'].get('neutral', 0)}</span>
                    <span>❌ Negative: {summary['sentiment_summary'].get('negative', 0)}</span>
                </div>
            </div>
            
            <h2 style="color: #667eea; margin-top: 30px;">⏰ Activity Patterns</h2>
            <div class="metric-card">
                <div><strong>Peak Hour:</strong> {summary['activity_patterns'].get('peak_hour', 'N/A')}</div>
                <div><strong>Peak Day:</strong> {summary['activity_patterns'].get('peak_day', 'N/A')}</div>
            </div>
            
            <h2 style="color: #667eea; margin-top: 30px;">💬 Engagement Metrics</h2>
            <div class="metric-card">
                <div><strong>Average Message Length:</strong> {summary['engagement_metrics'].get('avg_message_length', 0)} characters</div>
                <div><strong>Total Words:</strong> {summary['engagement_metrics'].get('total_words', 0):,}</div>
            </div>
            
            <div class="footer">
                <p>Generated by Chat Analyzer Pro 🚀</p>
                <p style="font-size: 12px;">Automated Weekly Digest | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        return html
    
    def _calculate_percentage(self, value: int, total: int) -> float:
        """Calculate percentage, handling division by zero."""
        return round((value / total * 100), 2) if total > 0 else 0
    
    def send_email_digest(
        self,
        summary: Dict,
        recipient_email: str,
        pdf_path: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """
        Send weekly digest via email.
        
        Args:
            summary: Weekly summary dictionary
            recipient_email: Recipient email address
            pdf_path: Optional path to PDF report to attach
            attachments: Optional list of additional file paths to attach
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.email_config:
            logger.error("Email configuration not provided")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Weekly Chat Digest: {summary['period']['start']} to {summary['period']['end']}"
            msg['From'] = self.email_config.get('sender_email')
            msg['To'] = recipient_email
            
            # Add HTML body
            html_body = self.format_digest_email(summary)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach PDF report if provided
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
                    pdf_attachment.add_header('Content-Disposition', 'attachment', filename='weekly_report.pdf')
                    msg.attach(pdf_attachment)
            
            # Attach additional files if provided
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            attachment = MIMEApplication(f.read())
                            attachment.add_header(
                                'Content-Disposition',
                                'attachment',
                                filename=os.path.basename(file_path)
                            )
                            msg.attach(attachment)
            
            # Send email
            with smtplib.SMTP(
                self.email_config.get('smtp_server'),
                int(self.email_config.get('smtp_port', 587))
            ) as server:
                server.starttls()
                server.login(
                    self.email_config.get('sender_email'),
                    self.email_config.get('sender_password')
                )
                server.send_message(msg)
            
            logger.info(f"Email digest sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email digest: {str(e)}")
            return False
    
    def send_telegram_digest(
        self,
        summary: Dict,
        chat_id: Optional[str] = None
    ) -> bool:
        """
        Send weekly digest via Telegram bot.
        
        Args:
            summary: Weekly summary dictionary
            chat_id: Optional Telegram chat ID (overrides config)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.telegram_config and not chat_id:
            logger.error("Telegram configuration not provided")
            return False
        
        try:
            import requests
            
            bot_token = self.telegram_config.get('bot_token')
            target_chat_id = chat_id or self.telegram_config.get('chat_id')
            
            if not bot_token or not target_chat_id:
                logger.error("Bot token or chat ID missing")
                return False
            
            # Format message
            message = self._format_telegram_message(summary)
            
            # Send via Telegram API
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': target_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Telegram digest sent successfully to chat {target_chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram digest: {str(e)}")
            return False
    
    def _format_telegram_message(self, summary: Dict) -> str:
        """Format summary as Telegram message with Markdown."""
        sentiment = summary['sentiment_summary']
        activity = summary['activity_patterns']
        engagement = summary['engagement_metrics']
        
        message = f"""
📊 *Weekly Chat Digest*
{summary['period']['start']} to {summary['period']['end']}

📈 *Key Metrics*
• Total Messages: *{summary['total_messages']:,}*
• Active Participants: *{summary['total_participants']}*
• Most Active Day: *{summary['most_active_day']}*

🏆 *Top Contributors*
"""
        
        for i, contrib in enumerate(summary['top_contributors'][:3], 1):
            message += f"{i}. {contrib['name']}: {contrib['message_count']} messages\n"
        
        message += f"""
😊 *Sentiment Breakdown*
✅ Positive: {sentiment.get('positive', 0)}
😐 Neutral: {sentiment.get('neutral', 0)}
❌ Negative: {sentiment.get('negative', 0)}

⏰ *Activity Patterns*
• Peak Hour: {activity.get('peak_hour', 'N/A')}
• Peak Day: {activity.get('peak_day', 'N/A')}

💬 *Engagement*
• Avg Message Length: {engagement.get('avg_message_length', 0)} chars
• Total Words: {engagement.get('total_words', 0):,}

_Generated by Chat Analyzer Pro 🚀_
"""
        
        return message
    
    def schedule_weekly_digest(
        self,
        df: pd.DataFrame,
        recipients: Dict[str, List[str]],
        pdf_generator_func: Optional[callable] = None
    ) -> Dict[str, bool]:
        """
        Generate and send weekly digests to all configured recipients.
        
        Args:
            df: Chat data DataFrame
            recipients: Dict with 'email' and 'telegram' keys containing recipient lists
            pdf_generator_func: Optional function to generate PDF report
            
        Returns:
            Dictionary with delivery status for each recipient
        """
        # Generate weekly summary
        summary = self.generate_weekly_summary(df)
        
        results = {}
        
        # Generate PDF if function provided
        pdf_path = None
        if pdf_generator_func:
            try:
                pdf_path = pdf_generator_func(df, summary)
            except Exception as e:
                logger.error(f"Failed to generate PDF: {str(e)}")
        
        # Send email digests
        if 'email' in recipients:
            for email in recipients['email']:
                success = self.send_email_digest(summary, email, pdf_path=pdf_path)
                results[f"email_{email}"] = success
        
        # Send Telegram digests
        if 'telegram' in recipients:
            for chat_id in recipients['telegram']:
                success = self.send_telegram_digest(summary, chat_id=chat_id)
                results[f"telegram_{chat_id}"] = success
        
        return results


# Utility functions for easy integration

def create_digest_bot(
    email_sender: Optional[str] = None,
    email_password: Optional[str] = None,
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    smtp_server: str = 'smtp.gmail.com',
    smtp_port: int = 587
) -> WeeklyDigestBot:
    """
    Factory function to create a configured WeeklyDigestBot instance.
    
    Args:
        email_sender: Sender email address
        email_password: Email password or app-specific password
        telegram_bot_token: Telegram bot API token
        telegram_chat_id: Default Telegram chat ID
        smtp_server: SMTP server address (default: Gmail)
        smtp_port: SMTP port (default: 587)
        
    Returns:
        Configured WeeklyDigestBot instance
    """
    email_config = None
    telegram_config = None
    
    if email_sender and email_password:
        email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'sender_email': email_sender,
            'sender_password': email_password
        }
    
    if telegram_bot_token:
        telegram_config = {
            'bot_token': telegram_bot_token,
            'chat_id': telegram_chat_id
        }
    
    return WeeklyDigestBot(email_config=email_config, telegram_config=telegram_config)


def send_quick_digest(
    df: pd.DataFrame,
    recipient_email: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    **bot_kwargs
) -> Dict[str, bool]:
    """
    Quick function to generate and send digest without explicit bot setup.
    
    Args:
        df: Chat data DataFrame
        recipient_email: Email address to send digest to
        telegram_chat_id: Telegram chat ID to send digest to
        **bot_kwargs: Additional arguments for create_digest_bot
        
    Returns:
        Dictionary with delivery status
    """
    bot = create_digest_bot(**bot_kwargs)
    summary = bot.generate_weekly_summary(df)
    
    results = {}
    
    if recipient_email and bot.email_config:
        results['email'] = bot.send_email_digest(summary, recipient_email)
    
    if telegram_chat_id and bot.telegram_config:
        results['telegram'] = bot.send_telegram_digest(summary, telegram_chat_id)
    
    return results


# Example usage for Google Colab
if __name__ == "__main__":
    """
    Example usage in Google Colab:
    
    # Load your chat data
    df = pd.read_csv('data/processed/example_parsed.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create bot instance
    bot = create_digest_bot(
        email_sender='your_email@gmail.com',
        email_password='your_app_password',  # Use app-specific password for Gmail
        telegram_bot_token='YOUR_BOT_TOKEN',
        telegram_chat_id='YOUR_CHAT_ID'
    )
    
    # Generate and send digest
    summary = bot.generate_weekly_summary(df)
    
    # Send via email
    bot.send_email_digest(summary, 'recipient@example.com')
    
    # Send via Telegram
    bot.send_telegram_digest(summary)
    
    # Or schedule for multiple recipients
    recipients = {
        'email': ['user1@example.com', 'user2@example.com'],
        'telegram': ['CHAT_ID_1', 'CHAT_ID_2']
    }
    results = bot.schedule_weekly_digest(df, recipients)
    print(results)
    """
    print("Weekly Digest Bot initialized. Import and use the functions above.")
