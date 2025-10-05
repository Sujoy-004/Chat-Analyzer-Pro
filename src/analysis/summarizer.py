"""
Conversation Summarizer Module
Day 11 - Chat Analyzer Pro

This module provides conversation summarization capabilities using T5-small model.
It can summarize individual conversations or generate periodic summaries.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')


class ConversationSummarizer:
    """
    A class to summarize conversations using T5-small transformer model.
    
    Features:
    - Summarize entire conversations
    - Summarize conversations by date range
    - Summarize conversations by participant
    - Generate extractive and abstractive summaries
    """
    
    def __init__(self, model_name: str = "t5-small", max_length: int = 150, min_length: int = 40):
        """
        Initialize the ConversationSummarizer with T5 model.
        
        Args:
            model_name (str): Hugging Face model name (default: t5-small)
            max_length (int): Maximum length of summary
            min_length (int): Minimum length of summary
        """
        print(f"Loading {model_name} model...")
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        
        # Initialize T5 model and tokenizer
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer
            )
            print(f"✅ {model_name} loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    
    def _prepare_text(self, df: pd.DataFrame, max_messages: int = 100) -> str:
        """
        Prepare conversation text from DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with 'sender' and 'message' columns
            max_messages (int): Maximum number of messages to include
            
        Returns:
            str: Formatted conversation text
        """
        # Limit number of messages to avoid token overflow
        df_subset = df.head(max_messages) if len(df) > max_messages else df
        
        # Format as conversation
        conversation_lines = []
        for _, row in df_subset.iterrows():
            sender = row.get('sender', 'Unknown')
            message = row.get('message', '')
            
            # Skip empty messages or system messages
            if pd.isna(message) or message.strip() == '' or '<Media omitted>' in message:
                continue
                
            conversation_lines.append(f"{sender}: {message}")
        
        conversation_text = " ".join(conversation_lines)
        
        # Truncate if too long (T5-small has 512 token limit)
        if len(conversation_text) > 2000:
            conversation_text = conversation_text[:2000]
        
        return conversation_text
    
    
    def summarize_conversation(
        self, 
        df: pd.DataFrame, 
        max_messages: int = 100,
        custom_max_length: Optional[int] = None,
        custom_min_length: Optional[int] = None
    ) -> Dict[str, Union[str, int]]:
        """
        Summarize an entire conversation.
        
        Args:
            df (pd.DataFrame): Conversation DataFrame
            max_messages (int): Maximum messages to consider
            custom_max_length (int): Override default max_length
            custom_min_length (int): Override default min_length
            
        Returns:
            Dict: Summary results with metadata
        """
        if df.empty:
            return {
                'summary': 'No messages to summarize.',
                'total_messages': 0,
                'messages_summarized': 0
            }
        
        # Prepare text
        conversation_text = self._prepare_text(df, max_messages)
        
        if not conversation_text.strip():
            return {
                'summary': 'No valid messages found to summarize.',
                'total_messages': len(df),
                'messages_summarized': 0
            }
        
        # Generate summary
        try:
            max_len = custom_max_length if custom_max_length else self.max_length
            min_len = custom_min_length if custom_min_length else self.min_length
            
            summary_result = self.summarizer(
                conversation_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            
            summary_text = summary_result[0]['summary_text']
            
            return {
                'summary': summary_text,
                'total_messages': len(df),
                'messages_summarized': min(len(df), max_messages),
                'model_used': self.model_name
            }
            
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            return {
                'summary': f'Error generating summary: {str(e)}',
                'total_messages': len(df),
                'messages_summarized': 0
            }
    
    
    def summarize_by_date_range(
        self, 
        df: pd.DataFrame, 
        start_date: str, 
        end_date: str,
        date_column: str = 'date'
    ) -> Dict[str, Union[str, int]]:
        """
        Summarize conversation within a specific date range.
        
        Args:
            df (pd.DataFrame): Conversation DataFrame
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            date_column (str): Name of date column
            
        Returns:
            Dict: Summary results
        """
        # Ensure date column is datetime
        if date_column not in df.columns:
            return {
                'summary': f'Date column "{date_column}" not found in DataFrame.',
                'total_messages': 0,
                'messages_summarized': 0,
                'date_range': f'{start_date} to {end_date}'
            }
        
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Filter by date range
        mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
        filtered_df = df.loc[mask]
        
        if filtered_df.empty:
            return {
                'summary': f'No messages found between {start_date} and {end_date}.',
                'total_messages': 0,
                'messages_summarized': 0,
                'date_range': f'{start_date} to {end_date}'
            }
        
        # Generate summary
        result = self.summarize_conversation(filtered_df)
        result['date_range'] = f'{start_date} to {end_date}'
        
        return result
    
    
    def summarize_by_participant(
        self, 
        df: pd.DataFrame, 
        participant: str,
        sender_column: str = 'sender'
    ) -> Dict[str, Union[str, int]]:
        """
        Summarize messages from a specific participant.
        
        Args:
            df (pd.DataFrame): Conversation DataFrame
            participant (str): Name of participant
            sender_column (str): Name of sender column
            
        Returns:
            Dict: Summary results
        """
        # Filter by participant
        participant_df = df[df[sender_column] == participant]
        
        if participant_df.empty:
            return {
                'summary': f'No messages found from {participant}.',
                'total_messages': 0,
                'messages_summarized': 0,
                'participant': participant
            }
        
        # Generate summary
        result = self.summarize_conversation(participant_df)
        result['participant'] = participant
        
        return result
    
    
    def generate_periodic_summaries(
        self, 
        df: pd.DataFrame, 
        period: str = 'W',
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Generate summaries for periodic intervals (daily, weekly, monthly).
        
        Args:
            df (pd.DataFrame): Conversation DataFrame
            period (str): Pandas period string ('D'=daily, 'W'=weekly, 'M'=monthly)
            date_column (str): Name of date column
            
        Returns:
            pd.DataFrame: DataFrame with period and summary columns
        """
        if date_column not in df.columns:
            print(f"❌ Date column '{date_column}' not found.")
            return pd.DataFrame()
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        
        # Group by period
        df['period'] = df[date_column].dt.to_period(period)
        
        summaries = []
        for period_val, group in df.groupby('period'):
            result = self.summarize_conversation(group, max_messages=50)
            summaries.append({
                'period': str(period_val),
                'start_date': group[date_column].min(),
                'end_date': group[date_column].max(),
                'summary': result['summary'],
                'message_count': result['total_messages']
            })
        
        return pd.DataFrame(summaries)
    
    
    def get_key_topics(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        """
        Extract key topics/words from conversation (simple frequency-based).
        
        Args:
            df (pd.DataFrame): Conversation DataFrame
            top_n (int): Number of top topics to return
            
        Returns:
            List[str]: Top n words/topics
        """
        from collections import Counter
        import re
        
        # Combine all messages
        all_messages = ' '.join(df['message'].dropna().astype(str))
        
        # Simple tokenization and cleaning
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_messages.lower())
        
        # Remove common stop words
        stop_words = {'that', 'this', 'with', 'have', 'from', 'they', 'been', 
                      'were', 'said', 'each', 'which', 'their', 'about', 'than',
                      'there', 'would', 'these', 'other', 'into', 'more', 'some'}
        
        words = [w for w in words if w not in stop_words]
        
        # Get most common
        word_counts = Counter(words)
        top_words = [word for word, count in word_counts.most_common(top_n)]
        
        return top_words
    
    
    def generate_full_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive summary report.
        
        Args:
            df (pd.DataFrame): Conversation DataFrame
            
        Returns:
            Dict: Complete summary report
        """
        print("🔄 Generating comprehensive summary report...")
        
        # Overall summary
        overall = self.summarize_conversation(df, max_messages=150)
        
        # Key topics
        topics = self.get_key_topics(df, top_n=10)
        
        # Participant summaries
        participants = df['sender'].unique()[:5]  # Limit to top 5
        participant_summaries = {}
        for participant in participants:
            p_summary = self.summarize_by_participant(df, participant)
            participant_summaries[participant] = p_summary['summary']
        
        report = {
            'overall_summary': overall,
            'key_topics': topics,
            'participant_summaries': participant_summaries,
            'total_participants': len(df['sender'].unique()),
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else 'N/A',
                'end': df['date'].max() if 'date' in df.columns else 'N/A'
            }
        }
        
        print("✅ Report generation complete!")
        return report


# Utility function for quick summarization
def quick_summarize(df: pd.DataFrame, max_length: int = 150) -> str:
    """
    Quick one-line summarization function.
    
    Args:
        df (pd.DataFrame): Conversation DataFrame
        max_length (int): Maximum summary length
        
    Returns:
        str: Summary text
    """
    summarizer = ConversationSummarizer(max_length=max_length)
    result = summarizer.summarize_conversation(df)
    return result['summary']
