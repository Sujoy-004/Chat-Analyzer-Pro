"""
PDF Report Generator Module

This module generates comprehensive PDF reports containing chat analysis insights,
visualizations, and relationship health metrics. The report includes:
- Executive summary
- Chat statistics and trends
- Sentiment analysis charts
- Relationship health assessment
- Detailed insights and recommendations

Dependencies: reportlab, matplotlib, pandas, numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import io
import base64

# PDF generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


class ChatAnalysisPDFGenerator:
    """
    Generate comprehensive PDF reports for chat analysis results.
    """
    
    def __init__(self, output_filename: str = "chat_analysis_report.pdf", page_size=A4):
        """
        Initialize PDF generator.
        
        Args:
            output_filename: Name of output PDF file
            page_size: Page size (default: A4)
        """
        self.output_filename = output_filename
        self.page_size = page_size
        self.width, self.height = page_size
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self.story = []
        
    def setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2C3E50'),
            alignment=TA_CENTER
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#34495E'),
            alignment=TA_LEFT
        )
        
        # Section header style
        self.section_style = ParagraphStyle(
            'CustomSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.HexColor('#2980B9'),
            alignment=TA_LEFT
        )
        
        # Body text style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leftIndent=0,
            rightIndent=0
        )
        
        # Highlight style
        self.highlight_style = ParagraphStyle(
            'CustomHighlight',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            textColor=colors.HexColor('#E74C3C'),
            alignment=TA_LEFT,
            leftIndent=20
        )
        
        # Footer style
        self.footer_style = ParagraphStyle(
            'CustomFooter',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        )

    def add_title_page(self, analysis_data: Dict[str, Any]):
        """Add title page to the report."""
        # Title
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph("Chat Analysis Report", self.title_style))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Subtitle with date range
        conv_stats = analysis_data.get('conversation_stats', {})
        date_range = conv_stats.get('date_range', 'Unknown period')
        self.story.append(Paragraph(f"Analysis Period: {date_range}", self.subtitle_style))
        self.story.append(Spacer(1, 0.3*inch))
        
        # Executive summary box
        summary_data = [
            ['Total Messages', f"{conv_stats.get('total_messages', 'N/A'):,}"],
            ['Participants', str(conv_stats.get('unique_senders', 'N/A'))],
            ['Conversations', f"{conv_stats.get('total_conversations', 'N/A'):,}"],
            ['Avg Response Time', f"{conv_stats.get('avg_response_time', 0):.1f} minutes" if conv_stats.get('avg_response_time') else 'N/A'],
            ['Health Score', f"{analysis_data.get('health_score', {}).get('overall_health_score', 0):.2f}/1.00"],
            ['Grade', analysis_data.get('health_score', {}).get('grade', 'N/A')]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ECF0F1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        
        self.story.append(summary_table)
        self.story.append(Spacer(1, 1*inch))
        
        # Generated timestamp
        generated_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        self.story.append(Paragraph(f"Report generated on {generated_time}", self.footer_style))
        self.story.append(PageBreak())

    def matplotlib_to_reportlab_image(self, fig, width: float = 6*inch, height: float = 4*inch) -> Image:
        """
        Convert matplotlib figure to reportlab Image object.
        
        Args:
            fig: matplotlib figure object
            width: desired width in reportlab units
            height: desired height in reportlab units
            
        Returns:
            reportlab Image object
        """
        # Save figure to BytesIO
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        
        # Create reportlab Image
        img = Image(img_buffer, width=width, height=height)
        plt.close(fig)  # Close figure to free memory
        
        return img

    def create_health_score_chart(self, health_data: Dict[str, Any]) -> plt.Figure:
        """Create health score gauge chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        overall_score = health_data.get('overall_health_score', 0)
        grade = health_data.get('grade', 'N/A')
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        colors_list = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#27AE60']
        ranges = [0.0, 0.4, 0.6, 0.8, 0.9, 1.0]
        
        for i in range(len(colors_list)):
            start_angle = ranges[i] * np.pi
            end_angle = ranges[i+1] * np.pi
            angles = np.linspace(start_angle, end_angle, 20)
            x = np.cos(angles)
            y = np.sin(angles)
            ax.fill_between(x, 0, y, color=colors_list[i], alpha=0.8)
        
        # Add needle
        needle_angle = overall_score * np.pi
        needle_x = [0, 0.8 * np.cos(needle_angle)]
        needle_y = [0, 0.8 * np.sin(needle_angle)]
        ax.plot(needle_x, needle_y, 'k-', linewidth=4)
        ax.plot(needle_x[1], needle_y[1], 'ko', markersize=8)
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'Relationship Health Score: {overall_score:.2f}\nGrade: {grade}', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.text(0, -0.2, f'{overall_score:.2f}', ha='center', va='center', 
                fontsize=24, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig

    def create_component_breakdown_chart(self, health_data: Dict[str, Any]) -> plt.Figure:
        """Create component breakdown radar chart."""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        component_scores = health_data.get('component_scores', {})
        categories = ['Initiation\nBalance', 'Responsiveness', 'Response\nBalance', 'Participation\nBalance']
        values = [
            component_scores.get('initiation_balance', 0),
            component_scores.get('responsiveness', 0),
            component_scores.get('response_balance', 0),
            component_scores.get('dominance_balance', 0)
        ]
        
        # Close the plot
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, color='#3498DB')
        ax.fill(angles, values, alpha=0.25, color='#3498DB')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Health Score Components', fontsize=16, fontweight='bold', pad=30)
        ax.grid(True)
        
        # Add score labels
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.05, f'{value:.2f}', ha='center', va='center', 
                    fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        return fig

    def create_message_distribution_chart(self, dominance_data: Dict[str, Any]) -> plt.Figure:
        """Create message distribution pie chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        msg_dist = dominance_data.get('message_distribution', {})
        if msg_dist:
            colors_list = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38BA8', '#A8E6CF'][:len(msg_dist)]
            wedges, texts, autotexts = ax.pie(msg_dist.values(), labels=msg_dist.keys(), 
                                              autopct='%1.1f%%', colors=colors_list, 
                                              startangle=90, textprops={'fontsize': 12})
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_color('white')
            
            ax.set_title('Message Distribution', fontsize=16, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, 'No message distribution data available', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Message Distribution', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig

    def add_health_analysis_section(self, analysis_data: Dict[str, Any]):
        """Add relationship health analysis section."""
        self.story.append(Paragraph("Relationship Health Analysis", self.subtitle_style))
        
        health_data = analysis_data.get('health_score', {})
        
        # Overall assessment
        overall_score = health_data.get('overall_health_score', 0)
        grade = health_data.get('grade', 'N/A')
        description = health_data.get('description', 'No description available')
        
        self.story.append(Paragraph(f"<b>Overall Health Score:</b> {overall_score:.2f}/1.00 ({grade})", self.body_style))
        self.story.append(Paragraph(description, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Add health score gauge chart
        health_chart = self.create_health_score_chart(health_data)
        self.story.append(self.matplotlib_to_reportlab_image(health_chart, width=5*inch, height=3*inch))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Component breakdown
        self.story.append(Paragraph("Component Analysis", self.section_style))
        component_chart = self.create_component_breakdown_chart(health_data)
        self.story.append(self.matplotlib_to_reportlab_image(component_chart, width=4*inch, height=4*inch))
        
        # Strengths and improvements
        strengths = health_data.get('strengths', [])
        improvements = health_data.get('areas_for_improvement', [])
        
        if strengths:
            self.story.append(Paragraph("Relationship Strengths", self.section_style))
            for strength in strengths:
                self.story.append(Paragraph(strength, self.body_style))
        
        if improvements:
            self.story.append(Paragraph("Areas for Improvement", self.section_style))
            for improvement in improvements:
                self.story.append(Paragraph(improvement, self.highlight_style))
        
        self.story.append(PageBreak())

    def add_conversation_patterns_section(self, analysis_data: Dict[str, Any]):
        """Add conversation patterns analysis section."""
        self.story.append(Paragraph("Conversation Patterns", self.subtitle_style))
        
        # Conversation statistics
        conv_stats = analysis_data.get('conversation_stats', {})
        initiator_data = analysis_data.get('initiator_analysis', {})
        response_data = analysis_data.get('response_analysis', {})
        dominance_data = analysis_data.get('dominance_analysis', {})
        
        # Basic stats table
        stats_data = [
            ['Metric', 'Value'],
            ['Total Messages', f"{conv_stats.get('total_messages', 'N/A'):,}"],
            ['Total Conversations', f"{conv_stats.get('total_conversations', 'N/A'):,}"],
            ['Average Response Time', f"{conv_stats.get('avg_response_time', 0):.1f} minutes" if conv_stats.get('avg_response_time') else 'N/A'],
            ['Conversation Balance Score', f"{initiator_data.get('balance_score', 0):.2f}"],
            ['Response Balance Score', f"{response_data.get('response_balance_score', 0):.2f}"]
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        
        self.story.append(stats_table)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Message distribution chart
        dist_chart = self.create_message_distribution_chart(dominance_data)
        self.story.append(self.matplotlib_to_reportlab_image(dist_chart, width=5*inch, height=3.5*inch))
        
        # Conversation initiation analysis
        self.story.append(Paragraph("Conversation Initiation", self.section_style))
        
        initiator_counts = initiator_data.get('initiator_counts', {})
        balance_score = initiator_data.get('balance_score', 0)
        interpretation = initiator_data.get('interpretation', 'No analysis available')
        
        if initiator_counts:
            init_text = "Conversation starters: " + ", ".join([f"{name}: {count}" for name, count in initiator_counts.items()])
            self.story.append(Paragraph(init_text, self.body_style))
        
        self.story.append(Paragraph(f"<b>Balance Score:</b> {balance_score:.2f}", self.body_style))
        self.story.append(Paragraph(f"<b>Assessment:</b> {interpretation}", self.body_style))
        
        self.story.append(PageBreak())

    def add_detailed_metrics_section(self, analysis_data: Dict[str, Any]):
        """Add detailed metrics section."""
        self.story.append(Paragraph("Detailed Analysis Metrics", self.subtitle_style))
        
        # Response analysis
        response_data = analysis_data.get('response_analysis', {})
        if 'response_stats' in response_data and response_data['response_stats']:
            self.story.append(Paragraph("Response Time Statistics", self.section_style))
            
            response_stats = response_data['response_stats']
            if 'mean' in response_stats:
                for person, avg_time in response_stats['mean'].items():
                    self.story.append(Paragraph(f"<b>{person}:</b> {avg_time:.1f} minutes average response time", self.body_style))
            
            total_responses = response_data.get('total_responses_analyzed', 0)
            overall_avg = response_data.get('overall_avg_response_minutes', 0)
            responsiveness_score = response_data.get('responsiveness_score', 0)
            
            self.story.append(Paragraph(f"Total responses analyzed: {total_responses:,}", self.body_style))
            self.story.append(Paragraph(f"Overall average response time: {overall_avg:.1f} minutes", self.body_style))
            self.story.append(Paragraph(f"Responsiveness score: {responsiveness_score:.2f}/1.00", self.body_style))
        
        # Dominance analysis
        dominance_data = analysis_data.get('dominance_analysis', {})
        self.story.append(Paragraph("Participation Balance", self.section_style))
        
        composite_score = dominance_data.get('composite_dominance_score', 0)
        interpretation = dominance_data.get('interpretation', 'No analysis available')
        
        self.story.append(Paragraph(f"<b>Composite Balance Score:</b> {composite_score:.2f}/1.00", self.body_style))
        self.story.append(Paragraph(f"<b>Assessment:</b> {interpretation}", self.body_style))
        
        # Component scores
        msg_balance = dominance_data.get('message_count_balance', 0)
        length_balance = dominance_data.get('message_length_balance', 0)
        control_balance = dominance_data.get('conversation_control_balance', 0)
        
        balance_data = [
            ['Balance Component', 'Score'],
            ['Message Count Balance', f"{msg_balance:.2f}"],
            ['Message Length Balance', f"{length_balance:.2f}"],
            ['Conversation Control Balance', f"{control_balance:.2f}"]
        ]
        
        balance_table = Table(balance_data, colWidths=[3*inch, 2*inch])
        balance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FADBD8')])
        ]))
        
        self.story.append(balance_table)
        self.story.append(Spacer(1, 0.3*inch))

    def add_recommendations_section(self, analysis_data: Dict[str, Any]):
        """Add recommendations and insights section."""
        self.story.append(Paragraph("Recommendations & Insights", self.subtitle_style))
        
        health_data = analysis_data.get('health_score', {})
        overall_score = health_data.get('overall_health_score', 0)
        
        # Generate personalized recommendations
        recommendations = []
        
        # Check initiation balance
        init_data = analysis_data.get('initiator_analysis', {})
        init_balance = init_data.get('balance_score', 0)
        if init_balance < 0.6:
            recommendations.append("Consider taking turns initiating conversations to create better balance.")
        
        # Check response times
        response_data = analysis_data.get('response_analysis', {})
        responsiveness = response_data.get('responsiveness_score', 0)
        if responsiveness < 0.6:
            recommendations.append("Try to respond more promptly to maintain engagement and show interest.")
        
        # Check dominance
        dominance_data = analysis_data.get('dominance_analysis', {})
        dominance_score = dominance_data.get('composite_dominance_score', 0)
        if dominance_score < 0.6:
            recommendations.append("Ensure both participants have equal opportunities to contribute to conversations.")
        
        # General recommendations based on overall score
        if overall_score >= 0.9:
            recommendations.append("Excellent communication! Keep maintaining this healthy balance.")
        elif overall_score >= 0.7:
            recommendations.append("Good communication patterns with room for minor improvements.")
        else:
            recommendations.append("Focus on creating more balanced communication patterns.")
        
        # Add default recommendations if none generated
        if not recommendations:
            recommendations = [
                "Continue maintaining healthy communication patterns.",
                "Stay responsive and engaged in conversations.",
                "Keep initiating conversations when appropriate."
            ]
        
        self.story.append(Paragraph("Based on your chat analysis, here are some personalized recommendations:", self.body_style))
        
        for i, rec in enumerate(recommendations, 1):
            self.story.append(Paragraph(f"<b>{i}.</b> {rec}", self.body_style))
        
        # Methodology note
        self.story.append(Spacer(1, 0.3*inch))
        self.story.append(Paragraph("Analysis Methodology", self.section_style))
        methodology_text = """
        This analysis uses advanced algorithms to evaluate conversation patterns, response times, 
        initiation balance, and participation equity. The relationship health score combines multiple 
        metrics weighted by their importance to relationship dynamics. Scores range from 0.0 to 1.0, 
        with higher scores indicating healthier communication patterns.
        """
        self.story.append(Paragraph(methodology_text, self.body_style))

    def generate_report(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate complete PDF report.
        
        Args:
            analysis_data: Complete analysis results from relationship health analysis
            
        Returns:
            Path to generated PDF file
        """
        # Create PDF document
        doc = SimpleDocTemplate(
            self.output_filename,
            pagesize=self.page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story
        self.story = []  # Reset story
        
        # Add sections
        self.add_title_page(analysis_data)
        self.add_health_analysis_section(analysis_data)
        self.add_conversation_patterns_section(analysis_data)
        self.add_detailed_metrics_section(analysis_data)
        self.add_recommendations_section(analysis_data)
        
        # Build PDF
        doc.build(self.story)
        
        return self.output_filename


def generate_chat_analysis_pdf(
    analysis_results: Dict[str, Any], 
    output_filename: str = "chat_analysis_report.pdf"
) -> str:
    """
    Main function to generate PDF report from analysis results.
    
    Args:
        analysis_results: Complete analysis results from analyze_relationship_health()
        output_filename: Output PDF filename
        
    Returns:
        Path to generated PDF file
    """
    generator = ChatAnalysisPDFGenerator(output_filename)
    return generator.generate_report(analysis_results)


# Example usage
def example_pdf_generation():
    """
    Example of how to generate a PDF report.
    This would typically use real analysis results.
    """
    # Sample analysis data structure
    sample_results = {
        'conversation_stats': {
            'total_messages': 1543,
            'unique_senders': 2,
            'date_range': '2023-12-01 to 2024-01-15',
            'total_conversations': 87,
            'avg_response_time': 15.3
        },
        'health_score': {
            'overall_health_score': 0.85,
            'grade': 'VERY GOOD',
            'description': 'Strong relationship with good communication balance and responsiveness',
            'component_scores': {
                'initiation_balance': 0.92,
                'responsiveness': 0.78,
                'response_balance': 0.85,
                'dominance_balance': 0.88
            },
            'strengths': [
                '✅ Balanced conversation initiation',
                '✅ Similar response time patterns',
                '✅ Excellent participation balance'
            ],
            'areas_for_improvement': [
                '⚠️ Could improve response speed slightly'
            ]
        },
        'initiator_analysis': {
            'initiator_counts': {'Alice': 45, 'Bob': 42},
            'balance_score': 0.92,
            'interpretation': 'Excellent balance - both participants initiate conversations equally'
        },
        'response_analysis': {
            'response_stats': {
                'mean': {'Alice': 12.5, 'Bob': 18.2}
            },
            'total_responses_analyzed': 156,
            'overall_avg_response_minutes': 15.3,
            'responsiveness_score': 0.78,
            'response_balance_score': 0.85
        },
        'dominance_analysis': {
            'message_distribution': {'Alice': 789, 'Bob': 754},
            'composite_dominance_score': 0.88,
            'interpretation': 'Good balance - minor differences in participation',
            'message_count_balance': 0.91,
            'message_length_balance': 0.85,
            'conversation_control_balance': 0.89
        }
    }
    
    # Generate PDF
    pdf_path = generate_chat_analysis_pdf(sample_results, "example_chat_report.pdf")
    print(f"PDF report generated: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    # Run example if script is executed directly
    example_pdf_generation()
