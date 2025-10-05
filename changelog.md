# Chat Analyzer Pro — 15-Day Plan & CHANGELOG

This document tracks the daily execution plan and repo evolution for Chat Analyzer Pro.

---

## Day-Wise Plan & Changelog

### **Day 1 — WhatsApp Parser**

**Tasks:**
- Parse WhatsApp `.txt` exports into structured DataFrame
- Handle date/time formats, sender names, message content
- Export to CSV for downstream analysis

**Deliverables:**
```
src/parser/whatsapp_parser.py
data/processed/example_parsed.csv
notebooks/01_data_parsing.ipynb
```

**Functions:**
- `parse_whatsapp_chat()` - Main parser
- `extract_datetime()` - Datetime extraction
- `clean_message_text()` - Text cleaning

---

### **Day 2 — Telegram Parser**

**Tasks:**
- Parse Telegram JSON exports
- Normalize schema to match WhatsApp format
- Unified DataFrame structure

**Deliverables:**
```
src/parser/telegram_parser.py
```

**Functions:**
- `parse_telegram_json()` - JSON parser
- `normalize_telegram_data()` - Schema normalization

---

### **Day 3 — EDA Module**

**Tasks:**
- Message volume analysis (daily, hourly, weekly)
- Top senders identification
- Activity patterns and peak hours
- Message length statistics

**Deliverables:**
```
src/analysis/eda.py
notebooks/02_exploratory_analysis.ipynb
```

**Functions:**
- `calculate_message_stats()`
- `analyze_activity_patterns()`
- `get_top_senders()`
- `plot_activity_heatmap()`

---

### **Day 4 — Sentiment Analysis**

**Tasks:**
- VADER sentiment scoring
- Hugging Face transformer models
- Sentiment timeline visualization
- Per-sender sentiment distribution

**Deliverables:**
```
src/analysis/sentiment.py
notebooks/03_sentiment_emotion.ipynb
```

**Functions:**
- `analyze_sentiment_vader()`
- `analyze_sentiment_transformers()`
- `plot_sentiment_timeline()`
- `calculate_sentiment_stats()`

---

### **Day 5 — Relationship Health Metrics**

**Tasks:**
- Conversation initiation ratio (who starts conversations)
- Response lag analysis (response times between participants)
- Dominance scores (message count, length, conversation control)
- Overall health scoring with weighted components

**Deliverables:**
```
src/analysis/relationship_health.py
notebooks/04_relationship_health.ipynb (Cells 1-8)
```

**Functions:**
- `identify_conversation_starters()` - Time gap based detection
- `calculate_initiator_ratio()` - Initiation balance
- `analyze_response_patterns()` - Response lag metrics
- `calculate_dominance_scores()` - Participation balance
- `calculate_relationship_health_score()` - Weighted scoring
- `analyze_relationship_health()` - Complete pipeline
- `plot_relationship_health_dashboard()` - 6-panel visualization

---

### **Day 6 — PDF Report Generator**

**Tasks:**
- Generate PDF reports with matplotlib charts
- Include all analysis sections (EDA, sentiment, health)
- Customizable templates

**Deliverables:**
```
src/reporting/pdf_report.py
notebooks/07_final_integration.ipynb
```

**Functions:**
- `generate_pdf_report()`
- `create_report_sections()`
- `embed_visualizations()`

---

### **Day 7 — Streamlit MVP**

**Tasks:**
- File upload interface (WhatsApp/Telegram)
- Interactive dashboards
- Real-time analysis
- PDF export button

**Deliverables:**
```
app/streamlit_app.py
app/assets/logo.png
app/assets/style.css
```

**Features:**
- Multi-page layout
- Session state management
- Caching for performance

---

### **Day 8 — Emotion Classification**

**Tasks:**
- Multi-class emotion detection (joy, sadness, anger, fear, surprise, disgust)
- Emotion timeline tracking
- Per-sender emotion profiles

**Deliverables:**
```
src/analysis/emotion.py
```

**Functions:**
- `classify_emotions()`
- `analyze_emotion_patterns()`
- `plot_emotion_distribution()`

---

### **Day 9 — Relationship Health Score Tracker**

**Tasks:**
- Rolling window health scores (configurable time windows)
- Trend analysis with linear regression
- Multi-period forecasting
- Component-level tracking over time
- Automated insights and recommendations

**Deliverables:**
```
src/analysis/relationship_health.py (extended with 5 new functions)
notebooks/04_relationship_health.ipynb (Cells 9-11 added)
```

**New Functions:**
- `calculate_rolling_health_scores(df, window_days)` - Time-windowed scoring
- `analyze_health_trend(rolling_health_df)` - Trend detection with scipy.linregress
- `forecast_health_scores(rolling_health_df, periods)` - Future predictions
- `generate_health_insights(rolling_health_df, df_prepared)` - Automated recommendations
- `plot_rolling_health_dashboard(rolling_health_df, trend_metrics, forecast_df)` - 7-panel visualization

**Notebook Cells:**
- Cell 9: Calculate rolling scores with configurable windows
- Cell 10: Visualize trends, components, activity, distribution, changes, summary
- Cell 11: Generate insights, alerts, recommendations, forecasts

---

### **Day 10 — Network Graphs (Group Chats)**

**Tasks:**
- Build directed interaction networks (who responds to whom)
- Calculate centrality metrics (degree, betweenness, PageRank)
- Identify key participants (most active, responsive, influential, bridge connectors)
- Interaction pattern analysis (reciprocity, strongest connections)
- Community detection (subgroup identification)
- Network visualizations (graph + comprehensive dashboard)

**Deliverables:**
```
src/analysis/network_graph.py
notebooks/05_network_graph.ipynb (6 cells)
```

**Functions:**
- `build_interaction_network(df, weight_threshold)` - Create NetworkX DiGraph
- `calculate_network_metrics(G)` - Centrality, density, connectivity
- `identify_key_participants(G, metrics)` - Role identification
- `analyze_interaction_patterns(df, G)` - Reciprocity, strongest links, interaction matrix
- `detect_subgroups(G)` - Community detection with modularity scoring
- `plot_network_graph(G, metrics, layout, node_size_metric)` - Single graph visualization
- `plot_network_dashboard(G, metrics, patterns)` - 5-panel comprehensive dashboard
- `analyze_network(df, weight_threshold)` - Complete pipeline wrapper

**Notebook Structure:**
- Cell 1: Environment setup, load Telegram group chat JSON
- Cell 2: Build network, show edges and weights
- Cell 3: Calculate metrics, identify key participants
- Cell 4: Analyze patterns, detect subgroups, visualize network
- Cell 5: Create comprehensive dashboard
- Cell 6: Run complete pipeline, generate summary

---

### **Day 11 — Conversation Summarizer**

**Tasks:**
- Implement T5-small transformer for summarization
- Conversation segmentation
- Extractive and abstractive summaries

**Deliverables:**
```
src/analysis/summarizer.py
notebooks/06_summarization.ipynb
```

---

### **Day 12 — Weekly Digest Bot**

**Tasks:**
- Automated weekly analysis reports
- Email integration
- Telegram bot notifications

**Deliverables:**
```
src/reporting/weekly_digest.py
```

---

### **Day 13 — Visualization Polish**

**Tasks:**
- Advanced heatmaps (hourly activity, sender interactions)
- Word clouds (per sender, overall)
- Enhanced plot styling

**Deliverables:**
```
src/utils/visualization.py
```

---

### **Day 14 — Gamification + Extras**

**Tasks:**
- Friendship index scoring
- Streak detection (consecutive days)
- Emoji usage analysis
- Response time leaderboards

**Deliverables:**
```
src/analysis/relationship_health.py (further extended)
src/utils/visualization.py (extended)
```

---

### **Day 15 — Final Integration & Deployment**

**Tasks:**
- Unit tests for all modules
- Integration tests
- Docker containerization
- Deployment configurations

**Deliverables:**
```
tests/test_parser.py
tests/test_analysis.py
tests/test_reporting.py
tests/test_end_to_end.py
deployment/Dockerfile
deployment/requirements.txt
deployment/Procfile
deployment/streamlit_config.toml
```

---

## Progress Summary

**Completed:** Days 1-10
**In Progress:** Day 11
**Remaining:** Days 12-15

**Current Repo Structure:**
- Parsers: WhatsApp, Telegram
- Analysis: EDA, Sentiment, Emotion, Relationship Health (with rolling tracker), Network Graphs
- Reporting: PDF generator
- App: Streamlit interface
- Notebooks: 5 complete (01-05)
