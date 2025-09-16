chat-analyzer-pro/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── .gitignore

├── data/
│   ├── sample_chats/
│   │   ├── whatsapp_sample.txt
│   │   ├── telegram_sample.json
│   └── processed/
│       └── example_parsed.csv

├── notebooks/
│   ├── 01_data_parsing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_sentiment_emotion.ipynb
│   ├── 04_relationship_health.ipynb
│   ├── 05_network_graph.ipynb
│   ├── 06_summarization.ipynb
│   └── 07_final_integration.ipynb

├── src/
│   ├── __init__.py
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── whatsapp_parser.py
│   │   └── telegram_parser.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py
│   │   ├── sentiment.py
│   │   ├── emotion.py
│   │   ├── relationship_health.py
│   │   ├── summarizer.py
│   │   └── network_graph.py
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── pdf_report.py
│   │   └── weekly_digest.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py
│       └── visualization.py

├── app/
│   ├── streamlit_app.py
│   └── assets/
│       ├── logo.png
│       └── style.css

├── tests/
│   ├── test_parser.py
│   ├── test_analysis.py
│   ├── test_reporting.py
│   └── test_end_to_end.py

└── deployment/
    ├── Dockerfile
    ├── requirements.txt
    ├── Procfile
    └── streamlit_config.toml
