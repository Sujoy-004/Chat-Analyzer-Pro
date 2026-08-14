"""
emotion.py - Emotion Classification Module
Chat Analyzer Pro - Day 8 Implementation

This module provides advanced emotion classification using HuggingFace transformers.
Classifies messages into 6 emotions: Joy, Sadness, Anger, Fear, Surprise, Love

Dependencies:
    - transformers>=4.30.0
    - torch>=2.0.0
    - pandas>=1.5.0
    - numpy>=1.24.0

Usage:
    from chat_analyzer.analysis.emotion import EmotionAnalyzer
    
    analyzer = EmotionAnalyzer()
    df_with_emotions = analyzer.analyze_emotions(df)
    summary = analyzer.get_emotion_summary(df_with_emotions)

Sampled path (Option C — large chats):
    analyze_emotions(df, sample_cap=50000) caps how many messages the
    DistilBERT model actually scores. When more scorable messages exist than
    the cap, a DETERMINISTIC stratified sample is scored instead: seats are
    allocated across senders proportionally (largest-remainder), then across
    ~50 time buckets per sender, with random_state=42 draws everywhere — the
    same frame + cap always yields the same sample, so sampled results are
    reproducible across runs. Non-selected rows keep the neutral 1/6 scores,
    the returned frame gains an `emotion_scored` boolean column and a
    `df.attrs["emotion_sample"]` metadata dict, and dominant_emotion /
    emotion_confidence are still computed over the full frame. The pipeline
    computes its summary over the scored sample rows only and labels the
    report/terminal "based on a sample of N of M messages". Below the cap
    (or with sample_cap=None) behavior is byte-for-byte the exact path.
"""

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
import matplotlib  # binds the name for the emotion_figure return annotation

# Global analyzer instance for reuse
_emotion_analyzer = None
_emotion_model_loaded = False

# Option C sampled path: fixed RNG seed (determinism across runs) and the
# number of time buckets used to stratify each sender's rows.
_EMOTION_SAMPLE_RNG = 42
_TIME_BUCKETS = 50


def _allocate_seats(counts: dict, cap: int) -> dict:
    """Deterministic largest-remainder allocation of ``cap`` seats.

    Each group with > 0 messages keeps at least one seat whenever the cap
    permits; leftover seats are refilled round-robin to the groups with the
    largest fractional remainder, tie-broken by group name. Allocations never
    exceed a group's own count and always sum to ``cap`` — a pure function of
    (counts, cap), so the sample is reproducible.
    """
    total = sum(counts.values())
    if total <= 0 or cap <= 0:
        return {}
    alloc = {g: min(counts[g], max(1, (counts[g] * cap) // total)) for g in counts}
    if sum(alloc.values()) > cap:
        # more groups than seats — drop the per-group floor entirely
        alloc = {g: (counts[g] * cap) // total for g in counts}
    seats = cap - sum(alloc.values())
    order = sorted(counts, key=lambda g: (-((counts[g] * cap) % total), g))
    i = 0
    while seats > 0:
        g = order[i % len(order)]
        if alloc[g] < counts[g]:
            alloc[g] += 1
            seats -= 1
        i += 1
    return alloc


def _global_time_edges(df: pd.DataFrame, scorable_mask: pd.Series):
    """~50 evenly-sized time edges across ALL scorable rows (bucket fallback).

    Returns the bin edges from a global qcut (deterministic) or None when the
    edges cannot be computed — callers then fall back to a plain
    random_state=42 sample. Used only when a sender's own qcut fails.
    """
    try:
        times = pd.to_datetime(df.loc[scorable_mask, "datetime"]).astype("int64")
        _, edges = pd.qcut(
            times,
            q=min(_TIME_BUCKETS, len(times)),
            retbins=True,
            duplicates="drop",
        )
        return edges
    except Exception:  # noqa: BLE001 - bucket-edge failure degrades to the plain-sample fallback
        return None


def _pick_in_sender(
    df: pd.DataFrame,
    sender_mask: pd.Series,
    seats: int,
    global_edges=None,
) -> list:
    """Pick ``seats`` rows from one sender's scorable rows (``sender_mask``).

    Stratifies across ~50 time buckets (qcut over the sender's own datetime
    positions, falling back to the global edges) and draws deterministically
    (random_state=42) within every bucket. Without a usable datetime column it
    degrades to a plain deterministic sample. Always returns exactly
    ``seats`` row labels.
    """
    labels = df.index[sender_mask]
    if len(labels) <= seats:
        return labels.tolist()
    if "datetime" not in df.columns:
        drawn = labels.to_series().sample(n=seats, random_state=_EMOTION_SAMPLE_RNG)
        return drawn.tolist()
    try:
        times = pd.to_datetime(df.loc[sender_mask, "datetime"]).astype("int64")
        buckets = pd.Series(
            pd.qcut(times, q=min(_TIME_BUCKETS, len(times)), duplicates="drop"),
            index=labels,
        )
    except Exception:  # noqa: BLE001 - per-sender qcut failure falls back to global edges / plain sample
        if global_edges is not None and len(global_edges) >= 2:
            try:
                times = pd.to_datetime(df.loc[sender_mask, "datetime"]).astype("int64")
                buckets = pd.Series(pd.cut(times, bins=global_edges), index=labels)
            except Exception:  # noqa: BLE001 - global-edge cut failure degrades to a plain sample
                drawn = labels.to_series().sample(
                    n=seats, random_state=_EMOTION_SAMPLE_RNG
                )
                return drawn.tolist()
        else:
            drawn = labels.to_series().sample(n=seats, random_state=_EMOTION_SAMPLE_RNG)
            return drawn.tolist()

    counts = buckets.value_counts().to_dict()
    alloc = _allocate_seats(counts, seats)
    picked: list = []
    for bucket, n_seats in alloc.items():
        bucket_labels = labels[buckets == bucket]
        if n_seats >= len(bucket_labels):
            picked.extend(bucket_labels.tolist())
        else:
            drawn = bucket_labels.to_series().sample(
                n=n_seats, random_state=_EMOTION_SAMPLE_RNG
            )
            picked.extend(drawn.tolist())
    return picked


def _stratified_sample_indices(
    df: pd.DataFrame,
    scorable_mask: pd.Series,
    cap: int,
) -> pd.Series:
    """Deterministic (participant, time)-stratified selection of scorable rows.

    Returns a boolean Series aligned to df.index with exactly
    min(cap, n_scorable) True entries, all on scorable rows. Pure function of
    (df, cap): identical inputs always produce the identical selection
    (random_state=42 + largest-remainder seat allocation), so sampled emotion
    scores are reproducible across runs. Original row order is preserved (the
    mask never reorders the frame).
    """
    chosen = scorable_mask.copy()
    if int(scorable_mask.sum()) <= cap:
        return chosen
    chosen[:] = False
    if "sender" not in df.columns:
        drawn = (
            df.index[scorable_mask]
            .to_series()
            .sample(n=cap, random_state=_EMOTION_SAMPLE_RNG)
        )
        chosen[drawn.to_numpy()] = True
        return chosen

    counts = df.loc[scorable_mask, "sender"].value_counts().to_dict()
    alloc = _allocate_seats(counts, cap)
    global_edges = _global_time_edges(df, scorable_mask) if "datetime" in df.columns else None
    for sender, seats in alloc.items():
        sender_mask = scorable_mask & (df["sender"] == sender)
        picked = _pick_in_sender(df, sender_mask, seats, global_edges)
        chosen[picked] = True
    return chosen


class EmotionAnalyzer:
    """
    Advanced emotion classification using HuggingFace transformers.
    Optimized for batch processing and cloud deployment.
    """
    
    def __init__(self, model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"):
        """
        Initialize the emotion analyzer.

        Args:
            model_name: HuggingFace model identifier for emotion classification
        """
        self.model_name = model_name
        self.pipeline = None
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the emotion classification model."""
        global _emotion_analyzer, _emotion_model_loaded
        
        if _emotion_model_loaded and _emotion_analyzer is not None:
            self.pipeline = _emotion_analyzer
            print("✅ Using cached emotion model")
            return
        
        try:
            from transformers import pipeline
            print(f"🚀 Loading emotion classification model: {self.model_name}")
            print("   This may take a moment on first run...")
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,  # Return all emotion scores
                device=-1  # CPU (use 0 for GPU if available)
            )
            
            _emotion_analyzer = self.pipeline
            _emotion_model_loaded = True
            print("✅ Emotion model loaded successfully!")
            
        except Exception as e:  # noqa: BLE001 - model-load failure must degrade to rule-based fallback, never crash (D-17)
            print(f"❌ Error loading emotion model: {e}")
            print("   Falling back to rule-based emotion detection...")
            self.pipeline = None
    
    def analyze_single_message(self, text: str) -> dict[str, float]:
        """
        Analyze emotion in a single message.
        
        Args:
            text: Message text to analyze
            
        Returns:
            Dictionary with emotion scores
        """
        # Handle empty or invalid messages
        if not text or not isinstance(text, str) or text.strip() == "":
            return self._get_neutral_emotions()
        
        # Skip media messages
        if "<Media omitted>" in text or "<media omitted>" in text.lower():
            return self._get_neutral_emotions()
        
        # Skip very short messages (likely just emojis or punctuation)
        if len(text.strip()) < 3:
            return self._get_neutral_emotions()
        
        try:
            if self.pipeline is not None:
                # Limit to 512 chars for efficiency
                return self._parse_pipeline_result(self.pipeline(text[:512]))
            else:
                # Fallback to rule-based detection
                return self._rule_based_emotion(text)
                
        except Exception as e:  # noqa: BLE001 - per-message scoring failure degrades to neutral scores, never crashes the batch
            print(f"⚠️ Error analyzing message: {e}")
            return self._get_neutral_emotions()

    def _parse_pipeline_result(self, res) -> dict[str, float]:
        """Normalize a transformers per-message result to our standard score dict.

        transformers 4.x with top_k=None returns a FLAT list of
        {"label":.., "score":..} dicts for every class, while 5.x nests it
        one level deeper: [[{"label":..,"score":..}, ...]]. Normalize both
        shapes so real scores surface (C-… 5.x compat).
        """
        if res and isinstance(res[0], list):
            res = res[0]

        # Convert to our standard format
        emotion_scores = {item['label']: float(item['score']) for item in res}

        # Ensure all 6 emotions are present
        for emotion in self.emotions:
            emotion_scores.setdefault(emotion, 0.0)

        return emotion_scores

    def _is_scorable(self, text) -> bool:
        """Whether a message would be scored instead of short-circuiting to neutral."""
        if not text or not isinstance(text, str) or text.strip() == "":
            return False
        if "<Media omitted>" in text or "<media omitted>" in text.lower():
            return False
        return len(text.strip()) >= 3

    def _score_batch(self, texts: list[str], batch_size: int) -> list[dict[str, float]]:
        """Score texts in chunks of batch_size, degrading per-message on any
        batched-call or parse failure so results stay identical to sequential."""
        scored = []
        size = max(1, batch_size)
        for i in range(0, len(texts), size):
            chunk = texts[i:i + size]
            try:
                batch_out = self.pipeline(chunk, batch_size=size, top_k=None)
                if not isinstance(batch_out, list):
                    raise TypeError("pipeline returned a non-list result")
                if len(batch_out) == 1 and isinstance(batch_out[0], list) and len(batch_out[0]) == len(chunk):
                    batch_out = batch_out[0]
                if len(batch_out) != len(chunk):
                    raise ValueError("pipeline returned a non-aligned result shape")
                chunk_scores = [self._parse_pipeline_result(item) for item in batch_out]
            except Exception as e:  # noqa: BLE001 - batched failure degrades to per-message scoring, never crashes the batch
                print(f"⚠️ Batch failed ({e}), falling back to per-message...")
                chunk_scores = [self.analyze_single_message(t) for t in chunk]
            scored.extend(chunk_scores)
        return scored

    def _get_neutral_emotions(self) -> dict[str, float]:
        """Return neutral emotion scores."""
        return {emotion: 1/len(self.emotions) for emotion in self.emotions}
    
    def _rule_based_emotion(self, text: str) -> dict[str, float]:
        """
        Simple rule-based emotion detection as fallback.
        Uses keyword matching.
        """
        text_lower = text.lower()
        scores = {emotion: 0.0 for emotion in self.emotions}
        
        # Joy keywords
        joy_words = ['happy', 'great', 'wonderful', 'amazing', 'love', 'excellent', 
                     'perfect', 'awesome', '😊', '😄', '🎉', '❤️', '😍']
        scores['joy'] = sum(1 for word in joy_words if word in text_lower)
        
        # Sadness keywords
        sad_words = ['sad', 'sorry', 'unfortunately', 'miss', 'lost', 'cry', 
                     '😢', '😭', '☹️']
        scores['sadness'] = sum(1 for word in sad_words if word in text_lower)
        
        # Anger keywords
        anger_words = ['angry', 'annoyed', 'frustrated', 'hate', 'terrible', 
                       'worst', '😠', '😡', '🤬']
        scores['anger'] = sum(1 for word in anger_words if word in text_lower)
        
        # Fear keywords
        fear_words = ['scared', 'afraid', 'worry', 'anxious', 'nervous', 'fear',
                      '😨', '😰', '😱']
        scores['fear'] = sum(1 for word in fear_words if word in text_lower)
        
        # Surprise keywords
        surprise_words = ['wow', 'amazing', 'unexpected', 'surprise', 'shocked',
                         '😮', '😲', '🤯', '!']
        scores['surprise'] = sum(1 for word in surprise_words if word in text_lower)
        
        # Love keywords
        love_words = ['love', 'adore', 'cherish', 'care', 'heart', '❤️', '😘', '💕']
        scores['love'] = sum(1 for word in love_words if word in text_lower)
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = self._get_neutral_emotions()
        
        return scores
    
    def analyze_emotions(self, 
                        df: pd.DataFrame, 
                        text_column: str = 'message',
                        batch_size: int = 32,
                        sample_cap: int | None = None) -> pd.DataFrame:
        """
        Analyze emotions for all messages in a DataFrame.
        
        Args:
            df: DataFrame with messages
            text_column: Column name containing message text
            batch_size: Number of messages to process at once (for efficiency)
            sample_cap: Option C cap on how many messages the model scores.
                None (default) or a frame at/below the cap runs the exact
                path, byte-for-byte identical to the legacy behavior. Above
                the cap, a deterministic (participant, time)-stratified
                sample is scored (random_state=42); the returned frame gains
                an `emotion_scored` boolean column and a
                `df.attrs["emotion_sample"]` dict. Any internal sampling
                failure degrades to exact scoring with the attrs marking the
                sample as failed so the pipeline labels honestly — the
                pipeline never crashes.
            
        Returns:
            DataFrame with added emotion columns
        """
        print(f"\n🎭 Analyzing emotions for {len(df)} messages...")
        
        df_copy = df.copy()
        
        # Initialize emotion columns
        for emotion in self.emotions:
            df_copy[f'emotion_{emotion}'] = 0.0
        
        # Process messages
        if self.pipeline is not None:
            sampled = False
            selected = None
            sample_meta = None
            if sample_cap is not None:
                # Option C gate: only when the scorable count exceeds the cap
                # does the deterministic stratified sample kick in; otherwise
                # the exact path below runs untouched.
                scorable_mask = df_copy[text_column].map(self._is_scorable)
                n_scorable = int(scorable_mask.sum())
                if n_scorable > sample_cap:
                    try:
                        selected = _stratified_sample_indices(
                            df_copy, scorable_mask, sample_cap
                        )
                        sampled = True
                    except Exception:
                        logger.exception(
                            "emotion sampling failed; degrading to exact scoring"
                        )
                        sample_meta = {
                            "scored": n_scorable,
                            "total": len(df_copy),
                            "cap": sample_cap,
                            "sampled": False,
                            "note": "emotion sampling failed; exact scoring used",
                        }

            if sampled:
                # Sampled path: score ONLY the deterministic sample through
                # the same batch path; every other row (scorable or not)
                # keeps the neutral 1/6 scores.
                pending = []
                skipped = []
                selected_set = set(df_copy.index[selected])
                for idx, text in df_copy[text_column].items():
                    if idx in selected_set:
                        pending.append((idx, str(text)[:512]))
                    else:
                        skipped.append(idx)

                scores_by_idx = {idx: self._get_neutral_emotions() for idx in skipped}
                if pending:
                    scored = self._score_batch([t for _, t in pending], batch_size)
                    for (idx, _), scores in zip(pending, scored):
                        scores_by_idx[idx] = scores

                for idx, scores in scores_by_idx.items():
                    for emotion in self.emotions:
                        df_copy.at[idx, f'emotion_{emotion}'] = scores.get(emotion, 0.0)

                df_copy["emotion_scored"] = df_copy.index.isin(selected_set)
                df_copy.attrs["emotion_sample"] = {
                    "scored": int(selected.sum()),
                    "total": len(df_copy),
                    "cap": sample_cap,
                    "sampled": True,
                }
            else:
                # Batch path: group scorable rows, score them in batches of
                # batch_size, and give skipped rows the neutral 1/6 scores.
                pending = []
                skipped = []
                for idx, text in df_copy[text_column].items():
                    if self._is_scorable(text):
                        pending.append((idx, str(text)[:512]))
                    else:
                        skipped.append(idx)

                scores_by_idx = {idx: self._get_neutral_emotions() for idx in skipped}

                if pending:
                    scored = self._score_batch([t for _, t in pending], batch_size)
                    for (idx, _), scores in zip(pending, scored):
                        scores_by_idx[idx] = scores

                for idx, scores in scores_by_idx.items():
                    for emotion in self.emotions:
                        df_copy.at[idx, f'emotion_{emotion}'] = scores.get(emotion, 0.0)

                if sample_meta is not None:
                    # Sampling machinery failed but every row was scored
                    # exactly — still mark it so the pipeline labels honestly.
                    df_copy["emotion_scored"] = True
                    df_copy.attrs["emotion_sample"] = sample_meta
        else:
            # Rule-based fallback, unchanged per-message loop
            for idx, row in df_copy.iterrows():
                message = row[text_column]
                emotion_scores = self.analyze_single_message(message)

                # Add scores to dataframe
                for emotion, score in emotion_scores.items():
                    df_copy.at[idx, f'emotion_{emotion}'] = score
        
        # Add dominant emotion column
        emotion_cols = [f'emotion_{e}' for e in self.emotions]
        df_copy['dominant_emotion'] = df_copy[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')
        df_copy['emotion_confidence'] = df_copy[emotion_cols].max(axis=1)
        
        print("✅ Emotion analysis complete!")
        return df_copy
    
    def get_emotion_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate comprehensive emotion summary statistics.
        
        Args:
            df: DataFrame with emotion analysis results
            
        Returns:
            Dictionary containing emotion statistics
        """
        summary = {
            'total_messages': len(df),
            'emotion_distribution': df['dominant_emotion'].value_counts().to_dict(),
            'average_emotion_scores': {
                emotion: df[f'emotion_{emotion}'].mean() 
                for emotion in self.emotions
            },
            'emotion_intensity': {
                emotion: {
                    'mean': df[f'emotion_{emotion}'].mean(),
                    'std': df[f'emotion_{emotion}'].std(),
                    'max': df[f'emotion_{emotion}'].max(),
                    'min': df[f'emotion_{emotion}'].min()
                }
                for emotion in self.emotions
            }
        }
        
        # Add per-sender statistics if sender column exists
        if 'sender' in df.columns:
            summary['by_sender'] = {}
            for sender in df['sender'].unique():
                sender_df = df[df['sender'] == sender]
                summary['by_sender'][sender] = {
                    'message_count': len(sender_df),
                    'dominant_emotions': sender_df['dominant_emotion'].value_counts().to_dict(),
                    'avg_emotion_scores': {
                        emotion: sender_df[f'emotion_{emotion}'].mean()
                        for emotion in self.emotions
                    }
                }
        
        # Add temporal analysis if datetime column exists
        if 'datetime' in df.columns:
            df_temp = df.copy()
            df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
            df_temp['date'] = df_temp['datetime'].dt.date
            
            summary['temporal_analysis'] = {
                'by_date': df_temp.groupby('date')['dominant_emotion'].apply(
                    lambda x: x.value_counts().to_dict()
                ).to_dict()
            }
        
        return summary
    
    def find_most_emotional_messages(self, 
                                     df: pd.DataFrame, 
                                     emotion: str | None = None,
                                     n: int = 5) -> pd.DataFrame:
        """
        Find messages with highest scores for specific emotion(s).
        
        Args:
            df: DataFrame with emotion analysis
            emotion: Specific emotion to filter (None for all)
            n: Number of top messages to return
            
        Returns:
            DataFrame with most emotional messages
        """
        if emotion:
            if emotion not in self.emotions:
                raise ValueError(f"Invalid emotion. Choose from: {self.emotions}")
            
            return df.nlargest(n, f'emotion_{emotion}')[
                ['datetime', 'sender', 'message', f'emotion_{emotion}', 'dominant_emotion']
            ]
        else:
            # Return top messages by confidence score
            return df.nlargest(n, 'emotion_confidence')[
                ['datetime', 'sender', 'message', 'dominant_emotion', 'emotion_confidence']
            ]
    
    def get_emotion_timeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create emotion timeline showing emotion evolution over time.
        
        Args:
            df: DataFrame with emotion analysis and datetime
            
        Returns:
            DataFrame with temporal emotion aggregations
        """
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame must contain 'datetime' column")
        
        df_temp = df.copy()
        df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
        
        # Group by date and calculate average emotion scores
        df_temp['date'] = df_temp['datetime'].dt.date
        
        emotion_cols = [f'emotion_{e}' for e in self.emotions]
        timeline = df_temp.groupby('date')[emotion_cols].mean().reset_index()
        
        return timeline


# Convenience functions for quick analysis
def quick_emotion_analysis(df: pd.DataFrame, 
                           text_column: str = 'message',
                           plot: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Perform complete emotion analysis with visualization.
    
    Args:
        df: DataFrame with messages
        text_column: Column containing message text
        plot: Whether to generate visualizations
        
    Returns:
        Tuple of (analyzed DataFrame, summary statistics)
    """
    analyzer = EmotionAnalyzer()
    df_analyzed = analyzer.analyze_emotions(df, text_column)
    summary = analyzer.get_emotion_summary(df_analyzed)
    
    if plot:
        try:
            plot_emotion_analysis(df_analyzed, summary)
        except Exception as e:  # noqa: BLE001 - plotting is best-effort; failure must never break the analysis
            print(f"⚠️ Could not generate plots: {e}")
    
    return df_analyzed, summary


def plot_emotion_analysis(df: pd.DataFrame, summary: dict):
    """
    Create comprehensive emotion visualizations.
    
    Args:
        df: DataFrame with emotion analysis
        summary: Summary statistics dictionary
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🎭 Emotion Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Emotion Distribution Pie Chart
    ax1 = axes[0, 0]
    emotion_dist = summary['emotion_distribution']
    colors = plt.cm.Set3(range(len(emotion_dist)))
    ax1.pie(emotion_dist.values(), labels=emotion_dist.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Overall Emotion Distribution', fontweight='bold')
    
    # 2. Average Emotion Scores Bar Chart
    ax2 = axes[0, 1]
    emotions = list(summary['average_emotion_scores'].keys())
    scores = list(summary['average_emotion_scores'].values())
    bars = ax2.bar(emotions, scores, color=plt.cm.Set3(range(len(emotions))))
    ax2.set_title('Average Emotion Scores', fontweight='bold')
    ax2.set_ylabel('Average Score')
    ax2.set_xlabel('Emotion')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Emotion Timeline
    ax3 = axes[1, 0]
    if 'datetime' in df.columns:
        analyzer = EmotionAnalyzer()
        timeline = analyzer.get_emotion_timeline(df)
        
        for emotion in analyzer.emotions:
            ax3.plot(timeline['date'], timeline[f'emotion_{emotion}'], 
                    marker='o', label=emotion.capitalize(), linewidth=2)
        
        ax3.set_title('Emotion Timeline', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Average Emotion Score')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, 'Timeline requires datetime column', 
                ha='center', va='center')
        ax3.set_title('Emotion Timeline', fontweight='bold')
    
    # 4. Per-Sender Emotion Comparison
    ax4 = axes[1, 1]
    if 'sender' in df.columns and 'by_sender' in summary:
        senders = list(summary['by_sender'].keys())
        emotions = analyzer.emotions
        
        x = np.arange(len(emotions))
        width = 0.35
        
        if len(senders) >= 2:
            sender1_scores = [summary['by_sender'][senders[0]]['avg_emotion_scores'][e] 
                            for e in emotions]
            sender2_scores = [summary['by_sender'][senders[1]]['avg_emotion_scores'][e] 
                            for e in emotions]
            
            ax4.bar(x - width/2, sender1_scores, width, label=senders[0], alpha=0.8)
            ax4.bar(x + width/2, sender2_scores, width, label=senders[1], alpha=0.8)
            
            ax4.set_title('Emotion Comparison by Sender', fontweight='bold')
            ax4.set_ylabel('Average Score')
            ax4.set_xlabel('Emotion')
            ax4.set_xticks(x)
            ax4.set_xticklabels([e.capitalize() for e in emotions])
            ax4.legend()
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'Need at least 2 senders', ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, 'Sender comparison unavailable', 
                ha='center', va='center')
        ax4.set_title('Emotion by Sender', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def emotion_figure(summary: dict) -> "matplotlib.figure.Figure":
    """Build a bar chart of the average emotion scores (figure-returning).

    Thin wrapper for the base64-embedded report chart (Pattern 2): builds the
    Axes and returns the Figure — NO plt.show(), so the pipeline's _safe_chart
    can encode it. Legacy plot_emotion_analysis (which calls plt.show) is
    untouched.
    """
    import matplotlib.pyplot as plt

    scores = summary.get("average_emotion_scores") or {}
    if not scores:
        scores = {e: 0.0 for e in ("joy", "sadness", "anger", "fear", "surprise", "love")}

    labels = list(scores.keys())
    values = [float(v) for v in scores.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color=plt.cm.Set3(range(len(labels))))
    ax.set_title("Emotion Scores")
    ax.set_ylabel("Average Score")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def combine_sentiment_emotion(df_sentiment: pd.DataFrame, 
                              df_emotion: pd.DataFrame) -> pd.DataFrame:
    """
    Combine sentiment and emotion analysis results.
    
    Args:
        df_sentiment: DataFrame with sentiment analysis
        df_emotion: DataFrame with emotion analysis
        
    Returns:
        Combined DataFrame with both sentiment and emotion features
    """
    # Merge on index
    df_combined = df_sentiment.copy()
    
    # Add emotion columns
    emotion_cols = [col for col in df_emotion.columns 
                   if col.startswith('emotion_') or col == 'dominant_emotion']
    
    for col in emotion_cols:
        if col in df_emotion.columns:
            df_combined[col] = df_emotion[col]
    
    return df_combined


# Module info
if __name__ == "__main__":
    print("🎭 Emotion Classification Module - Chat Analyzer Pro")
    print("This module provides advanced emotion analysis using transformers")
    print("Usage: from emotion import EmotionAnalyzer; analyzer = EmotionAnalyzer()")
