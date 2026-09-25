"""Bit-identical parity tests for the vectorized ChatEDA.analyze_content.

The optimized word/emoji path must reproduce the pre-optimization result
dict exactly (same counts, same Counter insertion order, Python int counts),
because consumers rely on most_common tie order (adapters.py:103-106) and
the int/str types flowing into the report.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import random
import re
from collections import Counter

import pandas as pd
import pytest

from chat_analyzer.analysis.eda import ChatEDA
from chat_analyzer.ingest.ingestion import messages_to_dataframe


def _clean_text(text):
    if pd.isna(text) or text == '<Media omitted>':
        return ""
    return re.sub(r'[^\w\s]', ' ', text.lower())


# Original analyze_content body, embedded verbatim as the reference oracle.
def _reference_analyze_content(df):
    all_text = ' '.join(df['message'].apply(_clean_text))
    words = [w for w in all_text.split() if len(w) > 2]
    word_freq = Counter(words)

    all_emojis = []
    for msg in df['message']:
        emojis = re.findall(r'[😀-🙏🌀-🗿]', str(msg))
        all_emojis.extend(emojis)

    return {
        'word_frequency': word_freq,
        'emoji_frequency': Counter(all_emojis),
        'total_words': sum(df['word_count']),
        'unique_words': len(word_freq),
    }


def _prepared(messages):
    df = messages_to_dataframe(messages)
    df['word_count'] = df['message'].str.split().str.len()
    return df


def _optimized(df):
    container = object.__new__(ChatEDA)
    container.df = df
    return container.analyze_content()


def _assert_parity(opt, ref):
    assert opt['word_frequency'] == ref['word_frequency']
    assert list(opt['word_frequency']) == list(ref['word_frequency'])
    assert opt['emoji_frequency'] == ref['emoji_frequency']
    assert list(opt['emoji_frequency']) == list(ref['emoji_frequency'])
    assert opt['total_words'] == ref['total_words']
    assert opt['unique_words'] == ref['unique_words']
    assert all(type(c) is int for c in opt['word_frequency'].values())
    assert all(type(c) is int for c in opt['emoji_frequency'].values())


@pytest.mark.parametrize('messages,label', [
    (
        [],
        'empty: zero rows must produce empty Counters, 0 totals',
    ),
    (
        ['আমি ভালো আছি hello', 'তুমি কেমন আছো? hi there', 'এক দুই তিন four five'],
        'unicode+bengali: \\w tokens and ?-punctuation behave like the reference',
    ),
    (
        ['Hello, world!! 😀🎉 test---word', 'email@example.com', 'wow!!! awesome stuff 🚀😀'],
        'emoji+punct: emojis/punct become spaces, >2len tokens survive',
    ),
    (
        ['😀🎉 real text here', '<Media omitted>', 'another real one', None],
        'media+nan: <Media omitted>/NaN rows contribute no words or emojis',
    ),
    (
        ['delta banana apple delta', 'apple banana', 'delta apple', 'delta banana apple'],
        'ties: equal counts keep first-occurrence insertion order, not sorted',
    ),
    (
        ['yes no maybe yes', 'no maybe yes maybe', 'yes yes no maybe', 'yes maybe no'],
        'repeated words: cross-message frequency and tie order match',
    ),
])
def test_analyze_content_parity(messages, label):
    if not messages:
        prepared = pd.DataFrame({
            'message': pd.Series([], dtype=object),
            'word_count': pd.Series([], dtype='int64'),
        })
    else:
        prepared = _prepared([{'datetime': f'2024-01-01T0{i}:00:00',
                               'sender': 'Alice', 'message': m} for i, m in enumerate(messages)])
    ref = _reference_analyze_content(prepared)
    opt = _optimized(prepared)
    _assert_parity(opt, ref)
    assert isinstance(opt['unique_words'], int)


@pytest.mark.slow
def test_analyze_content_parity_large():
    rng = random.Random(1234)
    pool = [
        'hello world this is a test',
        'hi there how are you doing',
        'wenk is my favorite word today',
        'আমি বাংলায় কথা বলি hello',
        'wow!!! 😀🎉 amazing stuff',
        'yeah right absolutely sure',
        'hmm okay whatever fine',
        'let me think about this one',
        'no way that cannot be true ok',
        '<Media omitted>',
        'just a short note',
    ]
    messages = []
    for i in range(100_000):
        messages.append({
            'datetime': f'2024-01-{i % 28 + 1:02d}T{i % 24:02d}:{i % 60:02d}:00',
            'sender': ['Alice', 'Bob', 'Carol'][i % 3],
            'message': rng.choice(pool),
        })
    prepared = _prepared(messages)
    ref = _reference_analyze_content(prepared)
    opt = _optimized(prepared)
    _assert_parity(opt, ref)
    assert opt['unique_words'] > 0
    assert opt['total_words'] > 0