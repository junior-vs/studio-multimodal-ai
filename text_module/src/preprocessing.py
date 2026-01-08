"""
Text preprocessing utilities

This module contains functions for text preprocessing and cleaning.
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextPreprocessor:
    """Text preprocessing utility class."""

    def __init__(self, language: str = "english"):
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Download required NLTK data
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")


def clean_text(
    text: str,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    lowercase: bool = True,
) -> str:
    """
    Clean and normalize text.

    Args:
        text: Input text
        remove_punctuation: Whether to remove punctuation
        remove_numbers: Whether to remove numbers
        lowercase: Whether to convert to lowercase

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", "", text)

    return text


def tokenize_text(
    text: str, remove_stopwords: bool = True, language: str = "english"
) -> List[str]:
    """
    Tokenize text and optionally remove stopwords.

    Args:
        text: Input text
        remove_stopwords: Whether to remove stopwords
        language: Language for stopwords

    Returns:
        List of tokens
    """
    tokens = word_tokenize(text)

    if remove_stopwords:
        stop_words = set(stopwords.words(language))
        tokens = [token for token in tokens if token.lower() not in stop_words]

    return tokens
