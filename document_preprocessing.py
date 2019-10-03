"""
document_preprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import re
import nltk
import requests
import numpy as np
import pandas as pd
from pprint import pprint
from bs4 import BeautifulSoup
from functools import wraps, reduce, partial
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, NoReturn, Optional, NewType
pd.set_option('display.max_columns', None)

#  sample documents
documents: Dict[str, str] = {
    "doc1": "Ben studies about computers in Computer lab",
    "doc2": "Steve teaches at Brown University",
    "doc3": "Data Scientists work on large datasets"
}
sample_docs = [ "Ben studies about computers in Computer lab",
                "Steve teaches at Brown University",
                "Data Scientists work on large datasets"]


# document processing pipeline
def build_document_matrix(data: dict, *stopwords) -> dict:
    """Builds out the document matrix."""
    porter_stem = nltk.stem.PorterStemmer()

    document_dict = {k:
        set(data[k].split(" "))
            for k in data.keys()
    }

    for k in document_dict.keys():
        document_dict[k] = {c: [] for c in document_dict[k] if c not in list(stopwords)}


    return document_dict

print(build_document_matrix(documents, 'at', 'in', 'on'))



# TF-term frequency algorithm
def term_frequency(term: str, doc: str):
    """
    :param term: Term that will be used to calculate tf.
    :param doc: Which document to search through.
    :return normalized_tf: tf(t,d) = N(t, d) / ||D||
        where ||D|| = total # of terms in the document
    """
    # split the document into a list of terms
    normalized_term_freq: List[str] = doc.lower().split()

    document_term = normalized_term_freq.count(term.lower())

    # get the document length
    document_length = len(normalized_term_freq)

    # calculate the normalized term frequency
    normalized_tf = document_term / document_length
    return normalized_tf

def inverse_document_frequency(term: str, documents: List[str]) -> float:

    current_term_count = 0
    for document in documents:
        if term.lower() in document.lower().split():
            current_term_count += 1

    if current_term_count > 0:
        total_docs = len(documents)

        idf_val = np.log(total_docs / current_term_count)
        return  idf_val
    else:
        return 0.0

def tf_idf(term: str, documents: List[str]) -> List[float]:
    return [term_frequency(term, document)*inverse_document_frequency(term, documents)
            for document in documents]




pprint(term_frequency(term="computer", doc=documents["doc1"]))



pprint(inverse_document_frequency(term="computer", documents=sample_docs))

pprint(tf_idf(term="computer", documents=sample_docs))








