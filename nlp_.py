"""
nlp_.py
~~~~~~~
"""
import re
import nltk
import requests
import pandas as pd
from pprint import pprint
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_columns', None)

# url = "https://www.oreilly.com/ideas/what-is-data-science"
# html = requests.get(url).text
# soup = BeautifulSoup(html, 'html5lib')
# content = soup.find("div", "article-body")
# regex = re.compile(r"[\w']+|[\.]")
# document = []
#
# def fix_unicode(text: str) -> str:
#     return text.replace(u"\u2019", "'")
#
# for parapgraph in content("p"):
#     words = regex.findall(fix_unicode(parapgraph.text))
#     document.extend(words)
#
# pprint(document)
#content = html.content.decode(encoding='utf-8', errors='ignore')
#soup = BeautifulSoup(content, 'html.parser')
#pprint(soup.find_all('p'))



# splitting text with nltk.tokenize.WhiteSpaceTokenizer
text = "This is Andrew's text, isn't it?"
text_list = text.split(" ")
tokenizer = nltk.tokenize.WhitespaceTokenizer()
print(tokenizer.tokenize(text))
print(text_list)


# Porter's stem
porter_stem = nltk.stem.PorterStemmer()
tok = "cats"
print(porter_stem.stem(tok))

#WordNet Lemmatizer
# wn_lemma = nltk.stem.WordNetLemmatizer()
# print(wn_lemma.lemmatize("wolves"))

text_two = "feet cats wolves talked"

tree_tokenizer = nltk.tokenize.TreebankWordTokenizer()
tree_tokens = tree_tokenizer.tokenize(text_two)

print(" ".join(porter_stem.stem(t) for t in tree_tokens))


# text vectorization: Bag of Words (BOW)
################################################################################
movie_reviews = ["good movie", "not a good movie", "did not like", "i liked it", "good one"]
words_dict = {k:
                  set(" ".join(review for review in movie_reviews).split(" "))
              for k in movie_reviews}

for k in words_dict.keys():
    words_dict[k] = {c: [] for c in words_dict[k]}

pprint(words_dict)

# TF-IDF example
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
features = tfidf.fit_transform(movie_reviews)
print(pd.DataFrame(features.todense(), columns=tfidf.get_feature_names()))