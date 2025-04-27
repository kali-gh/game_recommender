import logging
import pandas as pd
import csv

from bs4 import BeautifulSoup

from pprint import pprint  # pretty-printer
from collections import defaultdict

from gensim.utils import simple_preprocess
from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel
from gensim import corpora
from gensim.matutils import corpus2csc

from pathlib import Path

logger = logging.getLogger(__name__)


STOPLIST = [
    'about',
    'class',
    'bb',
    'for',
    'a',
    'of',
    'the',
    'and',
    'to',
    'in',
    'this',
    'game',
    'akamai',
    'steam',
    'apps',
    'bb_tag',
    'bb_img',
    'https',
    'steamstatic',
    'where',
    'other',
    'bb_paragraph',
    'strong',
    'extras',
    'shared',
    'their',
    'bb_ul',
    'including',
    'steampowered'
    
]

def parse_texts(documents, stoplist, min_word_length = 4):

    documents = remove_stopwords(documents, stoplist)
    
    updated_documents = [simple_preprocess(d, min_len = min_word_length) for d in documents]
    updated_documents = [' '.join(document) for document in updated_documents]
    
    updated_documents = remove_stopwords(updated_documents, stoplist)

    texts = [simple_preprocess(d, min_len = min_word_length) for d in updated_documents]
    
    return texts


def build_embeddings(texts, num_topics):

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    
    return model, dictionary, corpus


def remove_stopwords(documents, stoplist):
        
    # remove common words and tokenize

    stoplist = set(stoplist)
    texts = [
        ' '.join([word for word in document.lower().split() if word not in stoplist])
        for document in documents
    ]
    return texts

def build_df_embeddings(documents, min_word_length, parse=False):

    Path("model").mkdir(exist_ok=True, parents=True)
    
    if parse:
        logger.warning(f"Reparsing all the input files !!!")
        import time
        start = time.time()
        texts = parse_texts(
            documents=documents, 
            stoplist=STOPLIST,
            min_word_length=min_word_length
        )
        end = time.time()
        logger.info(f"Parsing completed in {round(end - start,2)} seconds")
        
        with open("model/texts.txt", "w") as f:
            writer = csv.writer(f)
            writer.writerows(texts)
    else:
        pass
    
    with open("model/texts.txt", "r") as f:
        csv_reader = csv.reader(f) 
        texts = list(csv_reader)
        #print(texts)
        
    model, dictionary, corpus = build_embeddings(
        texts=texts,
        num_topics=50
    )
    
    model.save('model/model.model')
    
    vectorized_corpus = model[corpus]

    vectorized_corpus_matrix = corpus2csc(vectorized_corpus)

    vectorized_corpus_matrix = vectorized_corpus_matrix.transpose()

    vectorized_corpus_matrix = vectorized_corpus_matrix.toarray()

    df = pd.DataFrame(vectorized_corpus_matrix)

    df.columns = [f'x_content_{i}' for i in df.columns]

    return df