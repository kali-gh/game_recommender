import logging
import pandas as pd
import csv
import time

from typing import List
from pathlib import Path
from gensim.utils import simple_preprocess
from gensim.models import LsiModel
from gensim import corpora
from gensim.matutils import corpus2csc

logger = logging.getLogger(__name__)


def parse_texts(
        documents : List[str],
        stoplist : List[str],
        min_word_length : int) -> List[str]:
    """
    removes stop words and preprocesses the documents using min word length
    
    Args:
        documents: list of documents (each a string)
        stoplist: words to exclude
        min_word_length: min word length to keep after simple processing

    Returns:
        cleaned documents as a list

    """""

    documents = remove_stopwords(documents, stoplist)
    
    updated_documents = [simple_preprocess(d, min_len = min_word_length) for d in documents]
    updated_documents = [' '.join(document) for document in updated_documents]
    
    updated_documents = remove_stopwords(updated_documents, stoplist)

    texts = [simple_preprocess(d, min_len = min_word_length) for d in updated_documents]
    
    return texts


def build_embeddings(
        texts,
        num_topics):
    """
    builds the LSI model

    Args:
        texts: list of documents
        num_topics: number of topics to build

    Returns:
        Tuple
            model - the LSA model
            dictionary - dict used by the model
            corpus - corpus used by the model
    """

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    
    return model, dictionary, corpus


def remove_stopwords(
        documents : List[str],
        stoplist : List[str]) -> List[str]:
    """
    removes words from the documents

    Args:
        documents: list of documents (each a string)
        stoplist: words to exclude

    Returns:
        cleaned documents as a list
    """

    # remove common words and tokenize
    stoplist = set(stoplist)
    texts = [
        ' '.join([word for word in document.lower().split() if word not in stoplist])
        for document in documents
    ]
    return texts

def build_df_embeddings(
        documents : List[str],
        model_dir : str,
        stoplist : List[str],
        num_topics : int,
        min_word_length : int,
        parse : bool = True) -> pd.DataFrame:
    """
    Builds the embeddings and return the vectors as a df which is N_doc x N_embed

    Args:
        documents: list of strings representing documents to vectorize
        model_dir: model directory
        stoplist: list of all words to exclude
        num_topics: number of dimensions for the embedding
        min_word_length: min word length
        parse: if we rebuild the model (should be true)

    Returns:
        Pandas df with the embeddings as above
    """

    Path(model_dir).mkdir(exist_ok=True, parents=True)
    
    if parse:
        logger.warning(f"Reparsing all the input files !")

        start = time.time()
        texts = parse_texts(
            documents=documents,
            stoplist=stoplist,
            min_word_length=min_word_length
        )
        end = time.time()
        logger.info(f"Parsing completed in {round(end - start,2)} seconds")
        
        with open(f"{model_dir}/texts.txt", "w") as f:
            writer = csv.writer(f)
            writer.writerows(texts)
    else:
        logger.info("Using existing model")
    
    with open(f"{model_dir}/texts.txt", "r") as f:
        csv_reader = csv.reader(f) 
        texts = list(csv_reader)
        logger.debug(texts)

    logger.info("Building the embeddings...")
    model, dictionary, corpus = build_embeddings(
        texts=texts,
        num_topics=num_topics
    )

    logger.info(f"Saving the model at {model_dir}")
    model.save(f'{model_dir}/model.model')

    logger.info("Creating vectorized corpus")

    vectorized_corpus = model[corpus]
    vectorized_corpus_matrix = corpus2csc(vectorized_corpus)
    vectorized_corpus_matrix = vectorized_corpus_matrix.transpose()
    vectorized_corpus_matrix = vectorized_corpus_matrix.toarray()

    df = pd.DataFrame(vectorized_corpus_matrix)

    df.columns = [f'x_content_{i}' for i in df.columns]

    logger.info(f"Created embedding with n dimensions : {len(df.columns)}")

    return df