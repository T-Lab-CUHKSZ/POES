import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(texts, max_features=5000):
    """Cosine similarity between TF-IDF vectors of `texts`, clipped at 0, with a unit diagonal."""
    try:
        vectors = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)).fit_transform(texts)
        if vectors.shape[1] == 0:
            raise ValueError("empty vocabulary")
    except ValueError:
        # Inputs without word tokens (e.g. pure arithmetic) fall back to character n-grams.
        vectorizer = TfidfVectorizer(max_features=max_features, analyzer="char_wb", ngram_range=(2, 4))
        vectors = vectorizer.fit_transform(texts)

    sim = np.maximum(cosine_similarity(vectors), 0.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim
