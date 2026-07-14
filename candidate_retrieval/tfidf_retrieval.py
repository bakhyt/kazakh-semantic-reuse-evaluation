from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_tfidf_candidates(
    suspicious_sentences,
    source_sentences,
    threshold=0.10
):
    """
    Retrieves suspicious-source sentence pairs using TF-IDF cosine similarity.

    Parameters
    ----------
    suspicious_sentences : list[str]
        Sentences from the suspicious document.
    source_sentences : list[str]
        Sentences from the source document.
    threshold : float
        Recall-oriented TF-IDF threshold. In the CLEF 2026 experiments,
        the long-document candidate retrieval threshold was 0.10.

    Returns
    -------
    list[dict]
        Candidate pairs with suspicious index, source index, and score.
    """
    all_sentences = suspicious_sentences + source_sentences

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    suspicious_matrix = tfidf_matrix[:len(suspicious_sentences)]
    source_matrix = tfidf_matrix[len(suspicious_sentences):]

    similarity_matrix = cosine_similarity(suspicious_matrix, source_matrix)

    candidates = []

    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            score = similarity_matrix[i, j]
            if score >= threshold:
                candidates.append({
                    "suspicious_index": i,
                    "source_index": j,
                    "score": float(score)
                })

    return candidates


if __name__ == "__main__":
    suspicious = [
        "Қазақстанда жасанды интеллект зерттеулері дамып келеді.",
        "Бұл сөйлем басқа тақырып туралы."
    ]

    source = [
        "Қазақстанда AI және табиғи тіл өңдеу зерттеулері дамуда.",
        "Ауа райы бүгін жылы болады."
    ]

    candidates = retrieve_tfidf_candidates(suspicious, source)
    print(candidates)
