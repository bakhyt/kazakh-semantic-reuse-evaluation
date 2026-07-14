import re
import string
import stanza


# Download once if needed:
# stanza.download("kk")

nlp = stanza.Pipeline(
    lang="kk",
    processors="tokenize,lemma",
    tokenize_no_ssplit=False
)


def clean_text(text: str) -> str:
    """
    Basic text normalisation used before sentence segmentation,
    tokenisation, lemmatisation, and candidate retrieval.
    """
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stanza_sentences_and_lemmas(text: str):
    """
    Returns a list of sentences.
    Each sentence is represented as a list of lemmas.
    """
    text = clean_text(text)
    doc = nlp(text)

    sentences = []
    for sent in doc.sentences:
        lemmas = []
        for word in sent.words:
            if word.lemma:
                lemmas.append(word.lemma)
            else:
                lemmas.append(word.text)
        sentences.append(lemmas)

    return sentences


if __name__ == "__main__":
    sample = "Бұл қазақ тіліндегі мәтінді өңдеу мысалы."
    result = stanza_sentences_and_lemmas(sample)
    print(result)
