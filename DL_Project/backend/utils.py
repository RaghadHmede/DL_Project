import spacy
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

def clean_text(text, language='en'):
    if language == 'en':
        doc = nlp_en(text.lower())
    elif language == 'fr':
        doc = nlp_fr(text.lower())
    else:
        raise ValueError("Unsupported language")

    return [token.text for token in doc if not token.is_punct and not token.is_space]
