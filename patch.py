import json

with open('notebooks/Avance2_#8.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'tfidf_vectorizer = TfidfVectorizer(' in source and 'norm=' not in source:
            cell['source'] = [
                "tfidf_vectorizer = TfidfVectorizer(\n",
                "    lowercase=True,\n",
                "    min_df=2,\n",
                "    max_df=0.95,\n",
                "    stop_words = stopwords_es,\n",
                "    ngram_range=(1, 2),\n",
                "    norm='l2' # Normalizacion L2 explicita para Similitud Coseno\n",
                ")"
            ]

with open('notebooks/Avance2_#8.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
