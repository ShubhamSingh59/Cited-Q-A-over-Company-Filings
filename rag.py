import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

## Load the data and variables
THERSOLD = float(os.getenv("THERSOLD", "0.2"))
HF_TOKEN = os.getenv("HF_TOKEN")

with open('data/filings.json', 'r') as f:
    docs = json.load(f)
    
## Class to run the search and vecotrization
class SearchIndex:
    def __init__(self, docs):
        self.docs = docs
        self.companies = {d['company'] for d in docs}
        self.vectorizer = TfidfVectorizer(stop_words="english")
        combined_texts = [
            f"{d.get('company', '')} {d.get('doc_type', '')} {d.get('quarter', '')} {d.get('section', '')} {d.get('text', '')}"
            for d in docs
        ]
        self.matrix = self.vectorizer.fit_transform(combined_texts)
    
    def search(self, query, top_k = 3):
        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, self.matrix)[0]
        ranked_result = sims.argsort()[::-1][:top_k]
        return [(self.docs[i], sims[i]) for i in ranked_result]
        
## Now the function for the extracting the quarter from the query
def extract_quarter(text):
    m = re.search("Q[1-4]FY\d{2}", text, re.IGNORECASE)
        
    if m:
        return m.group(0).upper()
    else:
        return None
    
## Now checking for the companies
def mentions_known_company(query, companies):
    return any(c.split()[0].lower() in query.lower() for c in companies)


## Now the promt for the our llm
def build_prompt(question, results):
    context = "\n".join(f"[{doc['doc_id']}] {doc['text']}" for doc, score in results)
    return (
        "Answer the question using ONLY the sources below. "
        "End every sentence with the matching [doc_id] tag. "
        "If the sources don't answer the question, say exactly: "
        "\"I don't have a reliable source for this.\"\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

## Now let us use the hugginface hub for llms
client = InferenceClient(
    api_key=HF_TOKEN
)

index = SearchIndex(docs)

def llm_result(query, top_k=3):
    results = index.search(query)
    if not results:
        return "I don't have a reliable source for this."
    if not mentions_known_company(query, index.companies):
        return "I don't have a reliable source for this."
    
    q_quarter = extract_quarter(query)
    top_doc, top_score = results[0]
    if q_quarter and extract_quarter(top_doc.get("quarter", "")) != q_quarter:
        top_score *= 0.4
    if top_score < THERSOLD:
        return "I don't have a reliable source for this."
        
    prompt = build_prompt(query, results)
    reply = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4.1-Flash:novita",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0,
    )
    return reply.choices[0].message.content.strip()
