"""
Build a RAFT training set (Question, Documents, Answer) from a FAISS index of FastAPI docs.
Uses Groq's free API (OpenAI-compatible) to generate questions and answers.

Setup:
    export GROQ_API_KEY="your_key"
"""

import json
import os
import random
import re
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI


INDEX_FOLDER = "faiss_index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"  


NUM_CHUNKS = 600          
QUESTIONS_PER_CHUNK = 5    
DOCS_PER_EXAMPLE = 4       
P_ORACLE = 0.8             
MIN_CHUNK_CHARS = 300      
OUTPUT_FILE = "raft_train.jsonl"

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


def clean(text):
    text = (text.replace("\u2011", "-").replace("\u2013", "-")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u2026", "..."))
    text = re.sub(r"\s*<ANSWER>:", "\n<ANSWER>:", text)
    return text.strip()


def ask_llm(prompt):
    """Call Groq. Waits between calls and retries if we hit the rate limit."""
    for attempt in range(3):
        try:
            time.sleep(3)  
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call failed ({e}), waiting 30s and retrying...")
            time.sleep(30)
    raise RuntimeError("LLM call failed 3 times")


def make_questions(chunk_text):
    prompt = f"""Below is a passage from the FastAPI documentation.
Write {QUESTIONS_PER_CHUNK} different questions that a developer could ask and that
can be answered ONLY with this passage.

Rules:
- Each question must make sense on its own, without seeing the passage.
  Never use words like "the passage", "the text", "the context", "described",
  "shown", "according to the", "this template" or "this page".
- Use different question types, for example: a factual question, a "how do I...?"
  question, or a "when/why should I use...?" question.
- Only ask about FastAPI features and usage, not about the documentation itself.
- Do not ask who created something or about the history of a tool.
- Only ask a "why" question if the passage actually states the reason.
- Do not refer to examples, tutorials or sections of the documentation (for example "in the first example").

Return ONLY a JSON list of strings, with no extra text.

Passage:
{chunk_text}"""
    text = ask_llm(prompt)
    text = text.replace("```json", "").replace("```", "").strip()  # remove code fences
    questions = json.loads(text)
    return [clean(q) for q in questions if isinstance(q, str)][:QUESTIONS_PER_CHUNK]

BAD_QUESTION_WORDS = ["passage", "context", "the text", "described", "shown",
                      "according to the", "this template", "this page",
                      "this document", "this example",
                      "in the tutorial", "in the first", "in the example",
                      "the example above"]

BAD_CHUNK_MARKERS = ["{%", "{{", ":::"]   

def question_is_ok(question):
    return not any(w in question.lower() for w in BAD_QUESTION_WORDS)

def make_answer(question, chunk_text):
    prompt = f"""Answer the question using only the context below.

Write your answer in exactly this format:

Reasoning: <Put each sentence you use from the context between ##begin_quote## and
##end_quote##, copied exactly. Do not use quotation marks. Then write one sentence
explaining how the quotes answer the question.>
<ANSWER>: <the final answer in one or two sentences>

Example of the format (different topic):
Reasoning: ##begin_quote##The sky looks blue because air scatters sunlight.##end_quote## So scattering of sunlight by the air is what gives the sky its color.
<ANSWER>: The sky looks blue because air scatters blue light more than red light.

Rules:
- Use only facts, code and examples that appear in the context. Never write code that is not in the context.
- Quote only the sentences needed, and never repeat a quote outside the markers.
- If the context does not answer the question, write only: <ANSWER>: NOT_ANSWERABLE

Context:
{chunk_text}

Question: {question}"""
    return clean(ask_llm(prompt))

LEAKED_PHRASES = ["think step by step", "put it between", "exactly this format"]

def has_reasoning(answer):
    """True if there is a real sentence of reasoning outside the quotes."""
    outside = re.sub(r"##begin_quote##.*?##end_quote##", "", answer, flags=re.DOTALL)
    reasoning = outside.split("<ANSWER>:")[0].replace("Reasoning:", "").strip()
    return len(reasoning) >= 30

def markers_are_valid(answer):
    """Every begin/end marker must be well formed, and the counts must match."""
    begins = answer.count("##begin_quote")
    ends = answer.count("##end_quote")
    return begins == ends == answer.count("##begin_quote##") == answer.count("##end_quote##")

def answer_is_ok(answer, chunk_text):
    return (
        "<ANSWER>:" in answer
        and "NOT_ANSWERABLE" not in answer
        and not any(p in answer.lower() for p in LEAKED_PHRASES)
        and markers_are_valid(answer)      # <-- new line
        and has_reasoning(answer)
        and quotes_are_exact(answer, chunk_text)
    )


def quotes_are_exact(answer, chunk_text):
    """True if the answer has at least one quote and every quote is really in the chunk."""
    quotes = re.findall(r"##begin_quote##(.*?)##end_quote##", answer, re.DOTALL)
    if not quotes:
        return False
    return all(q.strip() in chunk_text for q in quotes)


def pick_documents(oracle, neighbors):
    """Return the list of documents shown to the model (shuffled)."""
    if random.random() < P_ORACLE:
        docs = random.sample(neighbors, DOCS_PER_EXAMPLE - 1) + [oracle]
    else:
        docs = random.sample(neighbors, DOCS_PER_EXAMPLE)  # no oracle
    random.shuffle(docs)
    return docs



def main():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   
    )
    db = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

    # Get every stored chunk and keep the long enough ones that are not templates or stubs
    all_chunks = [d.page_content for d in db.docstore._dict.values()]
    total = len(all_chunks)
    all_chunks = [
        c for c in all_chunks
        if len(c) >= MIN_CHUNK_CHARS and not any(m in c for m in BAD_CHUNK_MARKERS)
    ]
    print(f"Kept {len(all_chunks)} of {total} chunks after filtering")

    oracles = random.sample(all_chunks, min(NUM_CHUNKS, len(all_chunks)))

    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, oracle in enumerate(oracles):

            # Distractors
            similar = db.similarity_search(oracle, k=10)
            neighbors = [d.page_content for d in similar if d.page_content != oracle]

            if len(neighbors) < DOCS_PER_EXAMPLE:
                continue

            try:
                questions = make_questions(oracle)
            except Exception as e:
                print(f"Skipping chunk {i}: could not get questions ({e})")
                continue

            for question in questions:

                if not question_is_ok(question):
                    continue

                answer = make_answer(question, oracle)

                if not answer_is_ok(answer, oracle):
                    continue  

                if not quotes_are_exact(answer, oracle):
                    continue  

                docs = pick_documents(oracle, neighbors)
                context = "\n\n".join(f"<DOC>\n{d}\n</DOC>" for d in docs)

                example = {
                    "messages": [
                        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
                        {"role": "assistant", "content": answer},
                    ]
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                f.flush()  
                count += 1

            print(f"Chunk {i + 1}/{len(oracles)} done - {count} examples so far")

    print(f"Saved {count} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
