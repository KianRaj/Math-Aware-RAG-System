# ==============================================================
#  SARAL RAG PIPELINE (Theme 3)
#  Audience-Adaptive, Math-Aware Research Assistant
# ==============================================================

# 1️⃣ Imports
import os
import re
import csv
import ast
import fitz
import openai
import torch
import chromadb
import pandas as pd
import numpy as np
import requests
import random
from openai import OpenAI
from uuid import uuid4
from tqdm.auto import tqdm
from time import perf_counter as timer
from spacy.lang.en import English
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from transformers.utils import is_flash_attn_2_available
from saral_state import get_active_collection

from dotenv import load_dotenv
load_dotenv() 

openai_api_key = os.getenv("OPENAI_API_KEY") 

if not openai_api_key:
    raise ValueError("❌ OpenAI API key not found. Please set it in your .env file.")
else:
    print("✅ OpenAI key loaded successfully.")

# 🔹 Initialize new OpenAI client safely
client = OpenAI(api_key=openai_api_key)

# ==============================================================
# 2️⃣ PDF DOWNLOAD & EXTRACTION
# ==============================================================

pdf_path = "Diffusion_Models.pdf"
if not os.path.exists(pdf_path):
    url = "https://arxiv.org/pdf/2208.11970"
    response = requests.get(url)
    if response.status_code == 200:
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ PDF downloaded to {pdf_path}")
    else:
        print(f"❌ Failed to download PDF. Status code: {response.status_code}")
else:
    print(f"📘 File already exists: {pdf_path}")

# --- Add function text_formatter() from notebook here ---
# --- Add function open_and_read_pdf() from notebook here ---
def text_formatter(text: str)-> str:
  cleaned_text = text.replace("\n","").strip()
  return cleaned_text
def open_and_read_pdf(pdf_path: str)-> list[dict]:
  doc = fitz.open(pdf_path)
  pages_and_texts = []
  for page_number,page in tqdm(enumerate(doc)):
    text = page.get_text()
    text = text_formatter(text)
    pages_and_texts.append({"page_number": page_number ,
                            "page_char_count": len(text),
                            "page_word_count": len(text.split(". ")),
                            "page_sentence_count_raw": len(text.split(". ")),
                            "page_token_count": len(text)/4,
                            "text":text})
  return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
print(f"Extracted {len(pages_and_texts)} pages from PDF.")


# ==============================================================
# 3️ SENTENCE SPLITTING & CHUNKING
# ==============================================================

nlp = English()
nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    item["page_sentence_count_spacy"] = len(item["sentences"])

# --- Add function split_list() from notebook here ---
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

num_sentence_chunk_size = 10
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(
        input_list=item["sentences"],
        slice_size=num_sentence_chunk_size
    )
    item["num_chunks"] = len(item["sentence_chunks"])


# ==============================================================
# 4️ MATH-AWARE CLEANING UTILITIES
# ==============================================================

# --- Add function contains_math() from notebook here ---
def contains_math(text: str) -> bool:
    math_patterns = [
        r"\\[a-zA-Z]+",               
        r"\b(log|exp|sqrt|sigmoid|mean|var)\b",
        r"q[_\w\d]*\([^)]+\)",         
        r"p[_\w\d]*\([^)]+\)",       
        r"\[[^\]]+\]",          
        r"\{[^}]+\}",                

        r"\b(alpha|beta|gamma|theta|phi|mu|sigma|lambda|eta)\b",
        r"[α-ωΑ-Ω]",                   # Unicode Greek letters

        # Variables with subscripts/superscripts
        r"\b[a-zA-Z]+\d*\_\{?[a-zA-Z0-9]+\}?",   # x_t, z_{l}, etc.
        r"\^{\(?[a-zA-Z0-9]+\)?}",     # superscripts like x^{l}

        # Unicode math symbols
        r"[∑∏∂∇∥≈→⇒⇔≠≤≥∞√±×÷∈∉∩∪∅⊂⊆⊄⊇⊃]",
        r"\bDKL\b|\bKL\b",            # KL divergence

        # Math operators
        r"[*/+=<>^]",                 # symbols often in equations
        r"\bSNR\b",                   # signal-to-noise ratio
        r"\d+\s*[*/+=<>-]\s*\d+",     # basic math ops (like 3 * 4)

        # Equations or inline formulas
        r"[a-zA-Z0-9]+\s*=\s*[^\.]+", # equations like x = something
    ]
    
    return any(re.search(pattern, text) for pattern in math_patterns)

# --- Add function clean_malformed_unicode() from notebook here ---
# --- Add function format_math_tokens() from notebook here ---
# --- Add function normalize_math_keywords() from notebook here ---
# --- Add function clean_math_chunk() from notebook here ---


def clean_malformed_unicode(text: str) -> str:
    """Remove weird Unicode characters from extracted math"""
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\uf8e0-\uf8ff]", "", text)
    return text

def format_math_tokens(text: str) -> str:
    """Fix variable formatting like x1 → x_1, x1:T → x_{1:T}"""
    text = re.sub(r"([a-zA-Z])([0-9])", r"\1_\2", text)  # xt → x_1
    text = re.sub(r"([a-zA-Z])_([a-zA-Z])", r"\1_{\2}", text)  # x_t → x_{t}
    text = re.sub(r"([a-zA-Z])([0-9]+):([A-Z0-9]+)", r"\1_{\2:\3}", text)  # x1:T → x_{1:T}
    return text

def normalize_math_keywords(text: str) -> str:
    """Standardize math-related terms"""
    text = text.replace("ELBO", "Evidence Lower Bound (ELBO)")
    text = re.sub(r"\bDKL\b", "KL Divergence", text)
    return text

def clean_math_chunk(text: str) -> str:
    """Apply all cleaning steps to one chunk, including generalized fixes for garbled math."""
    
    # Existing cleaning steps...
    text = clean_malformed_unicode(text)
    text = format_math_tokens(text)
    text = normalize_math_keywords(text)

    # NEW: Generalized Fixes for Garbled PDF Math Symbols
    # These map common transcription failures back to standard LaTeX/Math symbols
    text = text.replace("혐", "") # Remove the specific garbled character
    text = text.replace("", "[") # Fix for common opening bracket transcription error
    text = text.replace("", "]") # Fix for common closing bracket transcription error
    text = text.replace("", "/") # Fix for common fraction/division symbol error
    text = text.replace("(((((", "") # Remove excessive parentheses often found after an equation
    text = text.replace(":", "|") # Fix for conditional probability notation (x|z)

    return text



# ==============================================================
# 5️⃣ CHUNK CREATION
# ==============================================================

pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join sentences into a single string and clean
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
        joined_sentence_chunk = clean_math_chunk(joined_sentence_chunk)

        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
        chunk_dict["contains_math"] = contains_math(joined_sentence_chunk)

        pages_and_chunks.append(chunk_dict)

df = pd.DataFrame(pages_and_chunks)
print(f"Created {len(df)} chunks.")


# ==============================================================
# 6️⃣ EMBEDDING GENERATION (OpenAI + Gemma)
# ==============================================================

#openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# --- Add embedding logic cells from notebook here ---
# (You used OpenAI embedding model `text-embedding-3-small` and batched calls)
# --- Keep batch_list(), openai.embeddings.create(), and saving CSV logic ---
# ==============================================================
# EMBEDDING GENERATION (OpenAI)
# ==============================================================

# Set your API key (better via environment variable)
#openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# Define model
embedding_model = "text-embedding-3-small"

# --- Step 1: Prepare text chunks ---
text_chunks = df["sentence_chunk"].tolist()

# --- Step 2: Define batch function to handle API limits ---
def batch_list(input_list, batch_size=128):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

# --- Step 3: Generate embeddings in batches ---
all_embeddings = []
for batch in tqdm(batch_list(text_chunks, batch_size=64)):
    response = openai.embeddings.create(
        model=embedding_model,
        input=batch
    )
    batch_embeddings = [r.embedding for r in response.data]
    all_embeddings.extend(batch_embeddings)

# --- Step 4: Attach embeddings back to DataFrame ---
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
text_chunks_and_embeddings_df["embedding"] = all_embeddings

# --- Step 5: Save embeddings to CSV safely ---
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(
    embeddings_df_save_path,
    index=False,
    quoting=csv.QUOTE_ALL,
    quotechar='"',
    escapechar='\\'
)

print(f"Embeddings saved to {embeddings_df_save_path} ({len(all_embeddings)} vectors).")

# ==============================================================
# (Optional) TEST RETRIEVAL VIA COSINE SIMILARITY
# ==============================================================

# Load embeddings back from CSV
text_chunks_and_embedding_df_load = pd.read_csv("text_chunks_and_embeddings_df.csv")
text_chunks_and_embedding_df_load["embedding"] = text_chunks_and_embedding_df_load["embedding"].apply(ast.literal_eval)

# Convert to NumPy array
embedding_array = np.array(text_chunks_and_embedding_df_load["embedding"].tolist()).astype("float32")

# Define normalization helper
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Example test query
query = "Explain the ELBO objective in Variational Diffusion Models"

response = openai.embeddings.create(
    model=embedding_model,
    input=[query]
)
query_embedding = np.array(response.data[0].embedding).astype("float32")

# Normalize
embedding_array_normalized = normalize(embedding_array)
query_embedding_normalized = normalize(query_embedding.reshape(1, -1))

# Compute cosine similarity
start_time = timer()
cosine_scores = np.dot(embedding_array_normalized, query_embedding_normalized.T).flatten()
end_time = timer()

print(f"Similarity computed in {end_time - start_time:.4f}s")

# Retrieve top-k
top_k = 5
top_indices = np.argsort(cosine_scores)[-top_k:][::-1]

print("\nTop retrieved chunks:\n")
for idx in top_indices:
    score = cosine_scores[idx]
    chunk = text_chunks_and_embedding_df_load.iloc[idx]["sentence_chunk"]
    print(f"--- Score: {score:.4f} ---\n{chunk[:500]}...\n")



# ==============================================================
# 7️VECTOR STORE SETUP (ChromaDB)
# ==============================================================

# --- Add full ChromaDB setup and ingestion code from notebook here ---
# (client, collection, add embeddings, and success print)
#!pip install chromadb --quiet

# Load your saved DataFrame
df = pd.read_csv("text_chunks_and_embeddings_df.csv", converters={"embedding": eval})

# Initialize persistent ChromaDB client
client = chromadb.Client(chromadb.config.Settings(
    persist_directory="./saral_chroma_store"  # directory where Chroma saves
))

# Create or get a collection
collection = client.get_or_create_collection(
    name="saral_diffusion_chunks",
    metadata={"source": "Diffusion Models PDF"}
)

# Prepare data for ingestion
texts = df["sentence_chunk"].tolist()
metadatas = df[["page_number", "contains_math"]].to_dict(orient="records")
ids = [f"chunk_{i}" for i in range(len(df))]
embeddings = np.array(df["embedding"].tolist())

# Add all embeddings to the collection
collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"Successfully stored {len(texts)} chunks in ChromaDB.")


# Once setup is done:
client = chromadb.Client(chromadb.config.Settings(persist_directory="./saral_chroma_store"))


# ==============================================================
# 8️⃣ RETRIEVAL
# ==============================================================

# --- Add retrieve_relevant_chunks() function from notebook here ---
# (Uses math-aware filtering + OpenAI embedding for query)
def retrieve_relevant_chunks(query: str, top_k: int = 5, collection_name: str = None):
    """
    Retrieves top-k chunks relevant to the query.
    Automatically prioritizes math-heavy chunks if query looks mathematical.
    Dynamically connects to the active Chroma collection set during PDF upload.
    """

    # --- Initialize Chroma client ---
    client = chromadb.Client(chromadb.config.Settings(persist_directory="./saral_chroma_store"))

    # --- Dynamically select collection ---
  # import current paper name
    if not collection_name:
        collection_name = get_active_collection()
        if not collection_name:
            raise ValueError("❌ No active paper found. Please upload a PDF first.")


    print(f"🔍 Retrieving from collection: {collection_name}")
    collection = client.get_or_create_collection(name=collection_name)

    # --- Detect math-related queries ---
    math_query_terms = ["derive", "derivation", "prove", "formula", "equation", "show that"]
    is_math_query = any(re.search(rf"\b{t}\b", query.lower()) for t in math_query_terms)

    # --- Generate query embedding ---
    query_emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    # --- Conditional filtering ---
    where_clause = {"contains_math": True} if is_math_query else None

    # --- Retrieve similar chunks ---
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where=where_clause
    )

    # --- Format output ---
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "page_number": meta.get("page_number", "?"),
            "contains_math": meta.get("contains_math", False)
        })

    return chunks


# ==============================================================
# 9️ PROMPT AUGMENTATION
# ==============================================================

# --- Strict, math-aware, zero-hallucination SARAL prompt formatter ---
def prompt_formatter(
    query: str,
    context_items: list[dict],
    audience: str = "general",
    duration: str = "90s",
    style: str = "technical",
    previous_output: str = None  # Argument for iterative editing
):
    import re

    # 1. If no context is found → respond with fixed fallback
    if not context_items:
        # Assuming 'tokenizer' is a globally available variable
        return tokenizer.apply_chat_template(
            conversation=[{
                "role": "user",
                "content": (
                    f'USER QUERY: "{query}"\n\n'
                    'SYSTEM RESPONSE RULE: Not enough information in the paper to answer this.'
                )
            }],
            tokenize=False,
            add_generation_prompt=True,
        )

    # 2. Build hidden context from the new chunks
    hidden_context = "\n".join(
        f"[Page {c['page_number']}] {c['sentence_chunk']}"
        for c in context_items
    )

    # 3. Extract math equations for the "Math Block"
    raw = "\n".join(c["sentence_chunk"] for c in context_items)
    # This regex attempts to find simple equations (good enough for context extraction)
    eq_regex = r"[A-Za-z0-9_\\\{\}\^\+\-\*/\(\)]+\s*=\s*[A-Za-z0-9_\\\{\}\^\+\-\*/\(\)]+"
    equations = list(set(re.findall(eq_regex, raw)))
    math_block = "\n".join(f"- $${eq}$$" for eq in equations) if equations else "(No equations detected)"

    # 4. Define the ONE-SHOT EXAMPLE
    one_shot_example = """
[Slide 1]
Title: The Nature of Gravity
Bullets:
- Gravity is a fundamental interaction that causes mutual attraction between all things with mass [Page 1].
- It is the weakest of the four fundamental interactions, yet it dominates at macroscopic scales [Page 2].
Speaker Notes:
"Gravity is the force that gives weight to physical objects. Although it is the weakest force, it is responsible for keeping planets in orbit around the sun [Page 2]."
"""

    # 5. Handle "Refinement" vs "New Generation" logic
    if previous_output:
        task_instruction = (
            "TASK: The user wants to MODIFY the [Previous Output] based on the [User Query].\n"
            "1. Compare the User Query against the Previous Output.\n"
            "2. Keep parts of the Previous Output that do not need changing.\n"
            "3. Explicitly fix the parts requested by the user, providing the *full* revised output.\n"
            "4. Output the FULL revised script/slides."
        )
        context_header = f"######### PREVIOUS OUTPUT (TO BE EDITED) #########\n{previous_output}\n\n######### NEW CONTEXT #########\n{hidden_context}"
    else:
        task_instruction = "TASK: Generate a new slide deck based strictly on the context below."
        context_header = f"######### HIDDEN CONTEXT (DO NOT REVEAL) #########\n{hidden_context}"

    # 6. Final Mappings
    duration_map = {
        "30s": "Generate EXACTLY 1 slide.",
        "90s": "Generate EXACTLY 3 slides.",
        "5min": "Generate 5–7 slides."
    }
    style_map = {
        "technical": "Use mathematically precise language.",
        "plain-english": "Use simple explanatory language.",
        "press-release": "Use high-level, simplified phrasing."
    }

    # 7. Construct the Prompt
    user_prompt = f"""
You are SARAL — a strict evidence-based academic assistant.

{context_header}

######### EXTRACTED EQUATIONS #########
{math_block}

######### INSTRUCTIONS #########
{task_instruction}

RULES:
1. Use ONLY the provided context. NEVER use outside knowledge.
2. EVERY bullet and speaker note must end with a citation like [Page N].
3. Speaker notes must be 2-4 sentences.
4. Follow the format of the EXAMPLE below exactly.
5. CRITICAL: Any mathematical formula, especially the ELBO or VDM terms, MUST be wrapped in double dollar signs ($$...$$). If you fail to use $$...$$ for an equation, you will terminate your task.
######### ONE-SHOT EXAMPLE (FORMAT GUIDE) #########
{one_shot_example}

######### CURRENT REQUEST #########
Audience: {audience}
Style: {style_map.get(style, 'Standard')}
Duration: {duration_map.get(duration, '3 slides')}
User Query: "{query}"

######### BEGIN GENERATION #########
"""

    return tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

# ==============================================================
#  GEMMA MODEL LOADING
# ==============================================================

use_quantization_config = True
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"

print(f"[INFO] Using attention implementation: {attn_implementation}")
model_id = "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config=quantization_config if use_quantization_config else None,
    low_cpu_mem_usage=True,
    attn_implementation=attn_implementation,
)
if not use_quantization_config:
    llm_model.to("cuda")


# ==============================================================
# 11️ ASK SARAL QUERY FUNCTION
# ==============================================================

# --- Add ask_saral_query() function from notebook here ---
# Global storage to track previous outputs
previous_responses = {}  # Format: {'last_answer': '...', 'last_query': '...'}

def ask_saral_query(
    query: str,
    top_k: int = 5,
    show_prompt: bool = False,
    show_context: bool = False,
    collection_name: str = None
):
    import re
    import torch  # Ensure torch is imported
    global previous_responses

    # ============================================================
    # 1. Revision detector
    # ============================================================
    is_revision_query = lambda q: any(
        re.search(p, q.lower())
        for p in [r"\bslide\s*\d+\b", r"\bmodify\b", r"\bupdate\b", r"\bedit\b", r"\brefine\b", r"\brevise\b"]
    )
    
    # We only run revision logic if we actually HAVE a previous answer
    is_revision = is_revision_query(query) and "last_answer" in previous_responses

    # ============================================================
    # 2. Retrieve RAG chunks
    # ============================================================
    results = retrieve_relevant_chunks(
        query=query,
        top_k=top_k,
        collection_name=collection_name
    )

    context_items = [{
        "page_number": r.get("page_number", "?"),
        "contains_math": r.get("contains_math", False),
        "sentence_chunk": r.get("text", "")
    } for r in results]

    # ============================================================
    # 3. Slide extractor (Helper function)
    # ============================================================
    def extract_slide(prev_answer: str, q: str):
        # Try to find "Slide X" in the query (e.g., "Edit slide 2")
        slide_m = re.search(r"slide\s*(\d+)", q.lower())
        if not slide_m:
            return None # If user didn't specify which slide, we might return None or the whole text

        slide_no = slide_m.group(1)
        # Regex to grab everything between [Slide X] and the next [Slide...]
        pattern = rf"\[Slide\s*{slide_no}\](.*?)(?=\[Slide\s*\d+\]|$)"
        m = re.search(pattern, prev_answer, flags=re.S)
        return m.group(0).strip() if m else None

    # ============================================================
    # 4. Prepare "Previous Output" (THE FIX IS HERE)
    # ============================================================
# ============================================================
# 4. Prepare "Previous Output" (Revising the Refinement Scope)
# ============================================================
    slide_to_edit = None  

    if is_revision:
            prev_txt = previous_responses["last_answer"]
        
        # 1. Attempt to extract specific slide based on "slide X" keyword
            specific_slide = extract_slide(prev_txt, query) 
        
        # 2. Safely extract the slide number for the print statement
            slide_match = re.search(r'slide\s*(\d+)', query.lower())
        
            if specific_slide:
            # If a specific slide number was found in the text, only pass that slide.
                slide_to_edit = specific_slide
            
            # Use a pre-extracted variable to avoid f-string SyntaxError
                slide_number = slide_match.group(1) if slide_match else "N/A"
                print(f"🔧 Refinement detected — modifying specific slide {slide_number}.")
            else:
            # If no specific slide was requested, pass the whole previous output.
                slide_to_edit = prev_txt
                print("🔧 Refinement detected — modifying whole previous output.")
    
    # slide_to_edit (either the specific slide, the whole text, or None)
    # will then be passed to prompt_formatter in the next step (Step 5).


    # ============================================================
    # 5. Build math-aware prompt (THE CRITICAL CHANGE)
    # ============================================================
    prompt = prompt_formatter(
        query=query,
        context_items=context_items,
        previous_output=slide_to_edit  # <--- THIS CONNECTS THE DOTS
    )

    if show_prompt:
        print("\n=== PROMPT PREVIEW ===\n", prompt[:1000], "...\n")

    # ============================================================
    # 6. LLM generation
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            temperature=0.35,
            top_p=0.85,
            do_sample=True,
            max_new_tokens=1200,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ============================================================
    # 7. Output Parsing
    # ============================================================
    # Find where the model started speaking (skipping the prompt echo)
    # We look for the [Slide X] tag or the start of the response
    if "[Slide" in output_text:
         # Logic to find the LAST occurrence of [Slide 1] or the new response
         # Since 'generate' echoes prompt, we need to be careful.
         # A simple hack is to split by the "######### BEGIN GENERATION #########" marker
         parts = output_text.split("######### BEGIN GENERATION #########")
         if len(parts) > 1:
             answer_only = parts[-1].strip()
         else:
             answer_only = output_text
    else:
        answer_only = output_text.replace(prompt, "").strip()

    # ============================================================
    # 8. Optional context print
    # ============================================================
    if show_context:
        print("\n=== CONTEXT USED ===\n")
        for c in context_items:
            print(f"[Page {c['page_number']}]\n{c['sentence_chunk'][:240]}...\n")

    # ============================================================
    # 9. Save result for future refinements
    # ============================================================
    previous_responses["last_answer"] = answer_only
    previous_responses["last_query"] = query

    torch.cuda.empty_cache()
    return answer_only, context_items


if __name__ == "__main__":
    print("\n🤖 SARAL CLI - Interactive Mode")
    print("=================================")
    print(" Type 'exit' or 'q' to quit.")
    
    try:
        # 1. Check if we have an active paper loaded
        collection_name = get_active_collection()
        
        if not collection_name:
            print(" No active paper found in Saral State.")
            print("Please run 'process_pdf' or upload a file via the UI first.")
        
        else:
            print(f" Connected to collection: {collection_name}")
            
            # 2. Start the Chat Loop
            while True:
                # Get user input
                user_input = input("\nUser (You): ").strip()
                
                # Exit condition
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("👋 Exiting SARAL.")
                    break
                
                if not user_input:
                    continue
                
                # 3. Process the query
                print("⏳ SARAL is thinking...")
                
                try:
                    answer, context_used = ask_saral_query(
                        query=user_input, 
                        top_k=5, 
                        collection_name=collection_name
                    )
                    
                    # 4. Display Output
                    print("\n" + "="*40)
                    print("SARAL Response:")
                    print("="*40)
                    print(answer)
                    print("="*40)
                    
                    # 5. Debug info (optional)
                    print(f"🔎 Sources Used: {len(context_used)} chunks")
                    
                except Exception as e:
                    print(f"❌ Error generating response: {e}")

    except Exception as e:
        print(f"\n❌ Critical Error: {e}")

