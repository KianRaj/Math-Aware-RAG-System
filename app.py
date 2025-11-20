# ==============================================================
#  SARAL Chatbot UI (Theme 3 Extended)
#  Adaptive + Math-Aware + Iterative Slide Refinement
# ==============================================================

import os
import sys
os.environ["GRADIO_TEMP_DIR"] = "./gradio_tmp"  # Fix for permission error
os.makedirs("./gradio_tmp", exist_ok=True)

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import fitz
import shutil
import tempfile
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
import chromadb
import openai
global ACTIVE_COLLECTION
from saral_state import set_active_collection, get_active_collection



# --- Import core functions from saral.py ---
from backend import (
    ask_saral_query,
    retrieve_relevant_chunks,
    open_and_read_pdf,
    contains_math,
    clean_math_chunk
)


# ==============================================================
#  Environment setup
# ==============================================================

# Ensure .env or Api_key.env contains:  OPENAI_API_KEY=sk-xxxxxx
load_dotenv("Api_key.env")

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please add it to Api_key.env.")
else:
    print("OpenAI API key loaded successfully for Gradio app.")

# ==============================================================
# Core SARAL interaction (Chat handler)
# ==============================================================

def saral_chat(query, mode="script", audience="general", top_k=5, history=[]):
    if not query.strip():
        history.append({"role": "assistant", "content": "Please enter a valid query."})
        return history

    try:
        collection_name = get_active_collection()
        if not collection_name:
            history.append({
                "role": "assistant",
                "content": "No active paper found. Please upload a PDF before asking questions."
            })
            return history

        answer, context_used = ask_saral_query(
            query=query,
            top_k=top_k,
            collection_name=collection_name
        )

        provenance = "\n".join(
            [f"Page {c['page_number']} | Math: {c['contains_math']} → {c['sentence_chunk'][:200]}..."
             for c in context_used]
        )

        if history and "Slide" in history[-1].get("content", ""):
            prev = history[-1]["content"]
            delta = "Updated: SARAL refined the script based on your latest prompt." if prev != answer else "✅ No major changes detected."
        else:
            delta = ""

        response = f"{answer}\n\n---\nSource Context\n{provenance}"

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})

    return history

# ==============================================================
# PDF Upload → Extraction → Chunking → Embedding → ChromaDB update
# ==============================================================

def process_pdf(file_obj):
    """
    Takes uploaded PDF, extracts text, chunks, embeds, and stores in ChromaDB dynamically.
    """
    import shutil
    if os.path.exists("./saral_chroma_store"):
        shutil.rmtree("./saral_chroma_store")
        print("Cleared old ChromaDB store.")
    if not file_obj:
        return "Please upload a PDF."

    try:
        # Temporary save location
        pdf_path = file_obj.name  # Gradio already stores it in a temp folder

        print(f"📘 Processing uploaded file: {pdf_path}")
        pages_and_texts = open_and_read_pdf(pdf_path)
        num_pages = len(pages_and_texts)

        # Sentence segmentation
        from spacy.lang.en import English
        nlp = English()
        nlp.add_pipe("sentencizer")

        for item in tqdm(pages_and_texts):
            item["sentences"] = [str(s) for s in nlp(item["text"]).sents]

        # Split sentences into small chunks
        def split_list(input_list, slice_size):
            return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

        pages_and_chunks = []
        for item in pages_and_texts:
            for chunk in split_list(item["sentences"], 10):
                text_chunk = " ".join(chunk).strip()
                text_chunk = clean_math_chunk(text_chunk)
                pages_and_chunks.append({
                    "page_number": item["page_number"],
                    "sentence_chunk": text_chunk,
                    "contains_math": contains_math(text_chunk)
                })

        df = pd.DataFrame(pages_and_chunks)
        print(f"Created {len(df)} chunks from {num_pages} pages.")

        # --- Embed dynamically with OpenAI ---
        model_name = "text-embedding-3-small"

        def batch_list(input_list, batch_size=64):
            for i in range(0, len(input_list), batch_size):
                yield input_list[i:i + batch_size]

        all_embeddings = []
        for batch in tqdm(batch_list(df["sentence_chunk"].tolist())):
            response = openai.embeddings.create(model=model_name, input=batch)
            all_embeddings.extend([r.embedding for r in response.data])

        df["embedding"] = all_embeddings
        print(f"Generated {len(all_embeddings)} embeddings.")
        # --- Update persistent ChromaDB ---
        if os.path.exists("./saral_chroma_store"):
            shutil.rmtree("./saral_chroma_store")
            print("Old ChromaDB store cleared.")

        client = chromadb.Client(chromadb.config.Settings(persist_directory="./saral_chroma_store"))

# Create a unique collection per paper
        collection_name = os.path.splitext(os.path.basename(file_obj.name))[0].replace(" ", "_").lower()
        collection = client.get_or_create_collection(name=collection_name)
        set_active_collection(collection_name)


        print(f" Using Chroma collection: {collection_name}")

        ids = [f"new_chunk_{i}" for i in range(len(df))]
        metadatas = df[["page_number", "contains_math"]].to_dict(orient="records")

        collection.add(
            ids=ids,
            documents=df["sentence_chunk"].tolist(),
            embeddings=np.array(df["embedding"].tolist()),
            metadatas=metadatas,
        )

        print(f"Added {len(df)} new chunks to ChromaDB collection (auto-persisted).")


        return f"Uploaded & processed {file_obj.name} — {num_pages} pages → {len(df)} chunks stored in ChromaDB."

    except Exception as e:
        return f"Error processing file: {str(e)}"

# ==============================================================
# Build Gradio UI
# ==============================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Math Aware RAG System")
    gr.Markdown("""
    SARAL can:
    -  Generate slides and scripts from uploaded research papers
    -  Handle derivations and formulas
    -  Track refinements (“make slide #2 simpler”)
    -  Maintain context between turns
    """)

    # ---- Upload section ----
    with gr.Tab("Upload & Ingest"):
        gr.Markdown("### Upload a new research paper (PDF)\nSARAL will extract, chunk, embed, and add it to ChromaDB automatically.")
        pdf_file = gr.File(label="Upload PDF Paper", file_types=[".pdf"])
        upload_output = gr.Markdown()
        pdf_file.change(fn=process_pdf, inputs=pdf_file, outputs=upload_output)

    # ---- Chat section ----
    with gr.Tab("SARAL Assistant"):
        chatbot = gr.Chatbot(label="SARAL Conversation", type="messages")
        query = gr.Textbox(
            placeholder="e.g., Make a 7-slide talk for graduate students focusing on methods",
            label="Enter your question or refinement request",
            lines=3
        )

        with gr.Row():
            audience = gr.Dropdown(
                ["general", "graduate students", "researchers", "policymakers"],
                value="general",
                label="Audience Type"
            )
            mode = gr.Dropdown(
                ["script", "derivation", "summary"],
                value="script",
                label="Response Mode"
            )
            top_k = gr.Slider(3, 10, value=5, step=1, label="Context Chunks")

        send_btn = gr.Button("Generate / Refine")
        clear_btn = gr.ClearButton(components=[chatbot], value="Clear Chat")

        state = gr.State([])

        send_btn.click(
            fn=saral_chat,
            inputs=[query, mode, audience, top_k, state],
            outputs=[chatbot],
        )

# ==============================================================
# Launch
# ==============================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
