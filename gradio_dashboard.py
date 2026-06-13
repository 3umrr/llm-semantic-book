import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("./chroma_db"):
    db_books = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    
    # Filter and sort books to preserve the similarity order from the vector search
    book_recs = books[books["isbn13"].isin(books_list)].copy()
    book_recs["similarity_rank"] = book_recs["isbn13"].map({isbn: idx for idx, isbn in enumerate(books_list)})
    book_recs.sort_values(by="similarity_rank", inplace=True)


    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def generate_groq_explanation(query: str, top_books: list) -> str:
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert book curator. Explain why the following recommended books are a good match for the user's request: '{query}'. Be concise, engaging, and explain why each book fits the theme in 1-2 sentences. Avoid generic intros."),
            ("human", "Here are the top recommendations:\n\n{books_list}")
        ])
        
        books_str = "\n\n".join(top_books)
        chain = prompt_template | llm
        response = chain.invoke({"query": query, "books_list": books_str})
        return response.content
    except Exception as e:
        return f"Failed to generate AI explanation: {e}\n\nMake sure your GROQ_API_KEY is correctly set in your .env file."


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    print("DEBUG - Recommendations retrieved:", [(row["title"], row["isbn13"]) for _, row in recommendations.iterrows()])
    results = []
    books_metadata = []

    for _, row in recommendations.iterrows():
        description = row["description"] if pd.notna(row["description"]) else "No description available."
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = str(row["authors"]).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = str(row["authors"])

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
        books_metadata.append(f"- **{row['title']}** by {authors_str}: {description}")
        
    # Generate explanation for the top 3 recommended books using Groq
    if len(books_metadata) > 0:
        top_books = books_metadata[:3]
        explanation = generate_groq_explanation(query, top_books)
    else:
        explanation = "No recommendations found. Try adjusting your query or filters."

    return results, explanation

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## Recommendations")
            output_gallery = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)
        with gr.Column(scale=2):
            gr.Markdown("## AI Recommendation Analyst (Groq)")
            output_explanation = gr.Markdown("Enter a search request above to generate AI recommendations and insights.")

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = [output_gallery, output_explanation])


if __name__ == "__main__":
    dashboard.launch(share=True)