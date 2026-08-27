"""
Gradio Web Application for LSPP RAG Chatbot (Levels 1 to 5).
Features:
- Upload any PDF or pick from preloaded arXiv papers (Attention, DPO, PPO).
- Parser switcher: Standard PyPDF vs Table-aware PDFPlumber (Level 2).
- Tunable chunk size, overlap, and top-k retrieval.
- Token streaming into chat UI (Level 3).
- Document and page citations display (Level 4).
- Conversational pronoun resolution (Level 5).
- Side-by-side PDF table extraction inspector.
"""

import os
import gradio as gr
from rag_engine import RAGChatbot, compare_parsers_on_page

# Global RAG instance
bot = RAGChatbot()
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")

# Available sample papers
SAMPLE_PAPERS = {
    "Attention Is All You Need (Vaswani et al.)": os.path.join(SAMPLE_DOCS_DIR, "1706.03762_attention_is_all_you_need.pdf"),
    "Direct Preference Optimization (Rafailov et al.)": os.path.join(SAMPLE_DOCS_DIR, "2305.18290_direct_preference_optimization.pdf"),
    "Proximal Policy Optimization (Schulman et al.)": os.path.join(SAMPLE_DOCS_DIR, "1707.06347_proximal_policy_optimization.pdf"),
}

current_active_doc = {"path": None, "name": None}


def load_document(file_obj, sample_choice, parser_mode, chunk_size, chunk_overlap):
    """Loads either the uploaded file or the selected sample paper."""
    target_path = None
    if file_obj is not None:
        target_path = file_obj.name
    elif sample_choice in SAMPLE_PAPERS:
        target_path = SAMPLE_PAPERS[sample_choice]

    if not target_path or not os.path.exists(target_path):
        return "❌ Error: Please upload a PDF file or choose a valid sample paper.", ""

    info = bot.load_pdf(
        file_path=target_path,
        parser_mode=parser_mode,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        clear_existing=True
    )
    current_active_doc["path"] = target_path
    current_active_doc["name"] = info["document"]

    status_msg = (
        f"✅ **Loaded Document:** `{info['document']}`\n\n"
        f"- **Parser:** `{info['parser_used']}`\n"
        f"- **Pages Processed:** {info['pages']}\n"
        f"- **Chunks Indexed:** {info['chunks_created']}\n"
        f"- **Chunk Size:** {chunk_size} | **Overlap:** {chunk_overlap}"
    )
    return status_msg, f"Active: {info['document']}"


def chat_response(message, history, top_k):
    """Streaming chat response with history and citations."""
    if not message.strip():
        yield history
        return

    if not current_active_doc["path"]:
        # Auto-load default paper if none loaded
        default_paper = SAMPLE_PAPERS["Attention Is All You Need (Vaswani et al.)"]
        if os.path.exists(default_paper):
            bot.load_pdf(default_paper, parser_mode="pdfplumber")
            current_active_doc["path"] = default_paper
            current_active_doc["name"] = "1706.03762_attention_is_all_you_need.pdf"

    # Append user turn
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    for chunk in bot.ask_stream(message, k=int(top_k)):
        history[-1]["content"] = chunk["full_text"]
        yield history


def clear_chat():
    bot.clear_history()
    return []


def run_parser_comparison(page_number):
    """Demonstrates Level 2 table extraction difference on current document."""
    pdf_path = current_active_doc["path"]
    if not pdf_path or not os.path.exists(pdf_path):
        pdf_path = SAMPLE_PAPERS["Attention Is All You Need (Vaswani et al.)"]

    comp = compare_parsers_on_page(pdf_path, page_number=int(page_number))
    return comp["pypdf"], comp["pdfplumber"]


# Gradio UI Definition
custom_css = """
#main-container { max-width: 1200px; margin: auto; }
.chatbot-box { min-height: 520px; }
.stat-box { background: #f8fafc; border-radius: 8px; padding: 12px; border: 1px solid #e2e8f0; }
"""

with gr.Blocks(title="LSPP RAG Chatbot - Research Papers") as demo:
    gr.Markdown("""
    # 📚 LSPP Custom RAG Chatbot (Levels 1 – 6)
    **Bring Your Own PDF · Table Extraction · Token Streaming · Citations · Conversational Memory**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 1. Ingest Document")
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_PAPERS.keys()),
                value="Attention Is All You Need (Vaswani et al.)",
                label="Pick Curated arXiv Paper"
            )
            file_upload = gr.File(
                label="Or Upload Your Own PDF (Level 1)",
                file_types=[".pdf"],
                type="filepath"
            )
            
            with gr.Accordion("⚙️ Ingestion & Retrieval Settings", open=False):
                parser_select = gr.Radio(
                    choices=["pdfplumber", "pypdf"],
                    value="pdfplumber",
                    label="PDF Parser Mode",
                    info="pdfplumber parses tables into Markdown (Level 2). pypdf is standard (Level 1)."
                )
                chunk_slider = gr.Slider(minimum=200, maximum=2500, value=1000, step=50, label="Chunk Size")
                overlap_slider = gr.Slider(minimum=0, maximum=500, value=150, step=25, label="Chunk Overlap")
                top_k_slider = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Top-K Retrieval")

            load_btn = gr.Button("📥 Ingest Document", variant="primary")
            status_output = gr.Markdown("Status: Ready to ingest document.")
            active_badge = gr.Markdown("Active: *None*")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("💬 Chat & Citations"):
                    chatbot = gr.Chatbot(
                        label="RAG Assistant (Streaming + Sources)"
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask a question about the document (e.g. 'What is Multi-Head Attention?')...",
                            label="Your Question",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("🚀 Ask", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🧹 Clear Conversation", variant="secondary")

                with gr.TabItem("🥈 Level 2: Table & Parser Inspector"):
                    gr.Markdown("""
                    ### Compare Extraction: PyPDFLoader vs PDFPlumber (Table-Aware)
                    See how `pdfplumber` detects and converts tables to Markdown compared to standard `pypdf`.
                    """)
                    compare_page_input = gr.Number(value=9, label="Page Number to Inspect (e.g. Page 9 for Table 2)", precision=0)
                    compare_btn = gr.Button("🔍 Compare Parsers On Page")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Standard PyPDFLoader (Level 1)")
                            pypdf_output = gr.Textbox(label="Raw PyPDF Output", lines=15)
                        with gr.Column():
                            gr.Markdown("#### Enhanced PDFPlumber (Level 2 Table-Aware)")
                            pdfplumber_output = gr.Textbox(label="PDFPlumber with Markdown Tables", lines=15)

    # Event bindings
    load_btn.click(
        fn=load_document,
        inputs=[file_upload, sample_dropdown, parser_select, chunk_slider, overlap_slider],
        outputs=[status_output, active_badge]
    )

    send_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot, top_k_slider],
        outputs=[chatbot]
    ).then(lambda: "", None, [msg_input])

    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot, top_k_slider],
        outputs=[chatbot]
    ).then(lambda: "", None, [msg_input])

    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot])

    compare_btn.click(
        fn=run_parser_comparison,
        inputs=[compare_page_input],
        outputs=[pypdf_output, pdfplumber_output]
    )

if __name__ == "__main__":
    warm_theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=warm_theme,
        css=custom_css,
        share=False
    )
