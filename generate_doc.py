"""Generate the Agentic RAG explanation PDF."""

from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Agentic RAG with LangGraph - Technical Walkthrough", align="C")
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 70, 130)
        self.cell(0, 10, title)
        self.ln(8)

    def sub_title(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title)
        self.ln(7)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.cell(8, 6, "-")
        self.multi_cell(0, 6, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5.5, text, fill=True)
        self.ln(3)

    def diagram_line(self, text):
        self.set_font("Courier", "", 9)
        self.set_text_color(30, 70, 130)
        self.cell(0, 5, text, align="C")
        self.ln(5)


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# -- Title page --
pdf.ln(30)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(30, 70, 130)
pdf.cell(0, 15, "From Traditional RAG", align="C")
pdf.ln(14)
pdf.cell(0, 15, "to Agentic RAG", align="C")
pdf.ln(20)
pdf.set_font("Helvetica", "", 13)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "A complete walkthrough of the Company Policy Assistant", align="C")
pdf.ln(8)
pdf.cell(0, 10, "Built with ChromaDB, LangGraph, and GitHub Models", align="C")
pdf.ln(30)
pdf.set_font("Helvetica", "I", 10)
pdf.set_text_color(150, 150, 150)
pdf.cell(0, 10, "June 2026", align="C")

# -- Section 1: What is RAG? --
pdf.add_page()
pdf.section_title("1. What is RAG?")

pdf.body_text(
    "RAG stands for Retrieval-Augmented Generation. Instead of asking an AI model to answer "
    "from its general training data (which may be outdated or wrong), we first search our own "
    "documents for relevant information, then hand that information to the AI along with the "
    "question. This way, the AI answers based on YOUR data, not its imagination."
)

pdf.body_text("Think of it like this:")
pdf.bullet("Without RAG: You ask a stranger a question about your company's leave policy. They guess.")
pdf.bullet(
    "With RAG: You hand the stranger your company's policy handbook, highlight the relevant pages, "
    "and THEN ask the question. They answer based on what's actually written."
)

# -- Section 2: How Our Documents Get Ingested --
pdf.section_title("2. How Documents Get Ingested (ingest.py)")

pdf.body_text(
    "Before we can search our documents, we need to prepare them. This is what the ingestion "
    "step does. It runs once, before the app starts."
)

pdf.sub_title("Step 1: Read the Documents")
pdf.body_text(
    "The system reads all files from the documents/ folder. It supports plain text (.txt) "
    "and PDF (.pdf) files. Each file is read and its full text is extracted."
)

pdf.sub_title("Step 2: Split into Chunks")
pdf.body_text(
    "A full document can be thousands of words long. Searching through entire documents is "
    "slow and imprecise. So we split each document into smaller pieces called 'chunks' -- "
    "roughly 500 words each, with a 50-word overlap between consecutive chunks so we don't "
    "lose context at the boundaries."
)

pdf.body_text("Example: A 1500-word document becomes roughly 3-4 chunks.")

pdf.sub_title("Step 3: Convert to Vectors (Embeddings)")
pdf.body_text(
    "Here's the key idea: we convert each text chunk into a list of numbers called a 'vector' "
    "or 'embedding'. This is done by a Sentence Transformer model (BAAI/bge-base-en-v1.5). "
    "These numbers capture the MEANING of the text -- so chunks about similar topics will have "
    "similar numbers, even if they use different words."
)

pdf.sub_title("Step 4: Store in ChromaDB")
pdf.body_text(
    "ChromaDB is a vector database. It stores each chunk along with its vector and metadata "
    "(like which file it came from). Later, when a user asks a question, we convert the question "
    "into a vector too, and ChromaDB finds the chunks whose vectors are closest -- meaning the "
    "chunks most semantically similar to the question."
)

pdf.body_text("The ingested data is saved to disk in the chroma_db/ folder, so it persists across restarts.")

# -- Ingestion Diagram --
pdf.ln(2)
pdf.sub_title("Ingestion Flow")
lines = [
    "documents/                    Chunks                  Vectors",
    " leave_policy.txt    -->  [chunk1, chunk2, ...]  -->  [0.12, -0.34, ...]",
    " remote_work.txt     -->  [chunk1, chunk2, ...]  -->  [0.56, 0.78, ...]",
    " expense_policy.txt  -->  [chunk1, chunk2, ...]  -->  [-0.23, 0.45, ...]",
    "",
    "                    All stored in ChromaDB (chroma_db/)",
]
for line in lines:
    pdf.diagram_line(line)

# -- Section 3: Traditional RAG --
pdf.add_page()
pdf.section_title("3. Traditional RAG -- How It Worked (rag.py)")

pdf.body_text(
    "Our first version used a simple, linear pipeline. When a user types a question in the "
    "Streamlit chat UI, here's exactly what happened:"
)

pdf.sub_title("Step 1: Retrieve")
pdf.body_text(
    "The user's question is converted into a vector (using the same embedding model). "
    "ChromaDB searches for the 5 closest matching chunks. These are our 'context' documents."
)

pdf.sub_title("Step 2: Build Prompt")
pdf.body_text(
    "The retrieved chunks are formatted and injected into a prompt along with the user's "
    "question. A system prompt tells the AI to ONLY answer from the provided context, cite "
    "sources, and never make things up."
)

pdf.sub_title("Step 3: Generate")
pdf.body_text(
    "The prompt is sent to the LLM (GPT-4o-mini via GitHub Models). The AI reads the context "
    "and generates an answer. The response is streamed back to the chat UI token by token."
)

pdf.ln(2)
pdf.sub_title("Traditional RAG Flow")
lines = [
    "User Question",
    "     |",
    "     v",
    "[Retrieve 5 chunks from ChromaDB]",
    "     |",
    "     v",
    "[Build prompt with chunks + question]",
    "     |",
    "     v",
    "[Send to LLM, get answer]",
    "     |",
    "     v",
    "Display answer to user",
]
for line in lines:
    pdf.diagram_line(line)

# -- Section 4: The Problem --
pdf.add_page()
pdf.section_title("4. The Problem with Traditional RAG")

pdf.body_text("The traditional approach works well for simple, direct questions like:")
pdf.bullet('"What is the remote work policy?" -- Easy. The chunks about remote work are retrieved and answered.')
pdf.bullet('"How many sick leaves do I get?" -- Direct match. Works great.')

pdf.body_text("But it struggles with:")

pdf.sub_title("Problem 1: Broad or Vague Queries")
pdf.body_text(
    'A question like "List all dollar amounts in the policies" doesn\'t match any specific '
    "policy topic well. The vector search might return chunks that are only loosely related, "
    "and the AI either gives an incomplete answer or says it doesn't know."
)

pdf.sub_title("Problem 2: No Quality Check")
pdf.body_text(
    "Traditional RAG blindly trusts whatever chunks come back from the search. If the search "
    "returns irrelevant chunks (maybe the question was phrased in an unusual way), the AI tries "
    "to answer from bad context -- leading to wrong or unhelpful answers."
)

pdf.sub_title("Problem 3: One Shot, No Retry")
pdf.body_text(
    "You get one attempt at retrieval. If the wording of the question doesn't match the wording "
    "in the documents, you're stuck with poor results. There's no mechanism to rephrase and try again."
)

pdf.body_text(
    "In short: Traditional RAG is a one-way street. It retrieves once, generates once, and hopes "
    "for the best. There's no intelligence in the retrieval process itself."
)

# -- Section 5: Agentic RAG --
pdf.add_page()
pdf.section_title("5. Agentic RAG -- The Solution (agentic_rag.py)")

pdf.body_text(
    "Agentic RAG adds a brain to the retrieval process. Instead of a straight pipeline, we build "
    "a GRAPH -- a set of steps with decision points where the system can choose what to do next, "
    "including looping back to try again."
)

pdf.body_text("We use LangGraph to define this graph. Here are the four nodes (steps):")

pdf.sub_title("Node 1: Retrieve")
pdf.body_text(
    "Same as before -- query ChromaDB and get the top 5 matching chunks. But now, instead of "
    "going straight to the LLM, the chunks go through a quality check first."
)

pdf.sub_title("Node 2: Grade Documents")
pdf.body_text(
    "This is the key addition. For each retrieved chunk, we ask the LLM: 'Is this document "
    "actually relevant to the user's question?' The LLM evaluates each chunk individually and "
    "returns a yes/no verdict. Only chunks marked as relevant are kept."
)
pdf.body_text(
    "The grader is intentionally lenient -- if a chunk is even partially related, it's kept. "
    "This prevents the system from being too aggressive in filtering."
)

pdf.sub_title("Node 3: Rewrite Query")
pdf.body_text(
    "If the grading step finds that fewer than 1 relevant document was retrieved, something "
    "went wrong with the search. Instead of giving up, the system asks the LLM to REWRITE the "
    "question -- make it more specific, use different keywords, break it down. Then it loops "
    "back to retrieve again with the new query."
)
pdf.body_text("This can happen up to 2 times. After 2 retries, it generates with whatever it has.")

pdf.sub_title("Node 4: Generate")
pdf.body_text(
    "Once we have verified relevant documents, the system builds the prompt (same system prompt "
    "as before) and sends it to the LLM for the final answer. The response is streamed to the UI."
)

# -- Graph Diagram --
pdf.add_page()
pdf.section_title("6. The LangGraph Structure")

pdf.body_text("Here's how the graph is wired together:")
pdf.ln(4)

lines = [
    "                  +------------+",
    "                  |  RETRIEVE  |  <-- Entry point",
    "                  +-----+------+",
    "                        |",
    "                        v",
    "                +---------------+",
    "                | GRADE         |",
    "                | DOCUMENTS     |",
    "                +---+-------+---+",
    "                    |       |",
    "           >= 1 relevant   0 relevant",
    "                    |       |",
    "                    v       v",
    "             +----------+  +---------------+",
    "             | GENERATE |  | REWRITE QUERY |",
    "             +----------+  +-------+-------+",
    "                  |                |",
    "                  v                | (loop back, max 2x)",
    "                 END               |",
    "                        +----------+",
    "                        |",
    "                        v",
    "                  +------------+",
    "                  |  RETRIEVE  |  (retry with rewritten query)",
    "                  +------------+",
]
for line in lines:
    pdf.diagram_line(line)

pdf.ln(4)
pdf.body_text("The conditional edge after grading is the decision point:")
pdf.bullet("If at least 1 relevant document exists --> go to Generate")
pdf.bullet("If 0 relevant documents AND retries < 2 --> go to Rewrite Query")
pdf.bullet("If 0 relevant documents AND retries >= 2 --> go to Generate anyway (best effort)")

# -- Section 7: What Happens on User Input --
pdf.add_page()
pdf.section_title("7. What Happens When You Ask a Question")

pdf.body_text("Let's trace through a real example. You type in the UI:")
pdf.code_block('"Can a part-time intern work remotely while on probation?"')

pdf.sub_title("Step 1: Streamlit captures the input")
pdf.body_text(
    "The app.py file receives your message, adds it to the chat history, and calls "
    "stream_ask() from agentic_rag.py."
)

pdf.sub_title("Step 2: Retrieve")
pdf.body_text(
    "Your question is vectorized and sent to ChromaDB. It returns 5 chunks -- maybe some "
    "about remote work, some about leave policy, some about internship terms."
)

pdf.sub_title("Step 3: Grade Documents")
pdf.body_text(
    "The LLM checks each chunk: 'Is this relevant to part-time interns working remotely "
    "during probation?' Maybe 3 out of 5 pass. The other 2 are discarded."
)

pdf.sub_title("Step 4: Decision")
pdf.body_text(
    "3 relevant documents >= 1 threshold. Decision: go to Generate (no rewrite needed)."
)

pdf.sub_title("Step 5: Generate")
pdf.body_text(
    "The 3 verified chunks are formatted into the prompt. The LLM reads them and generates "
    "a specific answer about remote work eligibility for part-time interns on probation."
)

pdf.sub_title("Step 6: Display")
pdf.body_text(
    "The answer streams into the chat UI. The sidebar shows the agent trace -- you can see "
    "exactly which steps were taken: retrieve, grade (3/5 relevant), generate."
)

pdf.ln(4)
pdf.body_text("Now imagine a trickier query that DOES trigger a rewrite:")
pdf.code_block('"List all dollar amounts mentioned in the company policies"')

pdf.body_text(
    "1. Retrieve: Gets 5 chunks, but they're about general policy topics, not specifically about amounts.\n"
    "2. Grade: The grader finds 0 relevant chunks (none specifically list dollar amounts).\n"
    "3. Decision: 0 relevant docs, retries = 0 --> Rewrite Query.\n"
    "4. Rewrite: LLM rephrases to something like 'expense reimbursement limits and monetary amounts in policies'.\n"
    "5. Retrieve (again): New query gets better chunks -- ones mentioning specific dollar figures.\n"
    "6. Grade (again): 3 out of 5 are relevant.\n"
    "7. Decision: 3 >= 1 --> Generate.\n"
    "8. Generate: Produces a comprehensive answer listing the dollar amounts from the policies."
)

# -- Section 8: Agent Trace --
pdf.add_page()
pdf.section_title("8. The Agent Trace (Sidebar)")

pdf.body_text(
    "Every step the agent takes is logged in a trace list. After each query, the sidebar "
    "shows exactly what happened. This makes the system transparent and debuggable."
)

pdf.body_text("Example trace for a simple query:")
pdf.code_block(
    "retrieve(query='What is the remote work policy?')\n"
    "grade_documents(4/5 relevant)\n"
    "generate(streamed)"
)

pdf.body_text("Example trace for a query that needed rewriting:")
pdf.code_block(
    "retrieve(query='list all dollar amounts')\n"
    "grade_documents(0/5 relevant)\n"
    "rewrite_query('expense reimbursement limits and monetary amounts')\n"
    "retrieve(query='expense reimbursement limits and monetary amounts')\n"
    "grade_documents(3/5 relevant)\n"
    "generate(streamed)"
)

# -- Section 9: Comparison --
pdf.section_title("9. Traditional RAG vs Agentic RAG -- Summary")

pdf.ln(2)
# Table header
pdf.set_font("Helvetica", "B", 10)
pdf.set_fill_color(30, 70, 130)
pdf.set_text_color(255, 255, 255)
pdf.cell(60, 8, "  Aspect", fill=True)
pdf.cell(60, 8, "  Traditional RAG", fill=True)
pdf.cell(60, 8, "  Agentic RAG", fill=True)
pdf.ln()

rows = [
    ("Retrieval", "Single attempt", "Retrieves + retries if needed"),
    ("Quality Check", "None", "LLM grades each document"),
    ("Query Handling", "Uses query as-is", "Rewrites if results are poor"),
    ("Flow", "Linear (one-way)", "Graph with loops"),
    ("Transparency", "Black box", "Full agent trace visible"),
    ("Broad Queries", "Often fails", "Adapts via rewriting"),
    ("Complexity", "Simple, fast", "More LLM calls, smarter"),
]

pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(40, 40, 40)
for i, (aspect, trad, agentic) in enumerate(rows):
    fill = i % 2 == 0
    if fill:
        pdf.set_fill_color(245, 245, 245)
    pdf.cell(60, 7, f"  {aspect}", fill=fill)
    pdf.cell(60, 7, f"  {trad}", fill=fill)
    pdf.cell(60, 7, f"  {agentic}", fill=fill)
    pdf.ln()

# -- Section 10: Tech Stack --
pdf.ln(6)
pdf.section_title("10. Tech Stack")

pdf.bullet("Streamlit -- Chat UI for interacting with the assistant")
pdf.bullet("ChromaDB -- Vector database for storing and searching document embeddings")
pdf.bullet("Sentence Transformers (BAAI/bge-base-en-v1.5) -- Converts text to vectors")
pdf.bullet("OpenAI SDK (GitHub Models / GPT-4o-mini) -- LLM for grading, rewriting, and answering")
pdf.bullet("LangGraph -- Framework for building the agentic graph with nodes, edges, and loops")
pdf.bullet("Python + uv -- Runtime and package management")

pdf.output("Agentic_RAG_Walkthrough.pdf")
print("PDF generated: Agentic_RAG_Walkthrough.pdf")
