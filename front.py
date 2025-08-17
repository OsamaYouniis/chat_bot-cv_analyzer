import streamlit as st
import os, zipfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

# 🔑 Your Cohere API key (⚠️ remember to rotate later)
COHERE_API_KEY = "v4fXGio56DlgppNtr1YHAgwzDfmujPqsYOIKL4vN"
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

st.title("CV_ANALYZER")

uploaded_files = st.file_uploader("Upload candidate resumes (PDF or ZIP)", type=["pdf", "zip"], accept_multiple_files=True)

# Job description input
job_description = st.text_area("Paste the Job Description", height=200)

if st.button("Analyze Candidates"):
    if not uploaded_files or not job_description.strip():
        st.error("Please upload at least one resume (PDF or ZIP) and enter a job description.")
    else:
        extract_dir = "pdfs"
        os.makedirs(extract_dir, exist_ok=True)

        # Save & extract files
        for uploaded in uploaded_files:
            if uploaded.name.endswith(".zip"):
                with zipfile.ZipFile(uploaded, "r") as z:
                    z.extractall(extract_dir)
            elif uploaded.name.endswith(".pdf"):
                file_path = os.path.join(extract_dir, uploaded.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded.read())

        st.write("Extracted Files:", os.listdir(extract_dir))

        # Load PDFs into documents
        def load_pdf_cvs(folder_path):
            cvs = []
            for filename in os.listdir(folder_path):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(folder_path, filename)
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    combined = "\n".join([p.page_content for p in pages])
                    cvs.append(Document(page_content=combined, metadata={"source": filename}))
            return cvs

        cvs = load_pdf_cvs(extract_dir)


        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )
        split_docs = splitter.split_documents(cvs)

        # Create embeddings & FAISS index
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=COHERE_API_KEY
        )
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        # LLM for reasoning
        llm = Cohere(
            model="command",
            temperature=0.3,
            max_tokens=500
        )

        # Custom recruiter prompt
        hr_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a senior recruiter evaluating candidates for a Call Center Agent position.  
Your task is to analyze how well each candidate matches the given job requirements.  

JOB DESCRIPTION:
{question}

CANDIDATE PROFILES:
{context}

ANALYSIS GUIDELINES:
1. Evaluate each candidate on:
   - Communication skills and fluency  
   - Customer service or support experience  
   - Problem-solving ability and conflict resolution  
   - Technical or CRM tool knowledge (if mentioned)  
2. Provide a short, professional justification for suitability (2–3 sentences max).  
3. Highlight 2–3 key strengths for each candidate.  
4. Only include the **top 3 ranked candidates**, sorted by who is the strongest fit.  
5. Do not use scores — instead, make the ranking based on their strengths and fit.  

OUTPUT FORMAT (strictly follow):
- Candidate: [Full Name]  
  Match Summary: [2–3 sentence evaluation]  
  Strengths: [list of key strengths]  

TOP 3 CANDIDATES (Ranked):  
1. [Name] — Best fit because [reason]  
2. [Name] — Strong fit because [reason]  
3. [Name] — Suitable fit because [reason]  



"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            verbose=True,
            chain_type_kwargs={"prompt": hr_prompt}
        )

        # Run the analysis
        with st.spinner("Analyzing candidates..."):
            result = qa_chain.invoke({"query": job_description})

        st.subheader("Top Candidates Analysis")
        st.write(result["result"])
