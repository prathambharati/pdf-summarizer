import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

# 🔐 Paste your OpenAI API key here
OPENAI_API_KEY = "your-open-api-key"  # <-- Replace this with your actual key

# Sidebar contents
with st.sidebar:
    st.title('💬 PDF Summarizer and Q/A App')
    st.markdown('''
    ## About this application
    You can build your own customized LLM-powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)
    ''')
    add_vertical_space(2)
    st.write("Why drown in papers when your chat buddy can give you the highlights and summary? 📚")
    add_vertical_space(2)
    st.write("Made by Pratham")

def main():
    st.header("Ask About Your PDF 🤷‍♀️💬")

    # Upload file
    pdf = st.file_uploader("Upload your PDF File and Ask Questions", type="pdf")

    if pdf is not None:
        # Read the PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Ask a question
        user_question = st.text_input("Please ask a question about your PDF here:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.subheader("📄 Answer")
            st.write(response)

if __name__ == '__main__':
    main()
