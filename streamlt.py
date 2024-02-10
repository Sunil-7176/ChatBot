import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """ Use the following pieces of information to answer the user's question. If you don't know the answer,
please just say that you don't know the answer, don't try to make up an answer.

Context: {}
Question: {} 

Only return the helpful answer below and nothing else.
Helpful Answer: """

def set_custom_prompt():
    """
    Prompt Template for QA retrieval for each vector store
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens="512",
        temperature="0.2"
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})  # Run the model on CPU
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Streamlit
def main():
    st.title("Conversational AI Bot")

    # Set up the bot
    chain = qa_bot()

    st.sidebar.info("Hi, Bot is ready to answer your questions.")

    user_input = st.text_input("Ask your question:")
    if st.button("Ask"):
        st.info("Bot is processing your question...")
        response = chain({'Query': user_input})
        answer = response['result']
        sources = response["source_documents"]

        if sources:
            answer += f"\nSources:" + str(sources)
        else:
            answer += "\n No sources found"

        st.success(answer)

if __name__ == "__main__":
    main()
