try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores.faiss import FAISS



    DATA_PAth = "Data/"
    DB_FAISS_PATH = "vectorstores/db_faiss" # all the mbeddings get stores in this path

    #Create vector Database
    def create_vector_db():
        loader = DirectoryLoader(DATA_PAth, glob='*.pdf', loader_cls = PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', 
                                        model_kwargs = {'device': 'cpu'}) #Run the model in CPU
        
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)

    if __name__ == "__main__":
        create_vector_db()

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._pytree")


except ImportError as e:
    print(f"Error importing module: {e}")
    # Handle the error or provide more information for debugging
