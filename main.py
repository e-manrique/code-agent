import os
# from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Clone
repo_path = "./test_repo"
if not os.path.exists(repo_path):
    print(f"\nCreating repo in {repo_path}")

# Load
loader = GenericLoader.from_filesystem(
    repo_path + "/iching",
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
print(f"\nLoading documents from: {repo_path} ")
documents = loader.load()
print(f"\nthere are: {len(documents)} documents loaded")


python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
print(f"\nSpliting {len(documents)} documents")
texts = python_splitter.split_documents(documents)
print(f"\nSplitted into  {len(texts)} texts")

embedding_model = "codellama:34b"
print(f"\nLoading {embedding_model} as ChromaDB ollama embedding model")
db = Chroma.from_documents(texts, OllamaEmbeddings(model=embedding_model))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    chat_model="codellama:34b"
    print(f"\nUsing {chat_model} Chat model")
    llm = ChatOllama(model=chat_model)


    system_template = """
    Answer the user's questions based on the below context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible:

    {context}
    """

    # First we need a prompt that we can pass into an LLM to generate this search query
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    # Run, only returning the value under the answer key for readability
    response = qa_chain.pick("answer").invoke({"input": query})
    print(f"\n{response}")