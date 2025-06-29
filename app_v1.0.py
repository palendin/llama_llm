# added streamlit output for k and temperature - done
# converted pdf to docx for better table extraction
# alias mapping for synonyms to recognize the type of processes better

import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredExcelLoader # Add UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings # use this to avoid _languagemodel error. this is the new integrated ackage for vertex ai
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI # For chat models like gemini-pro
# from langchain.retrievers import MultiQueryRetriever # Needed for multi-query (good for document comparison)
# from langchain.retrievers.document_compressors import LLMChainExtractor # Needed for compression
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_loaders import UnstructuredPDFLoader
# from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import Field # Import Field from pydantic
from langchain_core.documents import Document
import time
from dotenv import load_dotenv
from typing import Sequence, List
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
import re
from standardize_conceptual_aliases import ConceptStandardizer
# torch.classes.__path__ = []

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# print(f"Torch version: {torch.__version__}")
# print(f"Streamlit version: {st.__version__}")

# check Load environment variables from .env file
load_dotenv()

# Confirm the credential path is set and valid for vertex AI embeddings.
# Don’t need to pass cred_path directly into the constructor. Instead, you just need to make sure the environment variable GOOGLE_APPLICATION_CREDENTIALS is set before you initialize the embeddings. Google’s Vertex AI SDK (and LangChain’s wrapper) uses Application Default Credentials (ADC), which automatically looks for the credentials in .env file
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print(f"Credential path: {cred_path}")
print(f"File exists: {os.path.exists(cred_path)}")
print('printing project name ',os.getenv("GCLOUD_PROJECT"))

# Ensure GCLOUD_PROJECT is set
if "GCLOUD_PROJECT" not in os.environ:
    os.environ["GCLOUD_PROJECT"] = "vertex-ai-460323"
   
# Initialize embedding model
try:
    # --- ADDED model_name HERE ---
    embeddings = VertexAIEmbeddings(
        project=os.getenv("GCLOUD_PROJECT"),
        model_name="text-embedding-004" # Specify the embedding model name so vertexAIembedding know which modelto use. Text-embedding-004 is Google's latest and recommended embedding model for Vertex AI
    )
    print("Embeddings initialized successfully.")

except Exception as e:
    print(f"Error initializing Embeddings. Error details: {e}")
    import traceback
    traceback.print_exc() # Keep this for detailed errors if it fails again

# Configure Gemini API for gemini
genai.configure(api_key=os.getenv("api_key"))


# Define your document directory
DOCUMENT_PATH = "./documents"
# Define a persistence directory for your Chroma DB
PERSIST_DIRECTORY = "./chroma_db" # You can change this name

TEMPORARY_DIRECTORY = "./temp"

# Define a prompt that preserves the default behavior but adds product recognition
product_prompt = PromptTemplate.from_template(
    
    """
    {query}
    
    The following are product names and aliases: MK3, SNX4, 501 (BI595501), 091 (BI754091), SN3 (SN3-01), 049 (BI765049), NVS4, MB1 (MB1-01). If a product name from this list is mentioned in the query, respond using information related to that product. If not, clearly state that no information is available for that product.

    Only if the query explicitly includes the phrase "material number", then extract and display both the Material Number and the Legacy Material Number for each relevant product. Otherwise, do not provide material numbers.
    """
)


alias_map = {
                "Affinity Chromatography": ['protein A chromatography','proteinA','proA', 'affinity chrom', "protein A", "Affinity", "Mab",'MabSelect', 'MabSure',"KanCapA"],
                "Viral Inactivation": ["VI",'viral inact','VI inactivation','VI inact'],
                "Depth Filtration": ["DP",'depth filt'],
                "Anionic Exchange Chromatography": ['capto adhere','capto',"AEX", "Poros HQ",'anionic exch','anionic echange','anion exchange','anion exch'],
                "Cationic Exchange Chromatography": ["CEX", 'fractogel','fracto', "Poros XS", "Poros HS",'cationic exch','cationic exchange''cation exchange','cation exch'],
                "Viral Filtration": ["VF", "Nano",'viral filt','nanofiltration','nano filt'],
                "Ultrafiltration/Diafiltration": ["TFF", "UF/DF",'UFDF'],
                "ultrafiltration": ["UF"],
                "diafiltration": ["DF"],
                "BDS": ["DS", "Bulk Filtration", "Formulation", "Freeze",'drug substance','bulk filtration','bulk filt']
            }
standardizer = ConceptStandardizer(alias_map)


# This decorator is used to cache the result of a function. This is useful for functions that take a long time to run, and you want to avoid running them multiple times. 
# When you use the @st.cache_resource decorator, Streamlit will cache the result of the function, and the next time you call the function, it will return the cached result instead of running the function again.
@st.cache_resource
def load_and_index_documents(path, persist_directory):
    """Loads documents, chunks them, and creates a vector store."""
    # loader = DirectoryLoader(path, glob="**/*.txt")  # Adjust glob for your file types
    # documents = loader.load()

    # Check if the database already exists and is populated
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing Chroma database from {persist_directory}...")
        # Ensure 'embeddings' is defined before this point (it is, globally)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Database loaded successfully.")
        return db
    else:
        print(f"Creating new Chroma database in {persist_directory}...")

        # Loads documents (txt, pdf, docx), chunks them, and creates a vector store
        all_documents = []

        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0] # get the file name to obtain the product name
                print(file_name_without_ext)

                current_file_docs = []

                try: 
                    if file.endswith(".txt"): 
                        loader = TextLoader(file_path) 
                        current_file_docs = loader.load() # initial load from one file, goal is to load all the raw document object from one specific file in directory. Then returns a new list call "current_file_docs"
                    elif file.endswith(".pdf"): 
                        try:
                        # Use UnstructuredPDFLoader here
                        # mode="elements" attempts to identify distinct elements like tables, paragraphs, titles
                        # It will often convert tables into markdown or other structured text.
                            loader = UnstructuredPDFLoader(file_path, mode="elements")
                            current_file_docs = loader.load() # typically the loader returns mulitple document object since it may have multiple pages. mode="elements" can further break a single page into many document object,s where each object represents heading, title, paragraph, etc
                        except Exception as e:
                            # Fallback if unstructured is not fully set up or fails
                            st.warning(f"Warning: UnstructuredPDFLoader failed for {file_path}. Error: {e}")
                            st.info(f"Attempting to load {file_path} with PyPDFLoader as a fallback.")
                            try:
                                loader = PyPDFLoader(file_path)
                                current_file_docs = loader.load()
                            except Exception as e_fallback:
                                st.error(f"Error loading PDF file {file_path} with PyPDFLoader fallback: {e_fallback}")
                        # loader = PyPDFLoader(file_path) 
                    elif file.endswith(".docx"): 
                        loader = Docx2txtLoader(file_path) 
                        current_file_docs = loader.load()
                    elif file.endswith((".xls", ".xlsx")):
                        # convert each sheet in excel file to csv and then load as a dataframe
                        df = pd.read_excel(file_path,sheet_name=None,engine='openpyxl')

                        for sheet_name, sheet_df in df.items():
                            csv_file_name = os.path.splitext(file_name_without_ext)[0] + "_" + sheet_name + ".csv"
                            processed_file_path = os.path.join(TEMPORARY_DIRECTORY, csv_file_name)
                            sheet_df.to_csv(processed_file_path, index=False, encoding='utf-8-sig')
                            loader = CSVLoader(processed_file_path, encoding='utf-8-sig')
                            current_file_docs = loader.load()
                            # loader = UnstructuredExcelLoader(file_path, mode="paged")
                            # current_file_docs = loader.load()
                            print(current_file_docs[0].page_content)
                            print("------------------------------------------------------------------")
                            print(current_file_docs[0].metadata)
                    elif file.endswith(".csv"):
                        loader = CSVLoader(file_path,encoding='utf-8-sig')
                        current_file_docs = loader.load()
                    else: 
                        continue 

                    # --- view loaded doc contents ---
                    # print(f"Loaded {len(current_file_docs)} docs from {file_path}")
                    # for doc in current_file_docs:
                    #     print(doc.page_content[:20]) 

                    # --- METADATA NORMALIZATION STEP ---
                    normalized_docs = []

                    # loop through each document object in the current_file_docs list
                    for doc in current_file_docs:
                        normalized_metadata = {}
                        for key, value in doc.metadata.items(): # doc.metadata is a dictionary of metadata for each document object, its populated automatically by loader
                            if isinstance(value, list):
                                # Convert list to a comma-separated string, or just take the first item if sensible
                                # For 'languages': ['eng'] -> 'eng'
                                # For other lists, you might need a different strategy or omit them
                                if key == 'languages' and value:
                                    normalized_metadata[key] = value[0] # Take the first language code
                                elif key == 'filetype' and value: # Unstructured sometimes gives filetype as a list
                                    normalized_metadata[key] = value[0]
                                else:
                                    # For other lists, convert to string or handle as needed
                                    normalized_metadata[key] = str(value)
                            elif isinstance(value, (str, int, float, bool)) or value is None:
                                normalized_metadata[key] = value
                            else:
                                # For any other complex object types in metadata, convert to string
                                normalized_metadata[key] = str(value)
                        
                        # reassigning the cleaned up version of the metadata to the doc.metadata 
                        doc.metadata = normalized_metadata
                        
                        doc.metadata['document_title'] = file_name_without_ext
                        # CRITICAL ADDITION: Prepend source/file name to the content. This makes the LLM aware of the source of the information within the chunk.
                        # this will be modified, add to the chroma vector store, and call in the query function
                        # doc.page_content = f"Document Source: {file_name_without_ext}\n" + doc.page_content
                        # print('loading function', doc.page_content)
                        
                        # You might also add page number if 'page' is consistently available in metadata:
                        # if 'page' in doc.metadata:
                        #    doc.page_content = f"Document Source: {file_name_without_ext}, Page: {doc.metadata['page']}\n" + doc.page_content

                        normalized_docs.append(doc)
                    
                    all_documents.extend(normalized_docs) # Add normalized docs to the main list
                    
                except Exception as e: 
                    st.warning(f"Error loading {file_path}: {e}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        texts = text_splitter.split_documents(all_documents)
        
        # Create the database and persist it
        #db = Chroma.from_documents(texts, embeddings)
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        db.persist() # Explicitly persist the database to disk
        print("Database created and persisted successfully.")

        return db


class PreserveTitleCompressor(BaseDocumentCompressor):
    base_compressor: BaseDocumentCompressor = Field(...)

    def compress_documents(self, documents: list[Document], query: str, callbacks=None) -> list[Document]:
        print(f"\n--- PreserveTitleCompressor compress_documents START ---")
        
        # 1. Debug prints for incoming documents (keep this as initial sanity check)
        # print(f"  Incoming to PreserveTitleCompressor ({len(documents)} docs):")
        # for i, doc in enumerate(documents):
        #     print(f"    Input Doc {i+1}: Source='{doc.metadata.get('source')}', Title='{doc.metadata.get('document_title')}', Page='{doc.metadata.get('page')}'")
        #     print(f"      Content start (before base compressor): '{doc.page_content[:100]}'")

        # 2. Call the base compressor (LLMChainExtractor)
        compressed_docs_from_base = self.base_compressor.compress_documents(documents, query, callbacks=callbacks)
        
        # print(f"\n--- After Base Compressor (LLMChainExtractor) output ---")
        # print(f"  Base compressor returned {len(compressed_docs_from_base)} documents.")
        
        final_processed_docs = []
        
        for i, doc_from_base_compressor in enumerate(compressed_docs_from_base):
            # # NEW DEBUG PRINTS HERE - EXTREMELY IMPORTANT
            # print(f"\n  DEBUG: Document {i+1} *immediately after* LLMChainExtractor (before PreserveTitleCompressor adds prefix):")
            # print(f"    Metadata on this doc: Source='{doc_from_base_compressor.metadata.get('source', 'N/A')}', Title='{doc_from_base_compressor.metadata.get('document_title', 'N/A')}', Page='{doc_from_base_compressor.metadata.get('page', 'N/A')}'")
            # print(f"    Content from base compressor (first 500 chars): '{doc_from_base_compressor.page_content[:500]}'")
            # print(f"    Content from base compressor (last 200 chars): '...{doc_from_base_compressor.page_content[-200:]}'")
            # print("-" * 70) # Separator for clarity
            # # END NEW DEBUG PRINTS

            # Check for empty content, if LLMChainExtractor returns 'NO_OUTPUT' or empty string
            if not doc_from_base_compressor.page_content or doc_from_base_compressor.page_content.strip().upper() == "NO_OUTPUT":
                print(f"    Skipping empty/NO_OUTPUT document.")
                continue
            
            
            # alias mapping, replace product names with aliases
            original_content = doc_from_base_compressor.page_content
            processed_content = original_content

            processed_content = standardizer.standardize(original_content) 
            doc_from_base_compressor.page_content = processed_content



            # Now, perform your prefixing logic
            doc_title_for_prefix = doc_from_base_compressor.metadata.get("document_title", "Unknown Document (Missing Title Metadata)")
            
            prefix = f"Document Source: {doc_title_for_prefix}\n"
            
            # This check is critical: is the content *already* starting with a source string?
            if not doc_from_base_compressor.page_content.strip().lower().startswith(prefix.strip().lower()):
                # print(f"    ACTION: Adding prefix '{prefix.strip()[:50]}...' to content.")
                doc_from_base_compressor.page_content = prefix + doc_from_base_compressor.page_content
            else:
                print(f"    (Prefix similar to '{prefix.strip()[:50]}...' already present. Not adding.)")
            
            final_processed_docs.append(doc_from_base_compressor)

        print(f"\n--- PreserveTitleCompressor compress_documents END ---")
        return final_processed_docs

# no compression class
class NoOpCompressor(BaseDocumentCompressor):
        """A compressor that simply returns the documents as is."""
        
        def compress_documents(self, documents: Sequence[Document], query: str, callbacks=None) -> List[Document]: # converts document into list and returns them unchanged
            # this overrides compress_documents() method from BaseDocumentCompressor
            return list(documents)



# multi query retriever
def query_gemini_with_context(db, query, k,temperature):
    """Retrieves relevant documents and queries Gemini."""
        
    total_start_time = time.time()

    #model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5) # Add temperature if desire. Low temperature is more deterministic, high temperature is more creative
    GenAImodel = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=temperature)
    
    # # 1. Create your base compressor (e.g., LLMChainExtractor) for actual compression to extract relevant sentence
    # LLMchainextractor_start_time = time.time()
    # base_compressor_instance = LLMChainExtractor.from_llm(model)
    # LLMchainextractor_end_time = time.time()
    # print(f"LLMChainExtractor time: {LLMchainextractor_end_time - LLMchainextractor_start_time} seconds")


    # # 2. instantiate the custom compressor, passing the LLMChainExtractor as the base_compressor
    # custom_compressor_start_time = time.time()
    # custom_compressor_instance = PreserveTitleCompressor(base_compressor=base_compressor_instance)
    # custom_compressor_end_time = time.time()
    # print(f"Custom Compressor time: {custom_compressor_end_time - custom_compressor_start_time} seconds")

    # # 2. using passthrough as base compressor
    custom_compressor_start_time = time.time()
    base_compressor_for_wrapper = NoOpCompressor()
    custom_compressor_instance = PreserveTitleCompressor(base_compressor=base_compressor_for_wrapper)
    custom_compressor_end_time = time.time()
    print(f"Custom Compressor time: {custom_compressor_end_time - custom_compressor_start_time} seconds")

     # 3. Use your custom compressor as the base_compressor for ContextualCompressionRetriever
    retriever_setup_start_time = time.time()
    retriever_base = db.as_retriever(search_kwargs={"k": k}) # Base k for initial retrieval
    
    # multi_query_retriever = MultiQueryRetriever.from_llm(
    #     retriever=retriever_base, llm=model
    # )
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=custom_compressor_instance, # <-- Pass your custom compressor here
    #     base_retriever=multi_query_retriever
    # )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=custom_compressor_instance, # <-- Pass your custom compressor here
        base_retriever=retriever_base) # <-- Pass your base retriever here

    retriever_setup_end_time = time.time()
    print(f"Retriever setup time: {retriever_setup_end_time - retriever_setup_start_time} seconds")

    # # initialize similarity score model
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Load embedding model
    # query_embedding = embedding_model.encode("material number")  # Encode the target concept
  
    # 4. retrieve relevant info based query using retriever
    retrieval_start_time = time.time()
    retrieved_docs = compression_retriever.get_relevant_documents(query)

    # Apply semantic ranking only if retrieval returns results
    # if retrieved_docs:
    #     query_embedding = embedding_model.encode(query)  # Encode user's query
    #     doc_embeddings = [embedding_model.encode(doc.page_content) for doc in retrieved_docs]  # Encode retrieved documents

    #     # Compute similarity scores
    #     similarities = [util.pytorch_cos_sim(query_embedding, doc_emb) for doc_emb in doc_embeddings]

    #     # Sort documents by similarity, x[0] gets the similarity score, x[1] gets the document
    #     sorted_docs = [doc for sim, doc in sorted(zip(similarities, retrieved_docs), key=lambda x: x[0], reverse=True)]
    # else:
    #     sorted_docs = []  # Avoid passing empty lists into LLM

    retrieval_end_time = time.time()
    print(f"Retrieval time: {retrieval_end_time - retrieval_start_time} seconds")

    # Display the retrieved documents in a table for troubleshooting
    if not retrieved_docs:
        st.warning("No documents were retrieved for this query.")
    else:
        # Create an empty list to store data for each row (each document)
        table_data_rows = []
        
        print("\n--- Final Streamlit Display Loop Debug ---")
        for i, doc in enumerate(retrieved_docs):

            source_info = doc.metadata.get('source', 'N/A')
            page_info = doc.metadata.get('page', 'N/A')
            content_snippet = doc.page_content[:500] # Capture a longer snippet for the table

            table_data_rows.append({
                "Document Index": i + 1,
                "Source": source_info,
                "Page": page_info,
                "Content Snippet": content_snippet
            })

        print("--- End Final Streamlit Display Loop Debug ---\n")
    
        # Create a pandas DataFrame from the list of dictionaries
        df = pd.DataFrame(table_data_rows)
        
        # Display the DataFrame as a table in Streamlit
        st.dataframe(df, use_container_width=True)


    # create a chain for question-answer model that generates answer base on retrieve documents
    qa_chain_instantiation_start_time = time.time()
    qa = RetrievalQA.from_llm(llm=GenAImodel, retriever=compression_retriever, return_source_documents=True)
    qa_chain_instantiation_end_time = time.time()
    print(f"QA Chain Instantiation time: {qa_chain_instantiation_end_time - qa_chain_instantiation_start_time} seconds")

    # If total_tokens is consistently much less than the LLM's context window (e.g., &lt; 100,000 for 1.5 Pro's 1M limit), then the issue is definitely retrieval quality, not the LLM's context limit
    total_tokens = sum(len(doc.page_content.split()) for doc in retrieved_docs) # rough word count as token estimate
    print(f"Approximate total tokens in retrieved docs: {total_tokens}")


    # triggers the actual query to Gemini model
    response_generation_start_time = time.time()
    formatted_query = product_prompt.format(query=query)
    result = qa.invoke({"query": formatted_query}) #, "documents": sorted_docs})
    #result = qa.invoke({"query": query})
    response_generation_end_time = time.time()
    print(f"Response generation time: {response_generation_end_time - response_generation_start_time} seconds")

    total_end_time = time.time()
    print(f"Total query function execution time: {total_end_time - total_start_time:.2f} seconds")

    return result["result"], result["source_documents"]


# Streamlit UI
st.title("Gemini-Powered Process Document Knowledge")

query = st.text_input("Enter your query:")

# Add sliders for temperature and k
k_value = st.slider("Number of Retrieved Documents (k)", min_value=1, max_value=2000, value=250)
temperature = st.slider("Temperature", min_value=float(0.1), max_value=float(1), value=float(0.3))
vector_store = load_and_index_documents(DOCUMENT_PATH, PERSIST_DIRECTORY)
st.write("Documents loaded and indexed successfully.")
if st.button("Submit"):
    # Query Gemini with context
    if query:
        answer, sources = query_gemini_with_context(vector_store, query, k=k_value,temperature=temperature)
        st.subheader("Response:")
        st.write(answer)
        st.subheader("Sources:")
        for doc in sources:
            st.write(f"- {doc.metadata['source']}")
    else:
        st.warning("Please enter a query.")

# run this app using python -m streamlit run app.py in terminal