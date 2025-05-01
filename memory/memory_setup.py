from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from langchain.tools import StructuredTool

from pydantic import BaseModel

# # Inicializamos la base de datos de Chroma
# db = chromadb.Client()
# chroma_collection = db.get_or_create_collection("long_term_memory")

# # Creamos el vector store
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# # Creamos o cargamos el StorageContext
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# # Cargamos el índice si existe o creamos uno vacío
# try:
#     index = load_index_from_storage(storage_context)
# except Exception:
#     index = VectorStoreIndex.from_documents([], storage_context=storage_context)

#-----------------------------------------------------------
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.schema import MetadataMode

# Inicializamos la base de datos de Chroma
db = chromadb.Client()
chroma_collection = db.get_or_create_collection(
    name="long_term_memory",
    metadata={"hnsw:space": "cosine"}  # Configuración opcional para optimizar vectores
)

# Ahora configuramos la Vector Store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
    metadata_mode=MetadataMode.LLM,  # ¡Esta es la clave para que metadatos sean accesibles luego!
)

# Creamos o cargamos el StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Cargamos o inicializamos el índice
try:
    index = load_index_from_storage(storage_context)
except Exception:
    index = VectorStoreIndex.from_documents([], storage_context=storage_context)



#-----------------------------------------------------------
# Funcion que guarda en la memoria largo plazo
class LongMemory(BaseModel):
    memoria: str
    user_name: str

# Función para guardar en memoria persistente
def save_long_term_memory(memoria: str, user_name: str)->bool:
    """Guarda un recuerdo en la memoria persistente asociado a un usuario específico."""

    doc = Document(
        text=memoria,
        metadata={"user_name": f"{user_name}"}
        #metadata={"user_name": "Luis"}
    )
    index.insert(doc)
    index.storage_context.persist(persist_dir="./almacenamiento")
    
    print(f"✅ Memoria guardada para usuario {user_name}: {memoria}")
    return True

save_long_term_memory_tool = StructuredTool(
    name="guardar_memoria",
    description="hace embbeding y almacena en una vector store recuerdos que consideres importantes para interactuar con el usuario. Los inputs son memoria:str ; user_name:str",
    func=save_long_term_memory,
    args_schema=LongMemory
)


def retrieve_relevant_memories(query: str, user_name: str, top_k: int = 3) -> list:
    """Recupera recuerdos relevantes de la memoria persistente, filtrados por usuario."""
    retriever = index.as_retriever(
        #filters={"user_name": user_name},  # Filtra solo recuerdos asociados a ese usuario
        similarity_top_k=top_k
    )
    nodes = retriever.retrieve(query)
    retrieved_memories = [n.node.get_content() for n in nodes]

     # Elegancia: si no hay recuerdos, igual pasamos un aviso
    if not retrieved_memories:
        retrieved_memories = ["No hay recuerdos relevantes disponibles."]

    return retrieved_memories



# from llama_index.core.schema import MetadataFilter, MetadataFilters

# def retrieve_relevant_memories(query: str, user_name: str, top_k: int = 3) -> list:
#     """Recupera recuerdos relevantes de la memoria persistente, filtrados por usuario."""
    
#     filters = MetadataFilters(
#         filters=[
#             MetadataFilter(key="user_name", value=user_name)
#         ]
#     )

#     retriever = index.as_retriever(
#         filters=filters,  # ahora sí bien armado
#         similarity_top_k=top_k
#     )

#     nodes = retriever.retrieve(query)
#     memories = [n.node.get_content() for n in nodes]
#     return memories