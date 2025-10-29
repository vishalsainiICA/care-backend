import os
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import json
from datetime import datetime
import uuid
import chromadb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain v0.2+ modular imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a local cache folder to ensure the model is saved consistently
MODEL_CACHE_PATH = '.model_cache'
VECTOR_DB_PATH = 'clinical_db'
COLLECTION_NAME = 'diseases'

# Create a directory for usage tracking if it doesn't exist
USAGE_TRACKING_DIR = "usage_tracking"
os.makedirs(USAGE_TRACKING_DIR, exist_ok=True)

# --- APPLICATION STATE ---
# Using a simple dictionary for app state as recommended for FastAPI
app_state = {}

# --- ONLINE RESOURCES FUNCTION ---
def search_online_resources(disease_name, max_results=5):
    """
    Search for online resources related to a disease using LangChain's search tools.
    Returns a list of dictionaries with title, url, and snippet.
    """
    try:
        # Initialize LangChain search tools
        search = DuckDuckGoSearchRun()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        # Create a list of tools
        tools = [
            Tool(
                name="Web Search",
                func=search.run,
                description="Search the web for information about medical conditions"
            ),
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Search Wikipedia for medical information"
            )
        ]
        
        # Create a search query focused on medical information
        query = f"{disease_name} clinical guidelines treatment diagnosis symptoms"
        
        # Use the search tool to get results
        search_results = search.run(query)
        
        # Use Wikipedia to get additional information
        wiki_results = wikipedia.run(disease_name)
        
        # Process the results
        results = []
        
        # Process web search results
        if search_results:
            # Split the results into lines and process each line
            lines = search_results.split('\n')
            for line in lines[:max_results]:
                if line.strip():
                    # Try to extract URL and title from the result
                    if 'http' in line:
                        parts = line.split('http')
                        if len(parts) > 1:
                            title = parts[0].strip()
                            url = 'http' + parts[1].split()[0]
                            snippet = line.strip()
                            
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet
                            })
        
        # Add Wikipedia result if available
        if wiki_results and len(results) < max_results:
            # Extract the first paragraph from Wikipedia
            lines = wiki_results.split('\n')
            if lines:
                snippet = lines[0].strip()
                if snippet:
                    # Create a Wikipedia URL
                    wiki_url = f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}"
                    
                    results.append({
                        "title": f"{disease_name} - Wikipedia",
                        "url": wiki_url,
                        "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet
                    })
        
        # If we still don't have enough results, add fallback resources
        if len(results) < max_results:
            fallback_results = create_fallback_resources(disease_name)
            results.extend(fallback_results[:max_results - len(results)])
        
        return results[:max_results]
        
    except Exception as e:
        logger.error(f"Error searching with LangChain tools: {e}")
        # Return fallback resources if search fails
        return create_fallback_resources(disease_name)

def create_fallback_resources(disease_name):
    """Create fallback resources when web scraping fails."""
    return [
        {
            "title": f"{disease_name} - Mayo Clinic",
            "url": f"https://www.mayoclinic.org/search/search-results?q={disease_name.replace(' ', '%20')}",
            "snippet": f"Find comprehensive information about {disease_name} including symptoms, causes, diagnosis, and treatment options from Mayo Clinic."
        },
        {
            "title": f"{disease_name} - MedlinePlus",
            "url": f"https://medlineplus.gov/search?query={disease_name.replace(' ', '%20')}",
            "snippet": f"Learn about {disease_name} from the National Library of Medicine's trusted health information resource."
        },
        {
            "title": f"{disease_name} - NIH",
            "url": f"https://www.nih.gov/search-results?term={disease_name.replace(' ', '%20')}",
            "snippet": f"Explore research and clinical information about {disease_name} from the National Institutes of Health."
        },
        {
            "title": f"{disease_name} - WebMD",
            "url": f"https://www.webmd.com/search/search_results/default.aspx?query={disease_name.replace(' ', '%20')}",
            "snippet": f"Get information about {disease_name} symptoms, treatments, medications, and more from WebMD."
        },
        {
            "title": f"{disease_name} - Healthline",
            "url": f"https://www.healthline.com/search?q={disease_name.replace(' ', '%20')}",
            "snippet": f"Find evidence-based information about {disease_name} including causes, symptoms, diagnosis, and treatment options."
        }
    ]

# --- USAGE TRACKING FUNCTIONS ---
def track_document_usage(retrieved_docs: List[Any], query_id: str):
    """Track which documents were retrieved for a specific query."""
    usage_data = {
        "query_id": query_id,
        "timestamp": datetime.now().isoformat(),
        "retrieved_documents": []
    }
    
    for doc in retrieved_docs:
        doc_data = {
            "id": doc.metadata.get("id", "unknown"),
            "disease": doc.metadata.get("disease", "unknown"),
            "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        }
        usage_data["retrieved_documents"].append(doc_data)
    
    # Save to file
    with open(f"{USAGE_TRACKING_DIR}/{query_id}.json", "w") as f:
        json.dump(usage_data, f, indent=2)
    
    return usage_data

def get_usage_statistics():
    """Analyze usage statistics from all tracked queries."""
    usage_files = [f for f in os.listdir(USAGE_TRACKING_DIR) if f.endswith(".json")]
    
    if not usage_files:
        return {"error": "No usage data available"}
    
    # Track document usage frequency
    doc_usage_count = {}
    disease_usage_count = {}
    total_queries = len(usage_files)
    
    for file in usage_files:
        with open(f"{USAGE_TRACKING_DIR}/{file}", "r") as f:
            data = json.load(f)
            
            for doc in data["retrieved_documents"]:
                doc_id = doc["id"]
                disease = doc["disease"]
                
                doc_usage_count[doc_id] = doc_usage_count.get(doc_id, 0) + 1
                disease_usage_count[disease] = disease_usage_count.get(disease, 0) + 1
    
    # Calculate percentages
    doc_usage_percentage = {k: (v / total_queries) * 100 for k, v in doc_usage_count.items()}
    disease_usage_percentage = {k: (v / total_queries) * 100 for k, v in disease_usage_count.items()}
    
    # Get total documents in the knowledge base
    try:
        vector_store = app_state.get("vector_store")
        if vector_store:
            total_docs = vector_store._collection.count()
        else:
            total_docs = 0
    except:
        total_docs = 0
    
    # Calculate coverage
    used_docs = len(doc_usage_count)
    coverage_percentage = (used_docs / total_docs) * 100 if total_docs > 0 else 0
    
    return {
        "total_queries": total_queries,
        "total_documents_in_kb": total_docs,
        "unique_documents_used": used_docs,
        "coverage_percentage": coverage_percentage,
        "document_usage_frequency": doc_usage_count,
        "document_usage_percentage": doc_usage_percentage,
        "disease_usage_frequency": disease_usage_count,
        "disease_usage_percentage": disease_usage_percentage
    }

# --- FASTAPI LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI app."""
    logger.info("Starting C.A.R.E. Backend with modern LangChain...")
    
    # Initialize components on startup
    try:
        # 1. Initialize the LLM, explicitly passing the API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0.2, 
            google_api_key=gemini_api_key
        )

        # 2. Initialize the embedding model with the same cache folder used in ingestion
        logger.info(f"Loading embedding model from cache: {MODEL_CACHE_PATH}")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                cache_folder=MODEL_CACHE_PATH,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully from cache.")
        except Exception as e:
            logger.error(f"Failed to load embedding model from cache: {e}")
            # Fallback to a different approach if cache loading fails
            logger.info("Attempting to load embedding model without cache...")
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        # 3. Connect to the existing vector store with the exact same configuration
        logger.info(f"Connecting to vector store at: {VECTOR_DB_PATH}")
        
        # First, try to connect directly to ChromaDB to verify the collection exists
        try:
            client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            collection = client.get_collection(name=COLLECTION_NAME)
            doc_count = collection.count()
            logger.info(f"Direct ChromaDB connection shows {doc_count} documents")
            
            if doc_count == 0:
                logger.error("Direct ChromaDB connection shows 0 documents! There's a serious issue with the vector store.")
            else:
                logger.info(f"Successfully connected to ChromaDB with {doc_count} documents")
        except Exception as e:
            logger.error(f"Error connecting directly to ChromaDB: {e}")
            raise
        
        # Now create the LangChain wrapper
        vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # Double-check the document count through the LangChain wrapper
        try:
            doc_count = vector_store._collection.count()
            logger.info(f"LangChain Chroma wrapper shows {doc_count} documents")
            
            if doc_count == 0:
                logger.error("LangChain Chroma wrapper shows 0 documents! There's a mismatch with the direct connection.")
        except Exception as e:
            logger.error(f"Error checking document count through LangChain wrapper: {e}")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Store vector_store for usage statistics
        app_state["vector_store"] = vector_store

        # 4. Define the improved RAG prompt template
        template = """
        You are a highly specialized medical AI, C.A.R.E. Your primary function is to analyze clinical reports and provide a comprehensive, evidence-based diagnosis in JSON format.

        CRITICAL INSTRUCTIONS:
        1. Use the provided "KNOWLEDGE BASE CONTEXT" to ground your answer.
        2. Synthesize information from the knowledge base, avoiding direct copying of fragmented phrases.
        3. Create complete, coherent sentences and well-structured paragraphs.
        4. Eliminate redundancy and repetition in your response.
        5. Organize information in a clinically relevant manner.
        6. Ensure all lists contain distinct, meaningful items without duplication.
        7. If the context is missing information, use your medical knowledge to fill gaps.

        Your response should include:
        1. A primary diagnosis with the highest confidence
        2. Up to 3 differential diagnoses with lower confidence scores
        3. For each diagnosis, provide comprehensive information including symptoms, diagnosis methods, tests, treatment, and prevention

        KNOWLEDGE BASE CONTEXT:
        {context}

        PATIENT REPORT:
        {report}

        JSON OUTPUT:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 5. Define the improved JSON output schema
        output_schema = {
            "title": "ClinicalAnalysis",
            "description": "Comprehensive analysis of a clinical report with disease information.",
            "type": "object",
            "properties": {
                "primary_diagnosis": {
                    "type": "object",
                    "properties": {
                        "disease_name": {"type": "string"},
                        "confidence_score": {"type": "number", "description": "A score from 0.0 to 1.0"},
                        "summary": {"type": "string", "description": "A concise, well-written paragraph describing the condition"},
                        "clinical_presentation": {
                            "type": "object",
                            "properties": {
                                "common_symptoms": {
                                    "type": "array", 
                                    "items": {"type": "string"},
                                    "description": "List of distinct symptoms without repetition"
                                },
                                "key_findings": {
                                    "type": "array", 
                                    "items": {"type": "string"},
                                    "description": "Key clinical findings relevant to diagnosis"
                                }
                            }
                        },
                        "diagnostic_approach": {
                            "type": "object",
                            "properties": {
                                "initial_tests": {
                                    "type": "array", 
                                    "items": {"type": "string"},
                                    "description": "First-line diagnostic tests"
                                },
                                "confirmatory_tests": {
                                    "type": "array", 
                                    "items": {"type": "string"},
                                    "description": "Tests to confirm the diagnosis"
                                }
                            }
                        },
                        "treatment_plan": {
                            "type": "object",
                            "properties": {
                                "first_line": {
                                    "type": "array", 
                                    "items": {"type": "string"},
                                    "description": "Initial treatment approaches"
                                },
                                "advanced_options": {
                                    "type": "array", 
                                    "items": {"type": "string"},
                                    "description": "Advanced or surgical options if needed"
                                }
                            }
                        },
                        "prevention_strategies": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Evidence-based prevention strategies"
                        },
                        "relevance_explanation": {
                            "type": "string", 
                            "description": "A clear explanation of why this diagnosis fits the patient presentation"
                        }
                    },
                    "required": ["disease_name", "confidence_score", "summary", "clinical_presentation", "diagnostic_approach", "treatment_plan", "prevention_strategies", "relevance_explanation"]
                },
                "differential_diagnoses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "disease_name": {"type": "string"},
                            "confidence_score": {"type": "number", "description": "A score from 0.0 to 1.0"},
                            "summary": {"type": "string", "description": "A concise, well-written paragraph describing the condition"},
                            "key_symptoms": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "Key symptoms that differentiate this condition"
                            },
                            "diagnostic_tests": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "Tests to confirm or rule out this diagnosis"
                            },
                            "treatment_approach": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "Treatment approach if this diagnosis is confirmed"
                            },
                            "relevance_explanation": {
                                "type": "string", 
                                "description": "Explanation of why this diagnosis should be considered"
                            }
                        },
                        "required": ["disease_name", "confidence_score", "summary", "key_symptoms", "diagnostic_tests", "treatment_approach", "relevance_explanation"]
                    }
                },
                "clinical_recommendations": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Specific, actionable recommendations for this patient"
                }
            },
            "required": ["primary_diagnosis", "differential_diagnoses", "clinical_recommendations"]
        }

        # 6. Chain it all together using LangChain Expression Language (LCEL)
        structured_llm = llm.with_structured_output(output_schema)

        # Create a RunnableLambda for the format_docs function
        format_docs_lambda = RunnableLambda(lambda docs: "\n\n---\n\n".join([d.page_content for d in docs]))

        # Custom retriever that tracks document usage
        def tracked_retriever_func(query):
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            try:
                docs = retriever.invoke(query)
                logger.info(f"Retrieved {len(docs)} documents")
                
                query_id = str(uuid.uuid4())
                app_state["last_query_id"] = query_id
                app_state["last_retrieved_docs"] = docs  # Store for debugging
                track_document_usage(docs, query_id)
                return docs
            except Exception as e:
                logger.error(f"Error during document retrieval: {e}")
                # Return empty list if retrieval fails
                return []
        
        # Create a RunnableLambda for the tracked retriever
        tracked_retriever_lambda = RunnableLambda(tracked_retriever_func)

        # Build the RAG chain using RunnableParallel and RunnableLambda
        rag_chain = (
            RunnableParallel(
                {
                    "context": tracked_retriever_lambda | format_docs_lambda,
                    "report": RunnablePassthrough()
                }
            )
            | prompt
            | structured_llm
        )
        
        app_state["rag_chain"] = rag_chain
        logger.info("LangChain RAG chain initialized successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}", exc_info=True)
    
    yield  # Application runs here
    
    logger.info("Shutting down C.A.R.E. Backend...")
    app_state.clear()


# --- API SETUP ---
app = FastAPI(
    title="C.A.R.E. Backend API",
    description="API for the Clinical Assistant for Reasoning and Evaluation",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://emrpro.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class ReportRequest(BaseModel):
    report_text: str = Field(..., min_length=20, description="The unstructured clinical report text.")

class ResourceRequest(BaseModel):
    disease_name: str = Field(..., description="The name of the disease to search for resources")

# --- API ENDPOINTS ---
@app.post("/api/v1/analyze")
async def analyze_report(request: ReportRequest):
    """
    Analyzes a clinical report using the RAG pipeline and returns a structured JSON diagnosis.
    """
    if "rag_chain" not in app_state:
        raise HTTPException(status_code=503, detail="RAG chain is not available. The service might be starting up or has encountered an error.")
    
    try:
        logger.info("Invoking RAG chain for analysis...")
        rag_chain = app_state["rag_chain"]
        response = await rag_chain.ainvoke(request.report_text)
        logger.info("Successfully received response from RAG chain.")
        
        # Include query ID in response for tracking
        query_id = app_state.get("last_query_id", "unknown")
        retrieved_docs = app_state.get("last_retrieved_docs", [])
        
        # Create a new response object that includes the query_id and retrieved documents
        result = {
            "analysis": response,
            "query_id": query_id,
            "retrieved_context": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in retrieved_docs
            ]
        }
        
        return result
    except Exception as e:
        logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the report.")

@app.post("/api/v1/clinical-analysis")
async def clinical_analysis(request: ReportRequest):
    """
    Analyzes a clinical report and returns a comprehensive structured diagnosis.
    This endpoint is optimized for frontend integration.
    """
    if "rag_chain" not in app_state:
        raise HTTPException(status_code=503, detail="RAG chain is not available. The service might be starting up or has encountered an error.")
    
    try:
        logger.info("Invoking RAG chain for clinical analysis...")
        rag_chain = app_state["rag_chain"]
        response = await rag_chain.ainvoke(request.report_text)
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response: {response}")
        
        # Handle the response which might be a list or dictionary
        if isinstance(response, list):
            logger.info("Response is a list, attempting to extract data...")
            # If it's a list, try to extract the first item or create a default structure
            if len(response) > 0:
                response_data = response[0] if isinstance(response[0], dict) else {}
                logger.info(f"Extracted data from list: {response_data}")
            else:
                logger.info("Empty list received, using empty dict")
                response_data = {}
        else:
            response_data = response
        
        # Ensure we have the required fields
        if not response_data:
            response_data = {
                "primary_diagnosis": {
                    "disease_name": "Unknown",
                    "confidence_score": 0.0,
                    "summary": "Unable to determine diagnosis",
                    "clinical_presentation": {
                        "common_symptoms": [],
                        "key_findings": []
                    },
                    "diagnostic_approach": {
                        "initial_tests": [],
                        "confirmatory_tests": []
                    },
                    "treatment_plan": {
                        "first_line": [],
                        "advanced_options": []
                    },
                    "prevention_strategies": [],
                    "relevance_explanation": "No relevant information found"
                },
                "differential_diagnoses": [],
                "clinical_recommendations": ["Please consult with a healthcare professional for proper diagnosis"]
            }
        
        # Format the response for frontend consumption
        result = {
            "success": True,
            "data": {
                "primary_diagnosis": response_data.get("primary_diagnosis", {}),
                "differential_diagnoses": response_data.get("differential_diagnoses", []),
                "clinical_recommendations": response_data.get("clinical_recommendations", []),
                "knowledge_base_sources": [
                    {
                        "disease": doc.metadata.get("disease", "Unknown"),
                        "source_url": doc.metadata.get("source_url", "")
                    } for doc in app_state.get("last_retrieved_docs", [])
                ]
            },
            "metadata": {
                "query_id": app_state.get("last_query_id", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result
    except Exception as e:
        logger.error(f"Error during clinical analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the report.")

@app.post("/api/v1/online-resources")
async def get_online_resources(request: ResourceRequest):
    """
    Search for online resources related to a specific disease using LangChain's search tools.
    """
    try:
        disease_name = request.disease_name
        logger.info(f"Searching for online resources for: {disease_name}")
        
        resources = search_online_resources(disease_name)
        
        logger.info(f"Found {len(resources)} resources for {disease_name}")
        
        return {
            "success": True,
            "disease": disease_name,
            "resources": resources,
            "count": len(resources)
        }
    except Exception as e:
        logger.error(f"Error fetching online resources: {e}", exc_info=True)
        # Return fallback resources even in case of error
        fallback = create_fallback_resources(request.disease_name)
        return {
            "success": True,
            "disease": request.disease_name,
            "resources": fallback,
            "count": len(fallback),
            "note": "Using fallback resources due to search error"
        }

@app.post("/api/v1/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyzes a clinical report from an uploaded file (PDF or image).
    Extracts text from the file and processes it through the RAG pipeline.
    """
    if "rag_chain" not in app_state:
        raise HTTPException(status_code=503, detail="RAG chain is not available. The service might be starting up or has encountered an error.")
    
    # Check file type
    file_type = file.content_type
    if file_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg", "image/tiff"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or image file.")
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        try:
            # Write the uploaded file to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
            # Extract text based on file type
            extracted_text = ""
            if file_type == "application/pdf":
                # Extract text from PDF
                import pdfplumber
                with pdfplumber.open(temp_file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
            elif file_type.startswith("image/"):
                # Extract text from image using OCR
                from PIL import Image
                import pytesseract
                image = Image.open(temp_file_path)
                extracted_text = pytesseract.image_to_string(image)
            
            # Check if text was extracted
            if not extracted_text or len(extracted_text.strip()) < 20:
                raise HTTPException(status_code=400, detail="Could not extract sufficient text from the uploaded file. Please ensure the file contains readable text.")
            
            # Process the extracted text through the RAG chain
            logger.info("Invoking RAG chain for file analysis...")
            rag_chain = app_state["rag_chain"]
            response = await rag_chain.ainvoke(extracted_text)
            logger.info("Successfully received response from RAG chain.")
            
            # Include query ID in response for tracking
            query_id = app_state.get("last_query_id", "unknown")
            retrieved_docs = app_state.get("last_retrieved_docs", [])
            
            # Create a new response object that includes the query_id and retrieved documents
            result = {
                "success": True,
                "data": {
                    "primary_diagnosis": response.get("primary_diagnosis", {}),
                    "differential_diagnoses": response.get("differential_diagnoses", []),
                    "clinical_recommendations": response.get("clinical_recommendations", []),
                    "knowledge_base_sources": [
                        {
                            "disease": doc.metadata.get("disease", "Unknown"),
                            "source_url": doc.metadata.get("source_url", "")
                        } for doc in retrieved_docs
                    ]
                },
                "metadata": {
                    "query_id": query_id,
                    "timestamp": datetime.now().isoformat(),
                    "extracted_text": extracted_text
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during file analysis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/v1/usage-stats")
async def get_usage_stats():
    """Get statistics about knowledge base usage."""
    return get_usage_statistics()

@app.get("/api/v1/recent-queries")
async def get_recent_queries(limit: int = 10):
    """Get information about recent queries and their retrieved documents."""
    usage_files = [f for f in os.listdir(USAGE_TRACKING_DIR) if f.endswith(".json")]
    
    # Sort by modification time (newest first)
    usage_files.sort(key=lambda x: os.path.getmtime(f"{USAGE_TRACKING_DIR}/{x}"), reverse=True)
    
    recent_queries = []
    for file in usage_files[:limit]:
        with open(f"{USAGE_TRACKING_DIR}/{file}", "r") as f:
            data = json.load(f)
            recent_queries.append(data)
    
    return {"recent_queries": recent_queries}

@app.get("/api/v1/debug/vector-store")
async def debug_vector_store():
    """Debug endpoint to check the vector store."""
    try:
        # Check direct ChromaDB connection
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        doc_count = collection.count()
        
        # Get a sample of documents
        sample_docs = []
        if doc_count > 0:
            sample = collection.get(limit=5, include=["metadatas", "documents"])
            for i in range(len(sample["ids"])):
                sample_docs.append({
                    "id": sample["ids"][i],
                    "metadata": sample["metadatas"][i],
                    "content_preview": sample["documents"][i][:200] + "..." if len(sample["documents"][i]) > 200 else sample["documents"][i]
                })
        
        # Check LangChain wrapper
        vector_store = app_state.get("vector_store")
        langchain_doc_count = 0
        if vector_store:
            langchain_doc_count = vector_store._collection.count()
        
        return {
            "direct_chromadb_count": doc_count,
            "langchain_wrapper_count": langchain_doc_count,
            "sample_documents": sample_docs
        }
    except Exception as e:
        logger.error(f"Error debugging vector store: {e}")
        return {"error": str(e)}

@app.post("/api/v1/debug/retrieval")
async def debug_retrieval(query: str = "chest pain"):
    """Debug endpoint to test document retrieval."""
    try:
        vector_store = app_state.get("vector_store")
        if not vector_store:
            return {"error": "Vector store not initialized"}
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        
        result = {
            "query": query,
            "retrieved_count": len(docs),
            "documents": []
        }
        
        for doc in docs:
            result["documents"].append({
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return result
    except Exception as e:
        logger.error(f"Error in debug retrieval: {e}")
        return {"error": str(e)}

@app.post("/api/v1/debug/rag-chain")
async def debug_rag_chain(report_text: str = "Patient is a 72-year-old male with atrial fibrillation and hypertension who was found by his family with sudden onset of right-sided weakness and speech difficulty."):
    """Debug endpoint to test the RAG chain directly."""
    try:
        if "rag_chain" not in app_state:
            return {"error": "RAG chain not initialized"}
        
        rag_chain = app_state["rag_chain"]
        response = await rag_chain.ainvoke(report_text)
        
        return {
            "response_type": str(type(response)),
            "response_content": response
        }
    except Exception as e:
        logger.error(f"Error in debug RAG chain: {e}")
        return {"error": str(e)}
