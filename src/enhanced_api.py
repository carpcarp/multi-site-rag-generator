import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from enhanced_rag_system import EnhancedHubSpotRAGSystem
from security import SecurityManager, QueryRequest, QueryResponse, ErrorResponse

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with validation"""
    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask")
    category_filter: Optional[str] = Field(None, description="Optional category filter")
    max_context_length: int = Field(4000, ge=500, le=10000, description="Maximum context length")

class EnhancedQueryResponse(BaseModel):
    """Enhanced query response"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_categories: List[str]
    processing_time: float
    cached: bool
    fallback: bool
    error_type: Optional[str] = None
    request_id: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    components: Dict[str, str]
    timestamp: float
    query_count: int

class StatsResponse(BaseModel):
    """System statistics response"""
    query_count: int
    performance_metrics: Dict[str, Any]
    memory_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    error_stats: Dict[str, Any]
    vector_store_stats: Dict[str, Any]

class BatchQueryRequest(BaseModel):
    """Batch query request model"""
    questions: List[str] = Field(..., max_items=10, description="List of questions to process")
    category_filter: Optional[str] = Field(None, description="Optional category filter")
    max_context_length: int = Field(4000, ge=500, le=10000, description="Maximum context length")

def get_client_ip(request: Request) -> str:
    """Extract client IP address"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

def create_enhanced_api(rag_system: EnhancedHubSpotRAGSystem = None, 
                       enable_auth: bool = False,
                       allowed_hosts: List[str] = None) -> FastAPI:
    """Create enhanced FastAPI application"""
    
    # Initialize RAG system
    if rag_system is None:
        rag_system = EnhancedHubSpotRAGSystem(enable_security=True)
    
    # Create FastAPI app
    app = FastAPI(
        title="Enhanced HubSpot Knowledge Base RAG API",
        description="Production-ready RAG system for HubSpot knowledge base queries with security and monitoring",
        version="2.0.0",
        docs_url="/docs" if not enable_auth else None,  # Disable docs in production
        redoc_url="/redoc" if not enable_auth else None
    )
    
    # Security middleware
    if allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=allowed_hosts
        )
    
    # CORS middleware (configure appropriately for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080"] if not enable_auth else [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        client_ip = get_client_ip(request)
        logger.info(f"Request {request_id}: {request.method} {request.url} from {client_ip}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(f"Request {request_id}: {response.status_code} in {process_time:.3f}s")
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request {request_id}: Error in {process_time:.3f}s - {str(e)}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "timestamp": time.time(),
                    "request_id": request_id
                }
            )
    
    # Authentication dependency
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Validate API key if authentication is enabled"""
        if not enable_auth:
            return "anonymous"
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Validate API key (implement your own logic)
        api_key = credentials.credentials
        expected_key = os.getenv("RAG_API_KEY")
        
        if not expected_key or api_key != expected_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return api_key  # or user ID
    
    @app.post("/query", response_model=EnhancedQueryResponse)
    async def query_rag(
        request_data: EnhancedQueryRequest,
        request: Request,
        user: str = Depends(get_current_user)
    ):
        """Query the enhanced RAG system"""
        try:
            client_ip = get_client_ip(request)
            user_id = f"{user}:{client_ip}"
            request_id = request.state.request_id
            
            # Query the RAG system
            response = rag_system.query(
                question=request_data.question,
                max_context_length=request_data.max_context_length,
                category_filter=request_data.category_filter,
                user_id=user_id
            )
            
            # Add request ID to response
            response_dict = {
                "answer": response.answer,
                "sources": response.sources,
                "query": response.query,
                "confidence": response.confidence,
                "suggested_categories": response.suggested_categories,
                "processing_time": response.processing_time,
                "cached": response.cached,
                "fallback": response.fallback,
                "error_type": response.error_type,
                "request_id": request_id
            }
            
            return EnhancedQueryResponse(**response_dict)
            
        except ValueError as e:
            # Validation or rate limit errors
            raise HTTPException(
                status_code=400,
                detail={
                    "error": str(e),
                    "error_code": "VALIDATION_ERROR",
                    "timestamp": time.time(),
                    "request_id": request.state.request_id
                }
            )
        except Exception as e:
            # Internal errors
            logger.error(f"Query error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "error_code": "INTERNAL_ERROR", 
                    "timestamp": time.time(),
                    "request_id": request.state.request_id
                }
            )
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """System health check"""
        try:
            health_data = rag_system.health_check()
            return HealthResponse(**health_data)
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Health check failed",
                    "error_code": "HEALTH_CHECK_FAILED",
                    "timestamp": time.time()
                }
            )
    
    @app.get("/stats", response_model=StatsResponse)
    async def get_system_stats(user: str = Depends(get_current_user)):
        """Get system statistics (requires authentication in production)"""
        try:
            stats = rag_system.get_system_stats()
            return StatsResponse(**stats)
        except Exception as e:
            logger.error(f"Stats error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to get statistics",
                    "error_code": "STATS_ERROR",
                    "timestamp": time.time()
                }
            )
    
    @app.get("/categories")
    async def get_categories():
        """Get available categories"""
        return {
            "categories": [
                "Contacts", "Companies", "Deals", "Service Hub", 
                "Reports", "Automation", "Integrations", "Getting Started", "General"
            ]
        }
    
    @app.post("/batch-query")
    async def batch_query(
        batch_request: BatchQueryRequest,
        user: str = Depends(get_current_user),
        request: Request = None
    ):
        """Process multiple queries in batch (limited to 10 for safety)"""
        try:
            client_ip = get_client_ip(request)
            user_id = f"{user}:{client_ip}"
            
            results = []
            for question in batch_request.questions:
                response = rag_system.query(
                    question=question,
                    max_context_length=batch_request.max_context_length,
                    category_filter=batch_request.category_filter,
                    user_id=user_id
                )
                results.append({
                    "question": question,
                    "answer": response.answer,
                    "confidence": response.confidence,
                    "cached": response.cached,
                    "fallback": response.fallback
                })
            
            return {"results": results}
            
        except Exception as e:
            logger.error(f"Batch query error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Batch query failed",
                    "error_code": "BATCH_QUERY_ERROR",
                    "timestamp": time.time()
                }
            )
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "Enhanced HubSpot RAG API",
            "version": "2.0.0",
            "status": "running",
            "timestamp": time.time()
        }
    
    return app

def main():
    """Main function to run the enhanced API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced HubSpot RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--enable-auth", action="store_true", help="Enable API key authentication")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize RAG system
    logger.info("Initializing enhanced RAG system...")
    rag_system = EnhancedHubSpotRAGSystem(enable_security=True)
    
    # Check system health
    health = rag_system.health_check()
    if health['status'] != 'healthy':
        logger.warning(f"System health check: {health['status']}")
        for component, status in health['components'].items():
            if status != 'healthy':
                logger.warning(f"Component {component}: {status}")
    
    # Create FastAPI app
    app = create_enhanced_api(
        rag_system=rag_system,
        enable_auth=args.enable_auth,
        allowed_hosts=["*"] if not args.enable_auth else ["localhost", "127.0.0.1"]
    )
    
    # Configure and run server
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )
    
    server = uvicorn.Server(config)
    
    logger.info(f"Starting enhanced API server on {args.host}:{args.port}")
    logger.info(f"Authentication: {'enabled' if args.enable_auth else 'disabled'}")
    logger.info(f"Documentation: http://{args.host}:{args.port}/docs")
    
    server.run()

if __name__ == "__main__":
    main()