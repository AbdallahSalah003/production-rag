import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricResponse,
    ErrorResponse
)
from app.security import SecurePipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector
from app.agent import ProductionAgent


load_dotenv()

security: SecurePipeline = None
cache: ResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None
logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown
    """
    global security, cache, metrics, agent 
    settings = get_settings()

    logger.info("Starting production API...", extra={
        "extra_data": {
            "environment": settings.app_env,
            "primary_model": settings.primary_model,
            "fallback_model": settings.fallback_model,
            "tracing": settings.langsmith_tracing,
        }
    })

    security = SecurePipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()

    logger.info("All components initialized. Ready to serve!")

    yield # App is running

    # Shutdown
    logger.info("Shutting down...", extra={
        "extra_data": metrics.get_summary()
    })


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Production LangGraph API",
    description="A production ready chat api with security, caching, and observability",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Too many requests. Limit: {exc.detail}",
        },
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.
    Processes user input through security, caching, and agent pipeline.
    """
    start_time = time.time()

    security_result = security.process(body.message)
    if security_result["blocked"]:
        raise HTTPException(
            status_code=400,
            detail={"error": "Input blocked by security", "reasons": security_result["security_notes"]}
        )

    clean_input = security_result["output"] or body.message

    cached_response = cache.get(clean_input)
    if cached_response is not None:
        processing_time = (time.time() - start_time) * 1000
        metrics.record_request(
            latency_ms=processing_time,
            input_tokens=0,
            output_tokens=0,
            cache_hit=True,
        )
        return ChatResponse(
            response=cached_response,
            thread_id=body.thread_id,
            model_used="cached",
            cached=True,
            processing_time_ms=round(processing_time, 2),
        )

    agent_result = agent.invoke(clean_input)

    if agent_result.get("error"):
        metrics.record_request(
            latency_ms=(time.time() - start_time) * 1000,
            input_tokens=0,
            output_tokens=0,
            error=True,
        )
        raise HTTPException(status_code=500, detail={"error": "Agent error", "detail": agent_result["error"]})

    response_text = agent_result["response"]

    is_valid, cleaned_output, val_reason = security.validator.validate(response_text)
    if not is_valid:
        response_text = cleaned_output

    cache.set(clean_input, response_text)

    processing_time = (time.time() - start_time) * 1000
    metrics.record_request(
        latency_ms=processing_time,
        input_tokens=0,
        output_tokens=0,
        cache_hit=False,
    )

    return ChatResponse(
        response=response_text,
        thread_id=body.thread_id,
        model_used=agent_result.get("model_used", "unknown"),
        cached=False,
        processing_time_ms=round(processing_time, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def get_health():
    settings = get_settings()
    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None
    }
    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        env=settings.app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricResponse)
async def get_metrics():
    """Metrics for monitoring dashboards"""
    return MetricResponse(**metrics.get_summary())

@app.get("/cache/stats")
async def cache_stats():
    """Cache performance stats"""
    return cache.stats
