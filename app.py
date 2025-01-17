from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.controllers.telegram_controller import telegram_router, setup_telegram_bot
from app.controllers.logger_controller import logger
import os

# Inicializar FastAPI
app = FastAPI(
    title="SintergiAI API",
    description="APi SintergiAI",
    version="1.0.0",
    license_info={
        "name": "Licence: Jacob",
        "url": "",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telegram_router)

@app.on_event("shutdown")
async def shutdown_event():
    """shut"""
    logger.info("Shutting down...")


@app.on_event("startup")
async def startup_event():
    """innit"""
    try:
        logger.info("Starting up...")
        await setup_telegram_bot(app)
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )