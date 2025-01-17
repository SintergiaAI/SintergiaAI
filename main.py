import uvicorn
from app.app import app
import os

PORT = int(os.getenv("PORT", 8002))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=PORT, reload=True)