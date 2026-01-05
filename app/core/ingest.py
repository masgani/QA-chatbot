from app.core.db import bootstrap_db
from app.core.rag import bootstrap_rag

if __name__ == "__main__":
    bootstrap_db()
    bootstrap_rag()