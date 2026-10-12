"""
Entry point to run the Clinical Document Classification & STG Compliance Prototype.
"""

import uvicorn
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    print("=" * 70)
    print("Starting Clinical Document Classification & STG Compliance System")
    print("Serving on: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False, access_log=False)
