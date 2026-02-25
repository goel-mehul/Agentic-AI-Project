# Setup Notes

## Environment
- Python 3.11+
- Virtual environment: `source venv/bin/activate`
- Dependencies: `pip install -r backend/requirements.txt`

## Running
- Backend: `cd backend && uvicorn main:app --reload`
- Frontend: `cd frontend && npm run dev`
- Tests: `pytest evals/ -v`