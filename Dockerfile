# Step Five (implemented): containerize the Flask API from app.py.
# The model bundle is trained during the image build on the repo's
# synthetic sample data, so the image is self-contained:
#   docker build -t threat-response .
#   docker run -p 5000:5000 threat-response
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train on the bundled synthetic user_interactions.csv -> rl_agent_bundle.pt
RUN python run_all.py --episodes 200 --no-mlflow

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
