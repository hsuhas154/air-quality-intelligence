FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
