# Use official Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit and Flask ports
EXPOSE 8501 5000

# Environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# Start both Flask and Streamlit
CMD ["bash", "-c", "python app.py & streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
