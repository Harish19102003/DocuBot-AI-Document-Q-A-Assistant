
FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing bytecode and enable live logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Streamlit port
EXPOSE 8501

# Start DocuBot
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]