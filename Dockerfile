# Base image — Python 3.11 slim (lighter than full Python image)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run the tracker
CMD ["python", "src/pose/tracker.py"]