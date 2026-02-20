FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download YOLOv8m model from Google Drive
RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1CHmF_c49hNzlBL2K72-qbkTMz2TEXbBL -O best.pt

# Expose port
EXPOSE 7860

# Start Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
