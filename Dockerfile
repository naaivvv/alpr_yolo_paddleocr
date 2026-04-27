FROM python:3.10-slim

# Install system dependencies required by OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Ensure the output directory exists for temporary image uploads
RUN mkdir -p $HOME/app/output

# Expose the standard port used by Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI application on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
