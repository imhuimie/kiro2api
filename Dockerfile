# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file
COPY requirements.txt .

# Configure pip and install Python dependencies
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application files
COPY app.py token_reader.py entrypoint.sh ./

# Create necessary directories
RUN mkdir -p /home/appuser/.aws/sso/cache && \
    chmod +x entrypoint.sh && \
    chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Set environment variables
ENV HOME=/home/appuser
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose port
EXPOSE 8989

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8989/health || exit 1

# Use entrypoint script for smart startup
ENTRYPOINT ["./entrypoint.sh"]