FROM python:3.14-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Dependencies first (own layer, cached independently of source changes).
COPY requirements.txt .
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bake the cross-encoder's weights into the image at build time, so the
# container needs zero Hugging Face network access at runtime.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Selective, explicit copies — never `COPY . .` — so there is no path by which
# .env could end up in an image layer even if .dockerignore had a gap.
COPY src/ src/
COPY data/ data/
COPY prompts/ prompts/

EXPOSE 8000
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
