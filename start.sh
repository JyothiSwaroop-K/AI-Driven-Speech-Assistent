# #!/bin/bash

# # Step 1: Run the script to generate Chroma DB
# echo "Running ConvertExcelToChroma.py to initialize ChromaDB..."
# python ConvertExcelToChroma.py

# # Step 2: Launch the Streamlit app
# echo "Starting Streamlit app..."
# streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
#!/bin/bash

# echo "🔁 Starting custom startup script..."

# echo "📦 Installing dependencies (if needed)..."
# pip install -r requirements.txt

# echo "📁 Files in project root:"
# ls -la

# echo "🔨 Running ConvertExcelToChroma.py to create Chroma DB..."
# python ConvertExcelToChroma.py

# if [ $? -ne 0 ]; then
#     echo "❌ ConvertExcelToChroma.py failed."
#     exit 1
# fi

# echo "🚀 Launching Streamlit..."
# streamlit run app.py --server.port=$PORT --server.address=0.0.0.0


#!/bin/bash

echo "🔁 Starting custom startup script..."

echo "📦 Installing dependencies..."
pip install -r requirements.txt || { echo "❌ Dependency install failed"; exit 1; }

echo "📁 Files in root:"
ls -la

echo "🔨 Building Chroma DB..."
python ConvertExcelToChroma.py || { echo "❌ Chroma DB creation failed"; exit 1; }

echo "🚀 Starting Streamlit app..."
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
