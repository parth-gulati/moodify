"""
Download HuggingFace model locally before deployment
Run this ONCE on your local machine before deploying
"""

from transformers import pipeline
import os

def download_model():
    print("🔄 Downloading emotion detection model...")
    print("This may take a few minutes (model is ~500MB)...")
    
    # Create models directory
    os.makedirs("models/emotion-model", exist_ok=True)
    
    # Download and save model
    model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
    
    # Save model locally
    model.save_pretrained("models/emotion-model")
    
    print("✅ Model downloaded successfully!")
    print("📁 Saved to: models/emotion-model/")
    print("\nNow you can commit this folder to git and deploy!")

if __name__ == "__main__":
    download_model()