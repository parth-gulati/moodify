"""
Flask App for Mood Detection and Music Recommendation
Two-model system with LangGraph orchestration
"""

import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
import time
from langgraph.graph import StateGraph, END
from typing import TypedDict
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")

# Configuration
class Config:
    DATASET_PATH = os.path.join('data', 'moodify_dataset.csv')
    MODEL_PATH = os.path.join('models', 'emotion-model')
    PORT = int(os.environ.get('PORT', 8000))
    DEBUG = os.environ.get('FLASK_ENV') != 'production'
    
    # Spotify API credentials
    SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', '')
    SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '')
    
    # Model selection
    EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"  # Better emotion model


# LangGraph State Definition
class MoodMusicState(TypedDict):
    user_input: str
    detected_emotion: str
    emotion_confidence: float
    mood: str
    song_features: dict
    recommended_songs: list
    error: str | None


class MoodMusicSystem:
    def __init__(self, dataset_path, model_path, spotify_client_id=None, spotify_client_secret=None):
        logger.info("🚀 Initializing Mood Music System...")
        
        # Load emotion classifier (BETTER MODEL)
        logger.info("📦 Loading emotion classifier model...")
        self.emotion_classifier = self._load_emotion_model(model_path)
        logger.info("✅ Emotion model loaded successfully!")
        
        # Initialize Spotify client
        self.spotify = self._init_spotify(spotify_client_id, spotify_client_secret)
        
        # Label mapping
        self.label_to_mood = {
            0: 'sad',
            1: 'happy',
            2: 'energetic',
            3: 'calm'
        }
        self.mood_to_label = {v: k for k, v in self.label_to_mood.items()}
        
        # Enhanced emotion to mood mapping (for 28 emotions)
        self.emotion_to_mood = {
            # Happy emotions
            "joy": "happy",
            "amusement": "happy",
            "excitement": "happy",
            "love": "happy",
            "gratitude": "happy",
            "admiration": "happy",
            "approval": "happy",
            "caring": "happy",
            "pride": "happy",
            "relief": "happy",
            "optimism": "happy",
            
            # Sad emotions
            "sadness": "sad",
            "grief": "sad",
            "remorse": "sad",
            "disappointment": "sad",
            "embarrassment": "sad",
            
            # Energetic emotions
            "anger": "energetic",
            "annoyance": "energetic",
            "disapproval": "energetic",
            "desire": "energetic",
            
            # Calm emotions
            "neutral": "calm",
            "realization": "calm",
            "confusion": "calm",
            "curiosity": "calm",
            "surprise": "calm",
            "fear": "calm",
            "nervousness": "calm",
            "disgust": "calm"
        }
        
        # Load dataset
        logger.info(f"📊 Loading dataset from {dataset_path}...")
        self.music_df = self._load_music_dataset(dataset_path)
        self.music_by_mood = self._organize_by_mood()
        logger.info(f"✅ Dataset loaded! Total songs: {len(self.music_df)}")
    
    def _init_spotify(self, client_id, client_secret):
        """Initialize Spotify client"""
        if client_id and client_secret:
            try:
                logger.info("🎵 Initializing Spotify client...")
                auth_manager = SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
                sp = spotipy.Spotify(auth_manager=auth_manager)
                logger.info("✅ Spotify client initialized!")
                return sp
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Spotify: {str(e)}")
                return None
        else:
            logger.warning("⚠️  Spotify credentials not provided.")
            return None
    
    def get_track_info_from_uri(self, uri):
        """Get track name and artist from Spotify URI"""
        if not self.spotify:
            return None, None
        
        try:
            track_id = uri.split(':')[-1] if ':' in uri else uri
            track = self.spotify.track(track_id)
            name = track['name']
            artists = ', '.join([artist['name'] for artist in track['artists']])
            return name, artists
        except Exception as e:
            logger.warning(f"Failed to fetch track info for {uri}: {str(e)}")
            return None, None
    
    def enrich_songs_with_spotify_data(self, songs):
        """Enrich song data with Spotify track names"""
        enriched_songs = []
        
        for song in songs:
            enriched_song = song.copy()
            
            if self.spotify and song.get('uri'):
                if (not song.get('name') or song.get('name') == 'Unknown Song' or 
                    not song.get('artist') or song.get('artist') == 'Unknown Artist'):
                    
                    name, artist = self.get_track_info_from_uri(song['uri'])
                    
                    if name and artist:
                        enriched_song['name'] = name
                        enriched_song['artist'] = artist
                        enriched_song['source'] = 'spotify_api'
            
            enriched_songs.append(enriched_song)
        
        return enriched_songs
    
    def _load_emotion_model(self, model_path):
        """Load better emotion detection model"""
        try:
            if os.path.exists(model_path):
                logger.info(f"📁 Loading model from local path: {model_path}")
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                classifier = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    top_k=None,
                    device=-1
                )
                logger.info(f"✅ Local model loaded")
                return classifier
        except Exception as e:
            logger.warning(f"⚠️  Failed to load local model: {str(e)}")
        
        # Use better HuggingFace model
        try:
            logger.info(f"🌐 Loading model: {Config.EMOTION_MODEL}")
            classifier = pipeline(
                "text-classification",
                model=Config.EMOTION_MODEL,
                top_k=None,
                device=-1
            )
            logger.info(f"✅ Model loaded")
            return classifier
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise Exception("Could not load emotion detection model.")
    
    def _load_music_dataset(self, path):
        try:
            df = pd.read_csv(path)
            logger.info(f"Dataset columns: {list(df.columns)}")
            
            if 'labels' not in df.columns and 'label' in df.columns:
                df['labels'] = df['label']
            
            if 'name' not in df.columns:
                df['name'] = None
            if 'artist' not in df.columns:
                df['artist'] = None
            if 'uri' not in df.columns:
                df['uri'] = ''
            
            logger.info(f"Loaded {len(df)} songs")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return self._create_mock_data()
    
    def _create_mock_data(self):
        """Create mock data"""
        return pd.DataFrame([
            {'name': 'Mock Song', 'artist': 'Mock Artist', 'uri': '', 'labels': 1, 'energy': 0.5}
        ])
    
    def _organize_by_mood(self):
        mood_dict = {}
        for mood_idx, mood in enumerate(['sad', 'happy', 'energetic', 'calm']):
            mood_songs = self.music_df[self.music_df['labels'] == mood_idx]
            mood_dict[mood] = mood_songs.to_dict('records')
            logger.info(f"Organized {len(mood_songs)} songs for '{mood}'")
        return mood_dict
    
    def detect_emotion(self, text):
        results = self.emotion_classifier(text)[0]
        top_emotion = max(results, key=lambda x: x['score'])
        return top_emotion['label'], top_emotion['score']
    
    def map_emotion_to_mood(self, emotion):
        return self.emotion_to_mood.get(emotion.lower(), "calm")
    
    def recommend_songs(self, mood, limit=10, preferences=None):
        songs = self.music_by_mood.get(mood, [])
        
        if not songs:
            return []
        
        filtered_songs = songs.copy()
        
        # Filter by music preferences if provided
        if preferences:
            if "high energy" in preferences:
                energy_songs = [s for s in filtered_songs if s.get("energy", 0) > 0.7]
                if energy_songs:
                    filtered_songs = energy_songs
            elif "calm" in preferences:
                calm_songs = [s for s in filtered_songs if s.get("energy", 0) < 0.4]
                if calm_songs:
                    filtered_songs = calm_songs
            elif "danceable" in preferences:
                dance_songs = [s for s in filtered_songs if s.get("danceability", 0) > 0.6]
                if dance_songs:
                    filtered_songs = dance_songs
            elif "acoustic" in preferences:
                acoustic_songs = [s for s in filtered_songs if s.get("acousticness", 0) > 0.5]
                if acoustic_songs:
                    filtered_songs = acoustic_songs
        
        # Shuffle and limit
        random.shuffle(filtered_songs)
        selected_songs = filtered_songs[:limit]
        
        # Enrich with Spotify data
        enriched_songs = self.enrich_songs_with_spotify_data(selected_songs)
        
        return enriched_songs


# Initialize system
logger.info("🚀 Starting application initialization...")
mood_system = MoodMusicSystem(
    Config.DATASET_PATH, 
    Config.MODEL_PATH,
    Config.SPOTIFY_CLIENT_ID,
    Config.SPOTIFY_CLIENT_SECRET
)
logger.info("✅ Mood system ready!")


def create_multi_agent_workflow(mood_system):
    
    # Initialize LLM for song descriptions (using FLAN-T5 or similar)
    logger.info("Loading LLM for song descriptions...")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        llm_model_name = "google/flan-t5-large"  # You can use "google/flan-t5-large" for better results
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
        
        song_description_llm = pipeline(
            "text2text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_length=100,
            device=-1
        )
        logger.info("✅ LLM loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load LLM: {str(e)}")
        song_description_llm = None
    
    def agent_1_user_emotion(state: MoodMusicState):
        """Agent 1: Analyze user's emotion and recommend songs"""
        try:
            emotion, confidence = mood_system.detect_emotion(state["user_input"])
            mood = mood_system.map_emotion_to_mood(emotion)
            
            state["detected_emotion"] = emotion
            state["emotion_confidence"] = confidence
            state["mood"] = mood
            state["error"] = None
            
            # Get initial recommendations
            songs = mood_system.recommend_songs(mood, limit=10)
            state["recommended_songs"] = songs
            
        except Exception as e:
            state["error"] = str(e)
        
        return state
    
    def agent_2_song_description(state: MoodMusicState):
        """Agent 2: Use LLM to generate descriptions about each song"""
        if state["error"]:
            return state
        
        enriched_songs = []
        
        for song in state["recommended_songs"]:
            song_copy = song.copy()
            
            song_name = song.get('name', 'Unknown')
            artist = song.get('artist', 'Unknown')
            
            # Create prompt for LLM
            prompt = f"Q. Provide a short description for the following song: '{song_name}' by {artist}."
            
            try:
                if song_description_llm:
                    # Generate description using LLM
                    result = song_description_llm(prompt, max_length=80, do_sample=False)
                    description = result[0]['generated_text'].strip()
                    song_copy['llm_description'] = description
                    logger.info(f"Generated description for '{song_name}': {description}")
                else:
                    song_copy['llm_description'] = f"A song by {artist}"
                    
            except Exception as e:
                logger.warning(f"Failed to generate description for {song_name}: {str(e)}")
                song_copy['llm_description'] = f"A {state['mood']} song by {artist}"
            
            enriched_songs.append(song_copy)
        
        state["recommended_songs"] = enriched_songs
        return state
    
    # Build graph
    workflow = StateGraph(MoodMusicState)
    
    workflow.add_node("agent_1_user", agent_1_user_emotion)
    workflow.add_node("agent_2_llm_description", agent_2_song_description)
    
    workflow.set_entry_point("agent_1_user")
    workflow.add_edge("agent_1_user", "agent_2_llm_description")
    workflow.add_edge("agent_2_llm_description", END)
    
    return workflow.compile()


# Create LangGraph workflow
logger.info("Creating LangGraph workflow...")
langgraph_workflow = create_multi_agent_workflow(mood_system)
logger.info("✅ LangGraph workflow ready!")


# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": mood_system.emotion_classifier is not None,
        "total_songs": len(mood_system.music_df)
    })


@app.route('/api/analyze', methods=['POST'])
@cross_origin()
def analyze_mood():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Text input required"}), 400
        
        logger.info(f"Analyzing: {text[:50]}...")
        
        # Use LangGraph workflow
        initial_state = {
            "user_input": text,
            "detected_emotion": "",
            "emotion_confidence": 0.0,
            "mood": "",
            "song_features": {},
            "recommended_songs": [],
            "error": None
        }
        
        result = langgraph_workflow.invoke(initial_state)
        
        if result.get("error"):
            return jsonify({"error": result["error"]}), 500
        
        return jsonify({
            "detected_emotion": result["detected_emotion"],
            "confidence": round(result["emotion_confidence"], 4),
            "mood": result["mood"],
            "mood_label": mood_system.mood_to_label[result["mood"]],
            "song_features": result["song_features"],
            "recommended_songs": result["recommended_songs"],
            "total_recommendations": len(result["recommended_songs"])
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify({
        "total_songs": len(mood_system.music_df),
        "songs_by_mood": {
            mood: len(songs) 
            for mood, songs in mood_system.music_by_mood.items()
        }
    })


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=Config.DEBUG
    )