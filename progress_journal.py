from typing import Any
import streamlit as st
from datetime import datetime
from pydantic import BaseModel
import base64
import io
from PIL import Image
import assemblyai as aai
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressEntry(BaseModel):
    """Data model for progress entries"""
    date: datetime
    weight: float
    measurements: dict[str, float]
    mood: str
    energy_level: str
    photos: list[str] | None  # Base64 encoded images
    voice_notes: str | None  # Transcribed text
    text_notes: str | None
    workout_intensity: str
    workout_duration: int  # in minutes
    goals_progress: dict[str, Any]
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProgressJournal:
    def __init__(self):
        self.aai_client = aai.Transcriber()
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    def process_voice_note(self, audio_file) -> str:
        """Transcribe voice note using AssemblyAI"""
        try:
            transcript = self.aai_client.transcribe(audio_file)
            return transcript.text
        except Exception as e:
            logger.error(f"Error transcribing voice note: {e}")
            return ""

    def process_progress_photo(self, image_file) -> str:
        """Process and encode progress photo"""
        try:
            image = Image.open(image_file)
            # Resize image to reduce storage size while maintaining quality
            max_size = (800, 800)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to JPEG format
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error processing progress photo: {e}")
            return ""

class ProgressJournalUI:
    def __init__(self, journal: ProgressJournal, supabase_handler):
        self.journal = journal
        self.supabase = supabase_handler

    def render_entry_form(self):
        """Render the progress entry form"""
        st.subheader("📝 Add Progress Entry")
        
        with st.form("progress_entry_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0)
                mood = st.select_slider("Mood", options=["😞", "😐", "🙂", "😊", "🤗"])
                energy_level = st.select_slider(
                    "Energy Level",
                    options=["Very Low", "Low", "Medium", "High", "Very High"]
                )
            
            with col2:
                # Measurements
                st.markdown("#### Body Measurements (cm)")
                chest = st.number_input("Chest", min_value=0.0)
                waist = st.number_input("Waist", min_value=0.0)
                hips = st.number_input("Hips", min_value=0.0)
            
            # Workout details
            col3, col4 = st.columns(2)
            with col3:
                workout_intensity = st.select_slider(
                    "Workout Intensity",
                    options=["Rest Day", "Light", "Moderate", "Intense", "Very Intense"]
                )
            with col4:
                workout_duration = st.number_input("Workout Duration (minutes)", min_value=0, max_value=300)
            
            # Media uploads
            st.markdown("#### Media")
            photos = st.file_uploader(
                "Progress Photos",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
            
            voice_note = st.file_uploader("Voice Note", type=['mp3', 'wav', 'm4a'])
            
            # Text notes
            text_notes = st.text_area("Additional Notes")
            
            # Goals progress
            st.markdown("#### Goals Progress")
            goals_progress = {}
            if 'user_info' in st.session_state and 'goals' in st.session_state.user_info.__dict__:
                goals = st.session_state.user_info.goals.split('\n')
                for goal in goals:
                    if goal.strip():
                        progress = st.slider(
                            f"Progress on: {goal}",
                            min_value=0,
                            max_value=100,
                            value=50,
                            help="Slide to indicate progress percentage"
                        )
                        goals_progress[goal] = progress
            
            submitted = st.form_submit_button("Save Entry")
            
            if submitted:
                try:
                    # Process media files
                    processed_photos = []
                    if photos:
                        for photo in photos:
                            processed_photo = self.journal.process_progress_photo(photo)
                            if processed_photo:
                                processed_photos.append(processed_photo)
                    
                    voice_note_text = None
                    if voice_note:
                        voice_note_text = self.journal.process_voice_note(voice_note)
                    
                    # Create entry
                    entry = ProgressEntry(
                        date=datetime.now(),
                        weight=weight,
                        measurements={
                            'chest': chest,
                            'waist': waist,
                            'hips': hips
                        },
                        mood=mood,
                        energy_level=energy_level,
                        photos=processed_photos,
                        voice_notes=voice_note_text,
                        text_notes=text_notes,
                        workout_intensity=workout_intensity,
                        workout_duration=workout_duration,
                        goals_progress=goals_progress
                    )
                    
                    # Save entry to Supabase
                    result = self.supabase.save_progress_data(
                        st.session_state.user_id,
                        entry.model_dump()
                    )
                    
                    if result["success"]:
                        st.success("Progress entry saved successfully! 🎉")
                        st.balloons()
                    else:
                        st.error(f"Error saving entry: {result['error']}")
                    
                except Exception as e:
                    st.error(f"Error saving progress entry: {str(e)}")

    def render_progress_view(self):
        """Render progress history view"""
        st.subheader("📊 Progress History")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date")
        with col2:
            end_date = st.date_input("To Date")
        
        # Get entries from Supabase
        result = self.supabase.get_progress_data(st.session_state.user_id)
        if not result["success"]:
            st.error(f"Error loading progress data: {result['error']}")
            return

        entries = result["data"]
        if not entries:
            st.info("No progress entries found for the selected date range.")
            return

        # Filter entries by date range
        filtered_entries = [
            entry for entry in entries
            if start_date <= datetime.fromisoformat(entry['date']).date() <= end_date
        ]
        
        # Display entries
        for entry in filtered_entries:
            with st.expander(f"Entry: {datetime.fromisoformat(entry['date']).strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Weight:** {entry['weight']} kg")
                    st.markdown(f"**Mood:** {entry['mood']}")
                    st.markdown(f"**Energy Level:** {entry['energy_level']}")
                    st.markdown(f"**Workout Intensity:** {entry['workout_intensity']}")
                    st.markdown(f"**Workout Duration:** {entry['workout_duration']} minutes")
                
                with col2:
                    st.markdown("**Measurements:**")
                    for part, measurement in entry['measurements'].items():
                        st.markdown(f"- {part.title()}: {measurement} cm")
                
                # Display photos in a grid
                if entry.get('photos'):
                    st.markdown("**Progress Photos:**")
                    photo_cols = st.columns(len(entry['photos']))
                    for idx, photo in enumerate(entry['photos']):
                        with photo_cols[idx]:
                            st.image(
                                f"data:image/jpeg;base64,{photo}",
                                use_container_width=True
                            )
                
                # Display voice note transcription
                if entry.get('voice_notes'):
                    st.markdown("**Voice Note Transcription:**")
                    st.markdown(f"*{entry['voice_notes']}*")
                
                # Display text notes
                if entry.get('text_notes'):
                    st.markdown("**Notes:**")
                    st.markdown(entry['text_notes'])
                
                # Display goals progress
                if entry.get('goals_progress'):
                    st.markdown("**Goals Progress:**")
                    for goal, progress in entry['goals_progress'].items():
                        st.progress(progress / 100)
                        st.markdown(f"*{goal}:* {progress}%")

def initialize_progress_journal(supabase_handler):
    """Initialize and return ProgressJournal instance"""
    journal = ProgressJournal()
    ui = ProgressJournalUI(journal, supabase_handler)
    return ui