"""Audio file storage service using Supabase Storage."""

import uuid
from datetime import datetime
from supabase import Client


class StorageService:
    """Service for storing and retrieving audio files from Supabase Storage."""
    
    BUCKET_NAME = "audio-files"
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    async def upload_audio(
        self,
        audio_bytes: bytes,
        user_id: str,
        file_extension: str = "wav",
    ) -> str:
        """
        Upload audio file to Supabase Storage.
        
        Returns the public URL of the uploaded file.
        """
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}/{timestamp}_{uuid.uuid4().hex[:8]}.{file_extension}"
        
        # Upload to Supabase Storage
        self.supabase.storage.from_(self.BUCKET_NAME).upload(
            path=filename,
            file=audio_bytes,
            file_options={"content-type": f"audio/{file_extension}"}
        )
        
        # Get public URL
        url = self.supabase.storage.from_(self.BUCKET_NAME).get_public_url(filename)
        
        return url
    
    async def download_audio(self, file_path: str) -> bytes:
        """Download audio file from Supabase Storage."""
        response = self.supabase.storage.from_(self.BUCKET_NAME).download(file_path)
        return response
    
    async def delete_audio(self, file_path: str) -> None:
        """Delete audio file from Supabase Storage."""
        self.supabase.storage.from_(self.BUCKET_NAME).remove([file_path])
