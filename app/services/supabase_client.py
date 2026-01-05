"""Supabase client initialization and utilities."""

from functools import lru_cache
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import httpx


@lru_cache(maxsize=1)
def get_supabase_client(url: str, key: str) -> Client:
    """
    Get cached Supabase client instance with extended timeouts.
    
    The default timeouts can be too short for embedding operations,
    so we extend them to avoid timeout errors.
    """
    # Create client with extended timeout (30 seconds instead of default)
    options = ClientOptions(
        postgrest_client_timeout=30,  # 30 second timeout for database operations
    )
    return create_client(url, key, options=options)
