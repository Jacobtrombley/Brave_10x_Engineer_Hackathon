#!/usr/bin/env python
"""
Data Preprocessing Script for Job Matching Recommendation System
Generates embeddings for job seekers and job listings
"""

import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Efficient, general-purpose embedding model
EMBEDDING_DIM = 384  # Output dimension for the chosen model

def load_data():
    """Load raw data from JSON files"""
    with open("data/raw/job_seekers.json", "r") as f:
        job_seekers = json.load(f)
    
    with open("data/raw/job_listings.json", "r") as f:
        job_listings = json.load(f)
    
    return job_seekers, job_listings

def create_candidate_text_representation(candidate):
    """Convert candidate profile to text representation for embedding"""
    # Combine relevant fields into a single text representation
    text = f"Name: {candidate['name']}. "
    text += f"Current title: {candidate['current_title']}. "
    text += f"Skills: {', '.join(candidate['skills'])}. "
    text += f"Experience level: {candidate['experience_level']} with {candidate['total_experience_months']} months of experience. "
    
    # Add work experience details
    text += "Work experience: "
    for exp in candidate['work_experience']:
        text += f"{exp['title']} at {exp['company']} for {exp['duration_months']} months. "
    
    # Add education
    text += f"Education: {candidate['education']['degree']} in {candidate['education']['field']} from {candidate['education']['institution']}. "
    
    # Add preferences
    text += f"Preferred job titles: {', '.join(candidate['preferred_job_titles'])}. "
    text += f"Preferred locations: {', '.join(candidate['preferred_locations'])}. "
    
    return text

def create_job_text_representation(job):
    """Convert job listing to text representation for embedding"""
    # Combine relevant fields into a single text representation
    text = f"Title: {job['title']}. "
    text += f"Company: {job['company']}. "
    text += f"Location: {job['location']}. "
    text += f"Remote: {job['remote']}. "
    text += f"Required skills: {', '.join(job['required_skills'])}. "
    
    if job['preferred_skills']:
        text += f"Preferred skills: {', '.join(job['preferred_skills'])}. "
    
    text += f"Experience level: {job['experience_level']} with minimum {job['min_experience_months']} months of experience. "
    text += f"Education requirement: {job['education_requirement']}. "
    
    # Short description
    text += f"Description summary: {job['description'][:300]}..."
    
    return text

def generate_embeddings(texts, model):
    """Generate embeddings for a list of texts using SentenceTransformer"""
    return model.encode(texts, show_progress_bar=True)

def preprocess_data():
    """Preprocess data and generate embeddings"""
    print("Loading data...")
    job_seekers, job_listings = load_data()
    
    print("Creating text representations...")
    candidate_texts = [create_candidate_text_representation(c) for c in tqdm(job_seekers)]
    job_texts = [create_job_text_representation(j) for j in tqdm(job_listings)]
    
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Generating candidate embeddings...")
    candidate_embeddings = generate_embeddings(candidate_texts, model)
    
    print("Generating job embeddings...")
    job_embeddings = generate_embeddings(job_texts, model)
    
    # Create output directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save embeddings
    np.save("data/processed/candidate_embeddings.npy", candidate_embeddings)
    np.save("data/processed/job_embeddings.npy", job_embeddings)
    
    # Save candidate and job IDs to map embeddings back to profiles
    candidate_ids = [c["id"] for c in job_seekers]
    job_ids = [j["id"] for j in job_listings]
    
    with open("data/processed/candidate_ids.json", "w") as f:
        json.dump(candidate_ids, f)
    
    with open("data/processed/job_ids.json", "w") as f:
        json.dump(job_ids, f)
    
    print("Data preprocessing complete!")
    print(f"Generated {len(candidate_embeddings)} candidate embeddings and {len(job_embeddings)} job embeddings")
    print("Embeddings saved to data/processed/")

if __name__ == "__main__":
    preprocess_data() 