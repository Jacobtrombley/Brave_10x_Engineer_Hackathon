#!/usr/bin/env python
"""
Data Generation Script for Job Matching Recommendation System
Generates synthetic data for job seekers and job listings
"""

import json
import os
import random
from datetime import datetime, timedelta
import numpy as np
from faker import Faker
from tqdm import tqdm

# Initialize Faker
fake = Faker()

# Constants for data generation
NUM_CANDIDATES = 1000
NUM_JOBS = 500
SKILLS = [
    # Programming Languages
    "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "Swift", "TypeScript", "PHP", "Ruby",
    # Frameworks
    "React", "Angular", "Vue.js", "Django", "Flask", "Spring", "Express.js", "TensorFlow", "PyTorch",
    # Database
    "SQL", "MongoDB", "PostgreSQL", "MySQL", "Redis", "Cassandra", "DynamoDB", "Firebase",
    # Cloud
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Serverless",
    # Data Science
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Analysis", "Big Data",
    # Soft Skills
    "Team Leadership", "Project Management", "Communication", "Problem Solving", "Agile", "Scrum"
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Boston, MA",
    "Chicago, IL", "Los Angeles, CA", "Denver, CO", "Atlanta, GA", "Portland, OR",
    "Remote"
]

JOB_TITLES = [
    "Software Engineer", "Data Scientist", "Machine Learning Engineer", "Frontend Developer", 
    "Backend Developer", "DevOps Engineer", "Data Engineer", "Product Manager", "UX Designer",
    "Full Stack Developer", "QA Engineer", "Site Reliability Engineer", "Engineering Manager",
    "Cloud Engineer", "Mobile Developer", "Security Engineer"
]

EXPERIENCE_LEVELS = ["Entry Level", "Mid Level", "Senior", "Lead", "Manager", "Director"]
EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]

def generate_job_seeker():
    """Generate a synthetic job seeker profile"""
    first_name = fake.first_name()
    last_name = fake.last_name()
    
    # Random skills selection
    num_skills = random.randint(5, 15)
    skills = random.sample(SKILLS, num_skills)
    
    # Generate work experience
    num_experiences = random.randint(1, 5)
    work_experience = []
    current_date = datetime.now()
    
    for _ in range(num_experiences):
        duration = random.randint(6, 48)  # months
        end_date = current_date
        start_date = end_date - timedelta(days=30*duration)
        
        work_experience.append({
            "company": fake.company(),
            "title": random.choice(JOB_TITLES),
            "start_date": start_date.strftime("%Y-%m"),
            "end_date": end_date.strftime("%Y-%m"),
            "duration_months": duration,
            "responsibilities": [fake.sentence() for _ in range(random.randint(2, 5))]
        })
        
        current_date = start_date - timedelta(days=random.randint(0, 90))
    
    # Calculate total experience
    total_experience = sum(exp["duration_months"] for exp in work_experience)
    
    return {
        "id": fake.uuid4(),
        "name": f"{first_name} {last_name}",
        "email": fake.email(),
        "phone": fake.phone_number(),
        "location": random.choice(LOCATIONS),
        "current_title": work_experience[0]["title"] if work_experience else random.choice(JOB_TITLES),
        "skills": skills,
        "experience_level": get_experience_level(total_experience),
        "total_experience_months": total_experience,
        "work_experience": work_experience,
        "education": {
            "degree": random.choice(EDUCATION_LEVELS),
            "field": fake.job(),
            "institution": fake.university()
        },
        "preferred_job_titles": random.sample(JOB_TITLES, random.randint(1, 3)),
        "preferred_locations": random.sample(LOCATIONS, random.randint(1, 3)),
        "open_to_remote": random.choice([True, False]),
        "salary_expectation": random.randint(50000, 200000)
    }

def generate_job_listing():
    """Generate a synthetic job listing"""
    num_required_skills = random.randint(5, 10)
    num_preferred_skills = random.randint(0, 5)
    
    required_skills = random.sample(SKILLS, num_required_skills)
    preferred_skills = random.sample([s for s in SKILLS if s not in required_skills], num_preferred_skills)
    
    title = random.choice(JOB_TITLES)
    experience_level = random.choice(EXPERIENCE_LEVELS)
    min_experience = get_min_experience_for_level(experience_level)
    
    return {
        "id": fake.uuid4(),
        "title": title,
        "company": fake.company(),
        "location": random.choice(LOCATIONS),
        "remote": random.choice([True, False, "Hybrid"]),
        "description": fake.text(max_nb_chars=1000),
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience_level": experience_level,
        "min_experience_months": min_experience,
        "education_requirement": random.choice(EDUCATION_LEVELS),
        "salary_range": {
            "min": random.randint(50000, 120000),
            "max": random.randint(120000, 250000)
        },
        "posting_date": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
        "application_deadline": (datetime.now() + timedelta(days=random.randint(7, 60))).strftime("%Y-%m-%d")
    }

def get_experience_level(months):
    """Map total experience in months to experience level"""
    if months < 24:
        return "Entry Level"
    elif months < 60:
        return "Mid Level"
    elif months < 96:
        return "Senior"
    elif months < 120:
        return "Lead"
    else:
        return "Manager"

def get_min_experience_for_level(level):
    """Map experience level to minimum months of experience"""
    mapping = {
        "Entry Level": 0,
        "Mid Level": 24,
        "Senior": 60,
        "Lead": 96,
        "Manager": 120,
        "Director": 144
    }
    return mapping.get(level, 0)

def generate_data():
    """Generate and save synthetic data for job seekers and job listings"""
    # Create output directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Generate job seekers
    job_seekers = []
    print("Generating job seeker profiles...")
    for _ in tqdm(range(NUM_CANDIDATES)):
        job_seekers.append(generate_job_seeker())
    
    # Generate job listings
    job_listings = []
    print("Generating job listings...")
    for _ in tqdm(range(NUM_JOBS)):
        job_listings.append(generate_job_listing())
    
    # Save data to JSON files
    with open("data/raw/job_seekers.json", "w") as f:
        json.dump(job_seekers, f, indent=2)
    
    with open("data/raw/job_listings.json", "w") as f:
        json.dump(job_listings, f, indent=2)
    
    print(f"Generated {NUM_CANDIDATES} job seeker profiles and {NUM_JOBS} job listings")
    print(f"Data saved to data/raw/job_seekers.json and data/raw/job_listings.json")

if __name__ == "__main__":
    generate_data() 