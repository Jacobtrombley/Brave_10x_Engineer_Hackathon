# Brave 10x Engineer Hackathon - Job Matching Recommendation System

A sophisticated people-matching recommendation system that connects job seekers with relevant roles based on skills, experience, and preferences using AI.

## 🚀 Overview

This project implements a complete job matching recommendation system with the following components:

- **Data Generation**: Synthetic data generation for job seekers and job listings
- **Embeddings**: Vector representations of candidates and jobs
- **Retrieval Model**: First-stage matching using Approximate Nearest Neighbors (ANN)
- **Ranking Model**: Second-stage scoring and ordering of matches
- **Evaluation**: Comprehensive metrics and performance analysis
- **UI Prototypes**: Modern interfaces for both job seekers and employers

## 📊 System Architecture

The recommendation system uses a two-stage pipeline:

1. **Retrieval Stage**: Uses Approximate Nearest Neighbors (ANN) to efficiently find potential candidate-job matches based on embedding similarity.
2. **Ranking Stage**: Scores and orders the retrieved candidate-job pairs using a more sophisticated model that considers multiple matching factors.

For more details, see [OVERVIEW.md](OVERVIEW.md).

## 🛠️ Components

### Data Pipeline

- [Data Generation](data/generate_data.py): Creates synthetic profiles for job seekers and job listings
- [Data Preprocessing](data/preprocess.py): Transforms raw data into embeddings

### Models

- [Retrieval Model](models/retrieval/ann_retrieval.py): Implements ANN search using FAISS
- [Ranking Model](models/ranking/rank_model.py): Scores candidate-job pairs with advanced features

### Evaluation

- [Metrics & Evaluation](evaluation/metrics.py): Calculates precision, NDCG, and other ranking metrics

### UI Prototypes

- [Job Seeker Interface](ui/job_seeker/index.html): Dashboard and job matching for candidates
- [Employer Interface](ui/employer/index.html): Candidate management and analytics for hiring managers
- [UI Styles](ui/styles.css): Shared CSS styles using Shadcn design principles

## 🚦 Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see [requirements.txt](requirements.txt))

### Installation

```bash
git clone https://github.com/yourusername/job-matching-system.git
cd job-matching-system
pip install -r requirements.txt
```

### Running the System

You can run the entire pipeline with:

```bash
python main.py
```

Or run individual components:

```bash
# Generate synthetic data
python main.py --step generate

# Preprocess data and generate embeddings
python main.py --step preprocess

# Run retrieval model
python main.py --step retrieve

# Run ranking model
python main.py --step rank

# Evaluate system performance
python main.py --step evaluate
```

## 📝 Documentation

For detailed information about the system architecture, implementation, and usage:

- [Project Overview](OVERVIEW.md): Comprehensive documentation of the system

## 🔮 Future Improvements

1. **Feedback Loop Integration**: Implement a human-in-the-loop system to improve model performance based on user feedback.
2. **Advanced NLP Models**: Integrate more sophisticated language models for better understanding of job descriptions and resumes.
3. **Real-time Recommendations**: Enable real-time updates as new jobs or candidates are added to the system.
4. **Explainable AI**: Enhance transparency with better explanations of why certain matches are recommended.
5. **Skill Graph**: Implement a skill graph to better understand relationships between skills and enable transfer learning.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.