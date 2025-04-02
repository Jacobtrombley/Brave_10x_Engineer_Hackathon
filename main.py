#!/usr/bin/env python
"""
Main script for the Job Matching Recommendation System
Runs the full pipeline from data generation to evaluation
"""

import os
import sys
import argparse
import time
import json
from tqdm import tqdm

def run_data_generation():
    """Run data generation step"""
    print("\n=== Step 1: Data Generation ===")
    from data.generate_data import generate_data
    generate_data()

def run_data_preprocessing():
    """Run data preprocessing step"""
    print("\n=== Step 2: Data Preprocessing ===")
    from data.preprocess import preprocess_data
    preprocess_data()

def run_retrieval():
    """Run retrieval step"""
    print("\n=== Step 3: Retrieval ===")
    from models.retrieval.ann_retrieval import ANNRetrieval
    retrieval = ANNRetrieval(index_type="flat")
    retrieval.run_pipeline(k=50, output_file="data/retrieval_results.json")

def run_ranking():
    """Run ranking step"""
    print("\n=== Step 4: Ranking ===")
    from models.ranking.rank_model import run_ranking_pipeline
    metrics, rankings = run_ranking_pipeline()
    return metrics

def run_evaluation():
    """Run evaluation step"""
    print("\n=== Step 5: Evaluation ===")
    from evaluation.metrics import run_evaluation
    metrics = run_evaluation()
    return metrics

def run_full_pipeline():
    """Run the full recommendation pipeline"""
    start_time = time.time()
    
    # Step 1: Generate synthetic data
    run_data_generation()
    
    # Step 2: Preprocess data and generate embeddings
    run_data_preprocessing()
    
    # Step 3: Run retrieval model
    run_retrieval()
    
    # Step 4: Run ranking model
    ranking_metrics = run_ranking()
    
    # Step 5: Evaluate results
    evaluation_metrics = run_evaluation()
    
    # Print summary
    print("\n=== Pipeline Complete ===")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("\nRanking Metrics:")
    for metric, value in ranking_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nEvaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")

def print_welcome():
    """Print welcome message"""
    print("=" * 80)
    print("Job Matching Recommendation System")
    print("Brave 10x Engineer Hackathon")
    print("=" * 80)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Job Matching Recommendation System")
    parser.add_argument("--step", type=str, choices=["all", "generate", "preprocess", "retrieve", "rank", "evaluate"],
                       default="all", help="Step to run (default: all)")
    return parser.parse_args()

if __name__ == "__main__":
    print_welcome()
    args = parse_args()
    
    if args.step == "all":
        run_full_pipeline()
    elif args.step == "generate":
        run_data_generation()
    elif args.step == "preprocess":
        run_data_preprocessing()
    elif args.step == "retrieve":
        run_retrieval()
    elif args.step == "rank":
        run_ranking()
    elif args.step == "evaluate":
        run_evaluation() 