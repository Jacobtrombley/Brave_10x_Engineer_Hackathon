#!/usr/bin/env python
"""
Approximate Nearest Neighbors (ANN) Retrieval Model
First stage of the recommendation pipeline: retrieves candidate-job pairs
"""

import numpy as np
import json
import os
import faiss
import time
from tqdm import tqdm

class ANNRetrieval:
    def __init__(self, index_type="flat"):
        """
        Initialize ANN retrieval model
        
        Args:
            index_type (str): Type of FAISS index to use ('flat', 'ivf', 'hnsw')
        """
        self.index_type = index_type
        self.candidate_index = None
        self.job_index = None
        self.candidate_embeddings = None
        self.job_embeddings = None
        self.candidate_ids = None
        self.job_ids = None
    
    def load_data(self):
        """Load preprocessed embeddings and IDs"""
        # Load embeddings
        self.candidate_embeddings = np.load("data/processed/candidate_embeddings.npy")
        self.job_embeddings = np.load("data/processed/job_embeddings.npy")
        
        # Load IDs
        with open("data/processed/candidate_ids.json", "r") as f:
            self.candidate_ids = json.load(f)
        
        with open("data/processed/job_ids.json", "r") as f:
            self.job_ids = json.load(f)
        
        print(f"Loaded {len(self.candidate_embeddings)} candidate embeddings and {len(self.job_embeddings)} job embeddings")
    
    def build_indexes(self):
        """Build FAISS indexes for fast similarity search"""
        d = self.candidate_embeddings.shape[1]  # embedding dimension
        
        # Initialize indexes based on chosen type
        if self.index_type == "flat":
            # Exact search with L2 distance
            self.candidate_index = faiss.IndexFlatL2(d)
            self.job_index = faiss.IndexFlatL2(d)
        
        elif self.index_type == "ivf":
            # IVF with 100 centroids
            nlist = 100
            quantizer = faiss.IndexFlatL2(d)
            
            self.candidate_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.job_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            
            # Need to train IVF indexes
            print("Training candidate index...")
            self.candidate_index.train(self.candidate_embeddings)
            
            print("Training job index...")
            self.job_index.train(self.job_embeddings)
        
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World graph index
            self.candidate_index = faiss.IndexHNSWFlat(d, 32)  # 32 neighbors per node
            self.job_index = faiss.IndexHNSWFlat(d, 32)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add embeddings to indexes
        print("Adding embeddings to indexes...")
        self.candidate_index.add(self.candidate_embeddings)
        self.job_index.add(self.job_embeddings)
        
        print(f"Built {self.index_type} indexes with {self.candidate_index.ntotal} candidates and {self.job_index.ntotal} jobs")
    
    def retrieve_candidates_for_job(self, job_idx, k=50):
        """
        Retrieve top-k candidate matches for a given job
        
        Args:
            job_idx (int): Index of the job in the embeddings array
            k (int): Number of candidates to retrieve
        
        Returns:
            tuple: (distances, candidate indices)
        """
        job_embedding = self.job_embeddings[job_idx:job_idx+1]
        
        # Search for nearest neighbors
        distances, indices = self.candidate_index.search(job_embedding, k)
        
        return distances[0], indices[0]
    
    def retrieve_jobs_for_candidate(self, candidate_idx, k=50):
        """
        Retrieve top-k job matches for a given candidate
        
        Args:
            candidate_idx (int): Index of the candidate in the embeddings array
            k (int): Number of jobs to retrieve
        
        Returns:
            tuple: (distances, job indices)
        """
        candidate_embedding = self.candidate_embeddings[candidate_idx:candidate_idx+1]
        
        # Search for nearest neighbors
        distances, indices = self.job_index.search(candidate_embedding, k)
        
        return distances[0], indices[0]
    
    def get_all_candidate_job_pairs(self, k=10, output_file=None):
        """
        Generate all candidate-job pairs by finding top-k matches for each job
        
        Args:
            k (int): Number of candidates to retrieve per job
            output_file (str): Optional path to save results
        
        Returns:
            list: List of dictionaries with job-candidate pairs and scores
        """
        results = []
        
        print(f"Retrieving top {k} candidates for each job...")
        start_time = time.time()
        
        for job_idx in tqdm(range(len(self.job_embeddings))):
            distances, candidate_indices = self.retrieve_candidates_for_job(job_idx, k)
            
            job_id = self.job_ids[job_idx]
            
            for i, (dist, cand_idx) in enumerate(zip(distances, candidate_indices)):
                # Convert L2 distance to similarity score (inverse and normalize)
                similarity = 1.0 / (1.0 + dist)
                
                results.append({
                    "job_id": job_id,
                    "candidate_id": self.candidate_ids[cand_idx],
                    "rank": i + 1,
                    "similarity_score": float(similarity),
                    "distance": float(dist)
                })
        
        elapsed = time.time() - start_time
        print(f"Generated {len(results)} candidate-job pairs in {elapsed:.2f} seconds")
        
        # Save results if output file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {output_file}")
        
        return results
    
    def run_pipeline(self, k=10, output_file="data/retrieval_results.json"):
        """Run full retrieval pipeline"""
        print("Loading data...")
        self.load_data()
        
        print(f"Building {self.index_type} indexes...")
        self.build_indexes()
        
        print("Retrieving candidate-job pairs...")
        results = self.get_all_candidate_job_pairs(k, output_file)
        
        return results

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="ANN Retrieval Model")
    parser.add_argument("--index-type", type=str, default="flat", choices=["flat", "ivf", "hnsw"],
                        help="Type of FAISS index to use")
    parser.add_argument("--k", type=int, default=10, help="Number of candidates to retrieve per job")
    parser.add_argument("--output", type=str, default="data/retrieval_results.json",
                       help="Output file for retrieval results")
    args = parser.parse_args()
    
    # Run retrieval
    retrieval = ANNRetrieval(index_type=args.index_type)
    retrieval.run_pipeline(k=args.k, output_file=args.output)

if __name__ == "__main__":
    main() 