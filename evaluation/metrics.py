#!/usr/bin/env python
"""
Evaluation Metrics for Job Matching Recommendation System
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
import os

class RecommendationEvaluator:
    """Evaluator for job matching recommendation system"""
    
    def __init__(self):
        self.job_seekers = None
        self.job_listings = None
        self.retrieval_results = None
        self.ranking_results = None
        self.ground_truth = None  # In a real system, this would be actual feedback data
    
    def load_data(self, job_seekers_file="data/raw/job_seekers.json",
                 job_listings_file="data/raw/job_listings.json",
                 retrieval_results_file="data/retrieval_results.json",
                 ranking_results_file="data/ranking_results.json"):
        """Load data for evaluation"""
        # Load job seekers
        with open(job_seekers_file, "r") as f:
            job_seekers_list = json.load(f)
            self.job_seekers = {js["id"]: js for js in job_seekers_list}
        
        # Load job listings
        with open(job_listings_file, "r") as f:
            job_listings_list = json.load(f)
            self.job_listings = {jl["id"]: jl for jl in job_listings_list}
        
        # Load retrieval results
        with open(retrieval_results_file, "r") as f:
            self.retrieval_results = json.load(f)
        
        # Load ranking results
        with open(ranking_results_file, "r") as f:
            self.ranking_results = json.load(f)
        
        print(f"Loaded {len(self.job_seekers)} job seekers, {len(self.job_listings)} job listings, "
              f"{len(self.retrieval_results)} retrieval results, and {len(self.ranking_results)} ranking results")
    
    def generate_synthetic_ground_truth(self, positive_ratio=0.05):
        """
        Generate synthetic ground truth for evaluation
        In a real system, this would be actual user feedback
        
        Args:
            positive_ratio (float): Ratio of positive examples
        """
        # Convert ranking results to DataFrame for easier processing
        df = pd.DataFrame(self.ranking_results)
        
        # Generate synthetic labels based on predicted scores and some randomness
        df["ground_truth_score"] = df["predicted_score"] * 0.8 + np.random.uniform(0, 0.2, size=len(df))
        
        # Convert to binary labels
        threshold = np.quantile(df["ground_truth_score"], 1 - positive_ratio)
        df["ground_truth"] = (df["ground_truth_score"] >= threshold).astype(int)
        
        # Save as ground truth
        self.ground_truth = df[["job_id", "candidate_id", "ground_truth", "ground_truth_score"]].to_dict("records")
        
        print(f"Generated synthetic ground truth with {sum(df['ground_truth'])} positive examples "
              f"({100 * df['ground_truth'].mean():.2f}% positive)")
    
    def calculate_precision_at_k(self, k_values=[1, 3, 5, 10]):
        """
        Calculate Precision@k for different k values
        
        Args:
            k_values (list): List of k values to calculate precision for
        
        Returns:
            dict: Precision@k values
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not available. Call generate_synthetic_ground_truth() first.")
        
        # Convert to DataFrame for easier processing
        gt_df = pd.DataFrame(self.ground_truth)
        rank_df = pd.DataFrame(self.ranking_results)
        
        # Merge ground truth with rankings
        df = pd.merge(rank_df, gt_df, on=["job_id", "candidate_id"])
        
        # Group by job
        grouped = df.sort_values(["job_id", "rank"]).groupby("job_id")
        
        precision_values = {}
        
        for k in k_values:
            # Calculate precision@k for each job
            precisions = []
            
            for _, group in grouped:
                # Get top k candidates
                top_k = group.head(k)
                
                # Calculate precision
                if len(top_k) > 0:
                    precision = top_k["ground_truth"].mean()
                    precisions.append(precision)
            
            # Average precision@k across all jobs
            precision_values[f"precision@{k}"] = np.mean(precisions)
        
        return precision_values
    
    def calculate_ndcg_at_k(self, k_values=[5, 10, 20]):
        """
        Calculate NDCG@k for different k values
        
        Args:
            k_values (list): List of k values to calculate NDCG for
        
        Returns:
            dict: NDCG@k values
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not available. Call generate_synthetic_ground_truth() first.")
        
        # Convert to DataFrame for easier processing
        gt_df = pd.DataFrame(self.ground_truth)
        rank_df = pd.DataFrame(self.ranking_results)
        
        # Merge ground truth with rankings
        df = pd.merge(rank_df, gt_df, on=["job_id", "candidate_id"])
        
        # Group by job
        grouped = df.sort_values(["job_id", "rank"]).groupby("job_id")
        
        ndcg_values = {}
        
        for k in k_values:
            # Calculate NDCG@k for each job
            ndcgs = []
            
            for _, group in grouped:
                if len(group) < 2:  # Need at least 2 items for NDCG
                    continue
                
                # Sort by predicted score (system ranking)
                system_ranking = group.sort_values("predicted_score", ascending=False).head(k)
                system_relevance = system_ranking["ground_truth_score"].values
                
                # Sort by ground truth score (ideal ranking)
                ideal_ranking = group.sort_values("ground_truth_score", ascending=False).head(k)
                ideal_relevance = ideal_ranking["ground_truth_score"].values
                
                # Calculate DCG and IDCG
                ranks = np.arange(1, len(system_relevance) + 1)
                dcg = np.sum((2 ** system_relevance - 1) / np.log2(ranks + 1))
                
                ideal_ranks = np.arange(1, len(ideal_relevance) + 1)
                idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(ideal_ranks + 1))
                
                # Calculate NDCG
                if idcg > 0:
                    ndcg = dcg / idcg
                    ndcgs.append(ndcg)
            
            # Average NDCG@k across all jobs
            if ndcgs:
                ndcg_values[f"ndcg@{k}"] = np.mean(ndcgs)
        
        return ndcg_values
    
    def calculate_ranking_metrics(self):
        """
        Calculate various ranking metrics
        
        Returns:
            dict: Ranking metrics
        """
        precision_values = self.calculate_precision_at_k()
        ndcg_values = self.calculate_ndcg_at_k()
        
        metrics = {}
        metrics.update(precision_values)
        metrics.update(ndcg_values)
        
        return metrics
    
    def evaluate_skill_coverage(self):
        """
        Evaluate skill coverage in top recommendations
        
        Returns:
            dict: Skill coverage metrics
        """
        # Convert to DataFrame for easier processing
        rank_df = pd.DataFrame(self.ranking_results)
        
        # Group by job and keep top 5 candidates for each job
        job_groups = rank_df.sort_values(["job_id", "rank"]).groupby("job_id")
        top_candidates = []
        
        for job_id, group in job_groups:
            top_candidates.extend(group.head(5)[["job_id", "candidate_id"]].to_dict("records"))
        
        # Calculate skill coverage
        coverage_ratios = []
        
        for match in top_candidates:
            job_id = match["job_id"]
            candidate_id = match["candidate_id"]
            
            job = self.job_listings.get(job_id)
            candidate = self.job_seekers.get(candidate_id)
            
            if not job or not candidate:
                continue
            
            required_skills = set(job["required_skills"])
            candidate_skills = set(candidate["skills"])
            
            if required_skills:
                coverage = len(required_skills.intersection(candidate_skills)) / len(required_skills)
                coverage_ratios.append(coverage)
        
        return {
            "avg_skill_coverage": np.mean(coverage_ratios),
            "median_skill_coverage": np.median(coverage_ratios)
        }
    
    def evaluate_retrieval_recall(self):
        """
        Evaluate retrieval recall - how many relevant candidates are retrieved
        
        Returns:
            float: Retrieval recall
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not available. Call generate_synthetic_ground_truth() first.")
        
        # Get positive examples from ground truth
        gt_df = pd.DataFrame(self.ground_truth)
        positive_examples = gt_df[gt_df["ground_truth"] == 1][["job_id", "candidate_id"]].to_dict("records")
        
        # Convert retrieval results to set of (job_id, candidate_id) tuples for fast lookup
        retrieval_set = set((r["job_id"], r["candidate_id"]) for r in self.retrieval_results)
        
        # Count how many positive examples were retrieved
        retrieved_positives = sum(1 for p in positive_examples if (p["job_id"], p["candidate_id"]) in retrieval_set)
        
        # Calculate recall
        recall = retrieved_positives / len(positive_examples) if positive_examples else 0
        
        return {"retrieval_recall": recall}
    
    def generate_evaluation_report(self, output_file="evaluation/results.json"):
        """
        Generate comprehensive evaluation report
        
        Args:
            output_file (str): Output file for evaluation results
        
        Returns:
            dict: Evaluation metrics
        """
        # Generate synthetic ground truth if not available
        if self.ground_truth is None:
            self.generate_synthetic_ground_truth()
        
        # Calculate metrics
        metrics = {}
        metrics.update(self.calculate_ranking_metrics())
        metrics.update(self.evaluate_skill_coverage())
        metrics.update(self.evaluate_retrieval_recall())
        
        # Print metrics
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save metrics to file
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Evaluation results saved to {output_file}")
        
        return metrics
    
    def plot_precision_recall_curve(self, output_file="evaluation/precision_recall.png"):
        """
        Plot precision-recall curve
        
        Args:
            output_file (str): Output file for plot
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not available. Call generate_synthetic_ground_truth() first.")
        
        # Convert to DataFrame for easier processing
        gt_df = pd.DataFrame(self.ground_truth)
        rank_df = pd.DataFrame(self.ranking_results)
        
        # Merge ground truth with rankings
        df = pd.merge(rank_df, gt_df, on=["job_id", "candidate_id"])
        
        # Calculate precision and recall at different thresholds
        thresholds = np.linspace(0, 1, 100)
        precision_values = []
        recall_values = []
        
        for threshold in thresholds:
            y_pred = (df["predicted_score"] >= threshold).astype(int)
            y_true = df["ground_truth"]
            
            if sum(y_pred) > 0:
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                
                precision_values.append(precision)
                recall_values.append(recall)
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall_values, precision_values, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Save plot
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            print(f"Precision-recall curve saved to {output_file}")
        
        return plt
    
    def plot_ndcg_distribution(self, k=5, output_file="evaluation/ndcg_distribution.png"):
        """
        Plot distribution of NDCG@k values across jobs
        
        Args:
            k (int): k value for NDCG
            output_file (str): Output file for plot
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not available. Call generate_synthetic_ground_truth() first.")
        
        # Convert to DataFrame for easier processing
        gt_df = pd.DataFrame(self.ground_truth)
        rank_df = pd.DataFrame(self.ranking_results)
        
        # Merge ground truth with rankings
        df = pd.merge(rank_df, gt_df, on=["job_id", "candidate_id"])
        
        # Group by job
        grouped = df.sort_values(["job_id", "rank"]).groupby("job_id")
        
        # Calculate NDCG@k for each job
        ndcg_values = []
        job_ids = []
        
        for job_id, group in grouped:
            if len(group) < 2:  # Need at least 2 items for NDCG
                continue
            
            # Sort by predicted score (system ranking)
            system_ranking = group.sort_values("predicted_score", ascending=False).head(k)
            system_relevance = system_ranking["ground_truth_score"].values
            
            # Sort by ground truth score (ideal ranking)
            ideal_ranking = group.sort_values("ground_truth_score", ascending=False).head(k)
            ideal_relevance = ideal_ranking["ground_truth_score"].values
            
            # Calculate DCG and IDCG
            ranks = np.arange(1, len(system_relevance) + 1)
            dcg = np.sum((2 ** system_relevance - 1) / np.log2(ranks + 1))
            
            ideal_ranks = np.arange(1, len(ideal_relevance) + 1)
            idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(ideal_ranks + 1))
            
            # Calculate NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_values.append(ndcg)
                job_ids.append(job_id)
        
        # Plot NDCG distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(ndcg_values, bins=20, kde=True)
        plt.xlabel(f'NDCG@{k}')
        plt.ylabel('Count')
        plt.title(f'Distribution of NDCG@{k} Values Across Jobs')
        
        # Save plot
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            print(f"NDCG distribution plot saved to {output_file}")
        
        return plt

def run_evaluation():
    """Run full evaluation pipeline"""
    evaluator = RecommendationEvaluator()
    evaluator.load_data()
    evaluator.generate_synthetic_ground_truth()
    
    # Generate evaluation report
    metrics = evaluator.generate_evaluation_report()
    
    # Generate plots
    evaluator.plot_precision_recall_curve()
    evaluator.plot_ndcg_distribution()
    
    return metrics

if __name__ == "__main__":
    run_evaluation() 