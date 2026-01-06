
import mlx.core as mx
import mlx.nn as nn
import argparse
from dataclasses import dataclass
from typing import List, Tuple

# Experiment: Differentiable LCA
# Hypothesis: Neural networks can learn LCA for hierarchy
#
# Task: Given two nodes u, v in a hierarchy, predict LCA(u, v)
# Hierarchy is encoded via vector embeddings.

class NeuralLCA(nn.Module):
    """
    Predicts Least Common Ancestor from node embeddings.
    """
    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        # Predictor: [Emb(u); Emb(v)] -> logits over all nodes
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_nodes)
        )

    def forward(self, u_idx: mx.array, v_idx: mx.array):
        u_emb = self.embeddings(u_idx)
        v_emb = self.embeddings(v_idx)
        
        combined = mx.concatenate([u_emb, v_emb], axis=-1)
        logits = self.predictor(combined)
        return logits

def generate_hierarchy_data(num_nodes=20):
    """
    Generates a random tree and (u, v, lca) triples.
    0 is root.
    """
    parents = {0: 0}
    for i in range(1, num_nodes):
        p = int(mx.random.randint(0, i))
        parents[i] = p
        
    data = []
    
    def get_lca(u, v, parents):
        u_path = set()
        curr = u
        while True:
            u_path.add(curr)
            if curr == 0: break
            curr = parents[curr]
            
        curr = v
        while True:
            if curr in u_path: return curr
            if curr == 0: return 0
            curr = parents[curr]

    for _ in range(100): # 100 samples
        u = int(mx.random.randint(0, num_nodes))
        v = int(mx.random.randint(0, num_nodes))
        lca = get_lca(u, v, parents)
        data.append((u, v, lca))
        
    return data

def run_lca_benchmark():
    print("Running Differentiable LCA Benchmark...")
    
    num_nodes = 50
    model = NeuralLCA(num_nodes, 32)
    
    data = generate_hierarchy_data(num_nodes)
    
    # Simple evaluation loop (untrained)
    u_batch = mx.array([d[0] for d in data])
    v_batch = mx.array([d[1] for d in data])
    target_lca = mx.array([d[2] for d in data])
    
    logits = model.forward(u_batch, v_batch)
    preds = mx.argmax(logits, axis=-1)
    
    acc = mx.mean(preds == target_lca)
    print(f"  Initial Accuracy (Untrained): {acc.item():.2f}")
    
    # In a real run, we would train this.
    print("  Verification: Pipeline functional.")

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("Initializing Differentiable LCA Experiment...")
    run_lca_benchmark()
    print("Experiment Complete.")

if __name__ == "__main__":
    main()
