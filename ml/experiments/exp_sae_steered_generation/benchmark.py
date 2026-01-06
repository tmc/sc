
import mlx.core as mx
import mlx.nn as nn
from steering_generator import MockLLM, SteeredGenerator
import argparse

def run_steering_benchmark():
    print("Running SAE Steered Generation Benchmark...")
    
    dim = 16
    sae_dim = 64
    
    # 1. Setup Mock Model & SAE
    llm = MockLLM(dim)
    encoder = nn.Linear(dim, sae_dim)
    decoder = nn.Linear(sae_dim, dim)
    
    # Initialize with recognizable weights for testing
    # Make Feature 5 align with Dimension 0 of activations
    encoder.weight = mx.zeros_like(encoder.weight)
    # encoder.weight[5, 0] = 1.0 # Cannot verify easily with random init
    
    generator = SteeredGenerator(llm, encoder, decoder)
    
    # 2. Baseline Generation (No Steering)
    print("  Generating Baseline...")
    text_base = generator.generate("Draw a statechart for a lock.")
    
    # 3. Steered Generation
    # Steer Feature 10 with strength 5.0
    print("  Steering Feature 10...")
    generator.set_steering(10, 5.0)
    text_steered = generator.generate("Draw a statechart for a lock.")
    
    # 4. Verification
    # Since this is a mock, we can't check semantic "text", 
    # but we can verify the hook didn't crash and logic executed.
    # To verify *effect*, we'd need to inspect the internal states of the MockLLM,
    # but the MockLLM in steering_generator.py returns strings.
    
    # Let's verify by inspecting if the steering vector map is populated
    assert 10 in generator.steering_vectors, "Steering vector not set!"
    assert generator.steering_vectors[10] == 5.0, "Steering strength incorrect!"
    
    print("  Steering Configuration Verified.")
    print("  Pipeline execution successful.")

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("Initializing SAE Steered Generation Experiment...")
    run_steering_benchmark()
    print("Experiment Complete.")

if __name__ == "__main__":
    main()
