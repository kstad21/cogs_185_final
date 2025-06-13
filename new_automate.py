import torch
import os
import argparse
from generate import generate  # assumes this function exists and takes (model, prime_str)

# --num_generations argument to control how many to generate per file
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_generations', type=int, default=10, help="How many texts to generate per model per iteration")
args = argparser.parse_args()

# Load all models from trained_new/
def get_models():
    models = []
    model_dir = "./trained_new"
    for file in os.listdir(model_dir):
        path = os.path.join(model_dir, file)
        if os.path.isfile(path) and file.endswith('.pt'):
            model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
            model.name = file
            model.eval()
            models.append(model)
    return models

# Create output filename
def get_output_file(model_name, iteration):
    model_name = model_name.split(".")[0]
    return f"./results/{model_name}_iteration{iteration}.txt"

def main():
    prime_str = "the"
    num_iterations = 5
    models = get_models()

    os.makedirs("results", exist_ok=True)

    for model in models:
        for i in range(num_iterations):
            output_path = get_output_file(model.name, i)
            with open(output_path, 'w') as f:
                for _ in range(args.num_generations):
                    generated = generate(model, prime_str)
                    f.write(generated.strip() + "\n\n")
            print(f"[✓] Saved: {output_path}")

if __name__ == "__main__":
    main()
