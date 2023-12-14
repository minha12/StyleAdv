import argparse
import os
import torch
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader

# add parent folder to default path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from criteria.id_loss import EnsembleIdLostMulti as IDLoss
from datasets.images_dataset import ImageDataset

# Define the thresholds for different models
threshold_dict = {
    "irse50": 0.412,
    "ir152": 0.42,
    "facenet": 0.36,
    "cur_face": 0.43,
    "mobile_face": 0.425,
}


# Function to calculate the pairwise distance
def calculate_distance(evaluator, input_tensor, output_tensor):
    input_tensor = input_tensor.cuda()
    output_tensor = output_tensor.cuda()
    return evaluator(input_tensor, output_tensor)[0]


# Function to process images and calculate success percentages
def eval_models(input_path, output_path, batch_size, num_workers):
    # Load the dataset
    input_dataset = ImageDataset(
        input_path, return_relative_paths=False, run_align=False
    )
    output_dataset = ImageDataset(
        output_path, return_relative_paths=False, run_align=False
    )

    # Create the DataLoader for both datasets
    input_loader = DataLoader(
        input_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    output_loader = DataLoader(
        output_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Loop through each model and calculate success percentages
    for model_name, threshold in threshold_dict.items():
        # Initialize the IDLoss evaluator with the specified model
        evaluator = IDLoss([model_name]).eval().cuda()

        # Initialize counters
        total_pairs = 0
        success_count = 0

        # Iterate over both DataLoaders simultaneously
        for (input_batch, _), (output_batch, _) in tqdm(
            zip(input_loader, output_loader), total=len(input_loader)
        ):
            # Calculate distances and update counts

            with torch.no_grad():
                distance = calculate_distance(evaluator, input_batch, output_batch)
            # print(distance)
            success = (distance > threshold).item()
            success_count += success
            total_pairs += 1

        # Calculate and print the percentage of successes for the current model
        percentage_successes = (success_count / total_pairs) * 100
        print(
            f"Model: {model_name}, Percentage of successes: {percentage_successes:.2f}%"
        )


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate pairwise distances between images."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input folder"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing images"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    return parser.parse_args()


# Main function to execute the script
def main():
    args = parse_arguments()
    eval_models(args.input_path, args.output_path, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
