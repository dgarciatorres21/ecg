import os
import json
import argparse

def generate_dataset_json(dataset_dir, channel_names_str, labels_str):
    # creates dataset.json file required by nnu-net.
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return

    imagesTr_dir = os.path.join(dataset_dir, "imagesTr")
    if not os.path.isdir(imagesTr_dir):
        print(f"Error: imagesTr folder not found in {dataset_dir}")
        return
        
    num_training_cases = len([f for f in os.listdir(imagesTr_dir) if f.endswith(".nii.gz")])
    
    try:
        channel_names = json.loads(channel_names_str)
        labels = json.loads(labels_str)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for channel_names or labels.")
        return

    dataset_info = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": os.path.basename(dataset_dir)
    }

    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    print(f"dataset.json created for {num_training_cases} training cases at {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset.json for an nnU-Net dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Full path to the dataset directory.")
    parser.add_argument("--channel_names", type=str, default='{"0": "ecg_lead"}', help="JSON string for channel names.")
    parser.add_argument("--labels", type=str, default='{"background": 0, "foreground": 1}', help="JSON string for labels.")
    args = parser.parse_args()
    
    generate_dataset_json(args.dataset_dir, args.channel_names, args.labels)
