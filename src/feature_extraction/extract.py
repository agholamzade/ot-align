
from datasets import load_dataset
from transformers import pipeline
import torch
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import os
import numpy as np
from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
import argparse
from sentence_transformers import SentenceTransformer

import torch.nn.functional as F

def compute_cosine_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_expanded = x.unsqueeze(1)  # [100, 1, 798]
    y_expanded = y.unsqueeze(0)  # [1, 100, 798]

    cosine_similarity_matrix = F.cosine_similarity(x_expanded, y_expanded, dim=-1)
    cosine_distance_matrix = 1.0 - cosine_similarity_matrix

    return cosine_distance_matrix

def tensors_to_npy(arr, is_tensor=False):

    if is_tensor:
        if arr.is_cuda:
            torch.cuda.synchronize()  
            tensor = arr.cpu()  
            arr = tensor.numpy()

    return arr

def extract_vision_features(model, dataset, batch_size=1):
    image_pipe = pipeline(task = "image-feature-extraction", model=model, device_map="auto",
                           pool = True, return_tensors="pt")
    # distributed_state = PartialState()
    # image_pipe.to(distributed_state.device)
    all_features = []
    for output in tqdm(image_pipe(KeyDataset(dataset, key= "image"), batch_size=batch_size)):
        all_features.append(output)

    all_features = torch.vstack(all_features)

    return all_features

def extract_language_features(model, sentences):
    model = SentenceTransformer("all-MiniLM-L6-v2", prompts={
        "image_classification": "a photo of a ",
    }, default_prompt_name="image_classification")

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    emb = model.encode_multi_process(sentences, pool)

    print("Embeddings computed. Shape:", emb.shape)

    return emb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str, default='language', choices=['avg', 'cls'])
    args = parser.parse_args()


    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    current_device = torch.cuda.current_device()
    print(f"Current device index: {current_device}")
    print(f"Current device name: {torch.cuda.get_device_name(current_device)}")

    for device_id in range(num_gpus):
        device_name = torch.cuda.get_device_name(device_id)
        print(f"Device {device_id}: {device_name}")

    save_path = "/ptmp/agholamzadeh/imagenet_cache"

    train_ds = load_dataset("/ptmp/agholamzadeh/imagenet", split="train")

    labels = np.array(train_ds["label"])
    
    # vision_model = "google/vit-base-patch16-224-in21k"
    # vision_features = extract_vision_features(vision_model, train_ds)
    # vision_features = tensors_to_npy(vision_features, is_tensor=True)

    language_model = "all-MiniLM-L6-v2"
    senteces = np.load("/ptmp/agholamzadeh/imagenet_cache/imagenet_classes.npy")
    language_features = extract_language_features(language_model, senteces)
    # language_features = tensors_to_npy(language_features, is_tensor=True)

    np.save(os.path.join(save_path, "prompt_features.npy"), language_features)
    np.save(os.path.join(save_path, "labels.npy"), labels)

    print("Done!")
