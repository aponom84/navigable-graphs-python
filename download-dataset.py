#!/usr/bin/env python
# coding: utf-8

import os
import requests
import argparse
import zipfile
import tarfile
import numpy as np
# from sentence_transformers import SentenceTransformer

import urllib.request
import requests

def download_file(url, output_file):
    if url.startswith("ftp://"):
        # Handle FTP download using urllib
        with urllib.request.urlopen(url) as response, open(output_file, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
    else:
        # Handle HTTP/HTTPS download using requests
        response = requests.get(url, stream=True)
        with open(output_file, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)

def extract_zip(zip_file, extract_to):
    """Extract a zip file to the specified directory"""
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def extract_tar(tar_file, extract_to):
    """Extract a tar.gz file to the specified directory"""
    print(f"Extracting {tar_file}...")
    with tarfile.open(tar_file, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def download_sift1m(output_dir):
    """Download and extract the SIFT1M dataset"""
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
    output_file = os.path.join(output_dir, "siftsmall.tar.gz")
    download_file(url, output_file)
    extract_tar(output_file, output_dir)

def download_glove(output_dir):
    """Download and extract the GloVe dataset"""
    url = "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
    output_file = os.path.join(output_dir, "glove.6B.zip")
    download_file(url, output_file)
    extract_zip(output_file, output_dir)

def download_deep1b(output_dir):
    """Download and extract the DEEP1B dataset"""
    url = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
    output_file = os.path.join(output_dir, "deep-image-96-angular.hdf5")
    download_file(url, output_file)

def download_fasttext(output_dir):
    """Download FastText word embeddings"""
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip"
    output_file = os.path.join(output_dir, "wiki.en.zip")
    download_file(url, output_file)
    extract_zip(output_file, output_dir)

# def download_sentence_embeddings(output_dir, model_name="paraphrase-MiniLM-L6-v2"):
#     """Download Sentence embeddings using Sentence Transformers"""
#     print(f"Downloading sentence embeddings using {model_name} model...")
#     model = SentenceTransformer(model_name)
#     sentences = [
#         "This is a sentence.",
#         "Sentence embeddings are useful for semantic search.",
#         "We are testing approximate nearest neighbor search using text embeddings.",
#     ]
#     embeddings = model.encode(sentences)
    
#     np.save(os.path.join(output_dir, f"{model_name}_embeddings.npy"), embeddings)
#     print(f"Sentence embeddings saved to {output_dir}")

def download_dataset(dataset_name, output_dir):
    """Download a dataset by name"""
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset_name == 'sift1m':
        download_sift1m(output_dir)
    elif dataset_name == 'glove':
        download_glove(output_dir)
    elif dataset_name == 'deep1b':
        download_deep1b(output_dir)
    elif dataset_name == 'fasttext':
        download_fasttext(output_dir)
    # elif dataset_name == 'sentence_embeddings':
    #     download_sentence_embeddings(output_dir)
    else:
        print(f"Unknown dataset {dataset_name}. Please choose 'sift1m', 'glove', 'deep1b', 'fasttext', or 'sentence_embeddings'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ANN datasets for testing approximate nearest neighbor methods")
    # parser.add_argument('--dataset', type=str, default='sift1m', help="Choose dataset: 'sift1m', 'glove', 'deep1b', 'fasttext', 'sentence_embeddings'")
    parser.add_argument('--dataset', type=str, default='sift1m', help="Choose dataset: 'sift1m', 'glove', 'deep1b', 'fasttext'")
    parser.add_argument('--output_dir', type=str, default='./datasets', help="Directory to save the dataset")
    
    args = parser.parse_args()
    
    download_dataset(args.dataset, args.output_dir)
