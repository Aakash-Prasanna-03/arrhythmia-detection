import kagglehub

# Download latest version
path = kagglehub.dataset_download("bjoernjostein/physionet-snomed-mappings")

print("Path to dataset files:", path)