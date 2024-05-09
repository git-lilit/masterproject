from pathlib import Path

def get_model_sizes(directory):
    p = Path(directory)
    for file in p.iterdir():
        if file.is_file():
            size_bytes = file.stat().st_size
            size_mb = size_bytes / (1024**2)
            print(f"{file.name}: {size_mb:.2f} MB")

get_model_sizes("./hyperopt_models")
