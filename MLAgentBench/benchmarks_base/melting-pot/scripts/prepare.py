import subprocess

url = "https://storage.googleapis.com/dm-meltingpot/meltingpot-results-2.3.0.feather"
save_path = "../env/meltingpot-results-2.1.1.feather"

# Run wget with the -O flag to specify the output location
subprocess.run(["wget", url, "-O", save_path], check=True)

print(f"File downloaded to {save_path}")
