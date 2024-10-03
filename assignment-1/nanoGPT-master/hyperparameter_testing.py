import subprocess
import time

# Define the ranges of hyperparameters
block_sizes = [128]
batch_sizes = [32]
n_layers = [4]
n_heads = [4]
n_embds = [256]
dropouts = [0.0]
char_values = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
# char_values = [250, 500]

# block_sizes = [64]
# batch_sizes = [12, 32]
# n_layers = [4]
# n_heads = [8]
# n_embds = [512]
# dropouts = [0.0]

with open("output_log_wot_v2.txt", "w") as log_file:

    # Iterate over combinations of hyperparameters
    for block_size in block_sizes:
        for batch_size in batch_sizes:
            for n_layer in n_layers:
                for n_head in n_heads:
                    for n_embd in n_embds:
                        for dropout in dropouts:
                            for char_value in char_values:
                                # Build the command string
                                command = [
                                    "python", "train.py", "config/train_shakespeare_char.py",
                                    "--device=mps",
                                    "--compile=False",
                                    "--eval_iters=20",
                                    "--log_interval=1",
                                    f"--block_size={block_size}",
                                    f"--batch_size={batch_size}",
                                    f"--n_layer={n_layer}",
                                    f"--n_head={n_head}",
                                    f"--n_embd={n_embd}",
                                    "--max_iters=2000",
                                    "--lr_decay_iters=2000",
                                    f"--dropout={dropout}", 
                                    f"--n_chars={char_value}"
                                    ]
                            
                                # Print the command to keep track of progress
                                print(f"Running command: {' '.join(command)}")

                                start_time = time.time()
                                # Execute the command
                                result = subprocess.run(command, capture_output=True, text=True, check=True)

                                end_time = time.time()
                                duration = end_time - start_time

                                log_file.write(f"Number of Input Characters: {char_value}\n")

                                for line in result.stdout.splitlines():
                                    if line.startswith("step 2000:"):
                                        # Print the line and write it to the file
                                        print(line)
                                        log_file.write(f"{line}\n")

                                command = [
                                    "python", "sample.py",
                                    "--out_dir=out-shakespeare-char",
                                    "--device=mps"
                                    ]
                                
                                print(f"Running command: {' '.join(command)}")
                                result = subprocess.run(command, capture_output=True, text=True)
                                for line in result.stdout.splitlines():
                                    if line.startswith("Self-BLEU Score"):
                                        print(line)
                                        log_file.write(f"{line}\n")

                                log_file.write("------------------------------------\n")
