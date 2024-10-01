import subprocess
import time

# Define the ranges of hyperparameters
block_sizes = [64, 128]
batch_sizes = [12, 32]
n_layers = [4, 16]
n_heads = [4, 16]
n_embds = [128, 256]
dropouts = [0.0, 0.2]

# block_sizes = [64]
# batch_sizes = [12, 32]
# n_layers = [4]
# n_heads = [8]
# n_embds = [512]
# dropouts = [0.0]

with open("output_log.txt", "w") as log_file:

    # Iterate over combinations of hyperparameters
    for block_size in block_sizes:
        for batch_size in batch_sizes:
            for n_layer in n_layers:
                for n_head in n_heads:
                    for n_embd in n_embds:
                        for dropout in dropouts:
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
                                f"--dropout={dropout}"
                            ]
                            
                            # Print the command to keep track of progress
                            print(f"Running command: {' '.join(command)}")

                            log_file.write(f"{block_size, batch_size, n_layer, n_head, n_embd}\n")

                            start_time = time.time()
                            # Execute the command
                            result = subprocess.run(command, capture_output=True, text=True)

                            end_time = time.time()
                            duration = end_time - start_time

                            log_file.write("=== Hyperparameter Configuration ===\n")
                            log_file.write(f"Block Size: {block_size}\n")
                            log_file.write(f"Batch Size: {batch_size}\n")
                            log_file.write(f"Number of Layers: {n_layer}\n")
                            log_file.write(f"Number of Heads: {n_head}\n")
                            log_file.write(f"Embedding Size: {n_embd}\n")
                            log_file.write(f"Dropout: {dropout}\n")
                            log_file.write(f"Time Taken: {duration:.2f} seconds\n")

                            for line in result.stdout.splitlines():
                                if line.startswith("step 2000:"):
                                    # Print the line and write it to the file
                                    print(line)
                                    log_file.write(f"{line}\n")
                                    break  # Exit the loop once the line is found

                            log_file.write("------------------------------------\n")
                            
                            # Print the output from the command
                            print(f"Command Output:\n{result.stdout}")
                            print(f"Command Error (if any):\n{result.stderr}")

                            # Optionally, handle results (e.g., logging the results)
                            # You can log results or save them to a file for later analysis
