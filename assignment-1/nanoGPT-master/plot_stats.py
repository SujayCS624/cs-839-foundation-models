import matplotlib.pyplot as plt

with open('output_log_wot_v2.txt', 'r') as file:
    data = file.read()

# Initialize lists to store the extracted data
num_input_chars = []
val_loss = []
val_perplexity = []
val_jsd = []
self_bleu_scores = []

# Parse the data
lines = data.strip().split('\n')
for i in range(0, len(lines), 6):
    print(lines[i].split(": "))
    num_chars = int(lines[i].split(": ")[1])
    val_loss_value = float(lines[i+1].split("val loss ")[1])
    val_perplex = float(lines[i+2].split("val perplexity ")[1])
    val_jsd_value = float(lines[i+3].split("val JSD ")[1])
    self_bleu = float(lines[i+4].split(": ")[1])
    
    num_input_chars.append(num_chars)
    val_loss.append(val_loss_value)
    val_perplexity.append(val_perplex)
    val_jsd.append(val_jsd_value)
    self_bleu_scores.append(self_bleu)

# Create the plots
plt.figure(figsize=(20, 5))

# # Plot 1: Validation Loss vs. Number of Input Characters
# plt.subplot(1, 4, 1)
# plt.plot(num_input_chars, val_loss, marker='o', color='blue')
# plt.title('Validation Loss vs Number of Input Characters')
# plt.xlabel('Number of Input Characters')
# plt.ylabel('Validation Loss')
# plt.grid(True)

# # Plot 2: Validation Perplexity vs. Number of Input Characters
# plt.subplot(1, 4, 2)
# plt.plot(num_input_chars, val_perplexity, marker='o', color='green')
# plt.title('Validation Perplexity vs Number of Input Characters')
# plt.xlabel('Number of Input Characters')
# plt.ylabel('Validation Perplexity')
# plt.grid(True)

# # Plot 3: Self-BLEU vs. Number of Input Characters
# plt.subplot(1, 4, 3)
# plt.plot(num_input_chars, self_bleu_scores, marker='o', color='orange')
# plt.title('Self-BLEU vs Number of Input Characters')
# plt.xlabel('Number of Input Characters')
# plt.ylabel('Self-BLEU Score')
# plt.grid(True)

# Plot 4: Validation JSD vs. Number of Input Characters
plt.subplot(1, 4, 4)
plt.plot(num_input_chars, val_jsd, marker='o', color='purple')
plt.title('Validation JSD vs Number of Input Characters')
plt.xlabel('Number of Input Characters')
plt.ylabel('Validation JSD')
plt.grid(True)

# # Show the plots
# plt.tight_layout()
plt.show()