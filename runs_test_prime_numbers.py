import numpy as np
import os

np.set_printoptions(precision=0)
np.set_printoptions(suppress=True)

# Import the prime numnbers file into a numpy array
prime_numbers = np.loadtxt(fname="primes.txt")

# Create array of all numbers between 1 and 100,000
all_numbers = np.arange(1,1000001)

results = np.zeros(1000000)

for i, val in enumerate(all_numbers):
    if i % 50000 == 0:
        print("Created: " + str(i))
    if val in prime_numbers:
        results[i] = 1

# We need to normalize our data because as per the prime numbers theorem, 
# In other words, the average gap between consecutive prime numbers among 
# the first N integers is roughly log(N).
# So, we need to figure out a way to normalize this.  What we will be doing
# is reducing the number of 0's between prime numbers according to log(N)
# Then, we will do the runs test on the final vector of 0's and 1's.
# N in this case is any number, but we will use the larger of the two primes
final_length = 0

# are we inside a run of 0's
running = False

start_indices = np.empty([0])
end_indices = np.empty([0])

for i, val in enumerate(results):
    if not running and val == 0:
        running = True
        start_indices = np.append(start_indices, [i])
    elif running and val == 0:
        continue
    elif running and val == 1:
        running = False
        end_indices = np.append(end_indices, [i])

start_indices = start_indices[:len(start_indices)-1]

cut_lengths = np.empty([0])
for i, val in enumerate(start_indices):
    cut_length = end_indices[i] - val
    if np.log(end_indices[i]) > 0:
        cut_length /= np.log(end_indices[i])
    cut_length = np.ceil(cut_length)
    cut_lengths = np.append(cut_lengths, [cut_length])

cut_lengths = cut_lengths.astype(np.int64)
# Create a new results array of 0's and 1's, but this time with shorter 0's.
length = np.sum(a=cut_lengths) + len(end_indices)
new_results = np.empty([0])
i = 0
h = 0
while i < len(results):
    if results[i] == 1:
        new_results = np.append(new_results, [1])
    elif results[i] == 0:
        index = -1
        for j, v in enumerate(start_indices):
            if v == i:
                index = j
        zeroes_num = cut_lengths[index]
        for j in range(zeroes_num.astype(np.int64)):
            new_results = np.append(new_results, [0])
        if h < len(end_indices):
            i += int(end_indices[h]-start_indices[h])
            h += 1
        else:
            i += 1
        continue
    i += 1

ones = np.sum(a=new_results)
zeroes = len(new_results) - ones

z_score = 1.96

mean = (2 * ones * zeroes / (ones + zeroes)) + 1
variance = (mean - 1) * (mean - 2) / (ones + zeroes - 1)

num_runs = 0
prev_val = 0
for val in new_results:
    if val != prev_val:
        num_runs += 1
        prev_val = val

predicted_runs = z_score * (variance**0.5) + mean

predicted_z = abs(num_runs - mean) / (variance**0.5)

print('Actual Runs: ' + str(num_runs))
print('Predicted Runs: ' + str(predicted_runs))
print('Predicted Z-Score: ' + str(predicted_z))
print('Mean: ' + str(mean))
print('Variance: ' + str(variance))
print('Z-Score: ' + str(z_score))
