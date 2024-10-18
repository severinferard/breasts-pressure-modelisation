import struct
import csv


def binary_to_csv(binary_file_path, csv_file_path):
    with open(binary_file_path, 'rb') as binary_file:
        # Read the entire binary content
        binary_data = binary_file.read()

    # Each int32 is 4 bytes, and each sample consists of 252 int32 values
    num_ints_per_sample = 252
    int_size = 4  # size of int in bytes
    sample_size = num_ints_per_sample * int_size

    # Calculate the number of samples in the binary data
    num_samples = len(binary_data) // sample_size

    # Prepare to write to CSV
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Iterate through each sample and unpack the int32 values
        for i in range(num_samples):
            start_index = i * sample_size
            sample_data = binary_data[start_index:start_index + sample_size]

            # Unpack the binary data into int32 values
            ints = struct.unpack(f'{num_ints_per_sample}i', sample_data)

            # Write the ints as a row in the CSV
            csv_writer.writerow(ints)


if __name__ == "__main__":
    binary_file_path = '/media/ubuntu/BREASTIES/data_20241015_172754.bin'  # Change to your input binary file path
    csv_file_path = 'output.csv'     # Change to your desired output CSV file path
    binary_to_csv(binary_file_path, csv_file_path)
