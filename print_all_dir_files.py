import os
import argparse


def read_files_in_directory(directory):
    """
    Reads all files under the specified directory and prints their content.
    This output can be used as input to an LLM for refactoring.

    Args:
        directory (str): The path to the directory.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            if not file_path.endswith(".py"):
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f"\n--- Start of File: {relative_path} ---\n")
                    print(f.read())
                    print(f"\n--- End of File: {relative_path} ---\n")
            except Exception as e:
                print(f"Error reading file {relative_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and print all files in a directory.")
    parser.add_argument("-d", "--directory", type=str, help="The path to the directory containing files to read.")
    args = parser.parse_args()

    if os.path.isdir(args.directory):
        read_files_in_directory(args.directory)
    else:
        print("The provided path is not a valid directory.")

