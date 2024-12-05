#!/usr/bin/env python3

import os
import subprocess
import datetime
import time
import argparse
from pathlib import Path


def get_tree_output(path):
    """Get the directory structure using tree command for specific path"""
    print(f"Getting directory structure for {path}...")
    try:
        return subprocess.check_output(["tree", "-L", "3", path]).decode("utf-8")
    except Exception as e:
        return f"Error running tree command: {e}"


def count_python_files(start_path):
    """Count total Python files in specified directory and subdirectories"""
    print(f"Scanning for Python files in {start_path}...")
    count = 0
    python_files = []
    start_path = Path(start_path).resolve()

    try:
        for root, _, files in os.walk(start_path):
            root_path = Path(root)
            # Skip hidden directories
            if any(part.startswith(".") for part in root_path.parts):
                continue

            for file in files:
                if file.endswith(".py"):
                    count += 1
                    python_files.append(os.path.join(root, file))
                    print(f"\rFound {count} Python files...", end="")

        print(f"\nTotal Python files found: {count}")
        return count, python_files
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return 0, []


def create_review_file(target_path):
    """Generate code review file for specified path"""
    target_path = os.path.abspath(target_path)
    print(f"Generating review for path: {target_path}")

    if not os.path.exists(target_path):
        print(f"Error: Path {target_path} does not exist!")
        return

    # Generate unique filename based on the directory name
    dir_name = os.path.basename(target_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"code_review_{dir_name}_{timestamp}.txt"

    print("\nInitializing review process...")
    total_files, python_files = count_python_files(target_path)

    if total_files == 0:
        print("No Python files found in the specified path!")
        return

    content = [
        f"# Code Review for {dir_name}",
        f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Target Directory: {target_path}",
        "# " + "=" * 50,
        "",
        "# Directory Structure",
        "# " + "=" * 20,
        "",
        get_tree_output(target_path),
        "",
        "# Python Files Found",
        "# " + "=" * 20,
        "",
    ]

    # Process files
    processed_files = 0
    start_time = time.time()

    print("\nProcessing files:")
    for filepath in python_files:
        processed_files += 1
        progress = (processed_files / total_files) * 100

        # Create progress bar
        bar_length = 40
        filled_length = int(progress * bar_length // 100)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)

        print(f"\r[{bar}] {progress:.1f}% ({processed_files}/{total_files})")
        print(f"Processing: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                file_content = f.read()
                content.extend([
                    "\n" + "=" * 80,
                    f"# File: {filepath}",
                    "=" * 80 + "\n",
                    file_content,
                    "\n\n",
                ])
        except Exception as e:
            print(f"\nError reading {filepath}: {e}")
            content.append(f"# Error reading {filepath}: {e}\n")

    # Write output file
    print("\nWriting to output file...")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

        file_size = os.path.getsize(filename)
        print(f"\nSuccessfully created: {filename}")
        print(f"Location: {os.path.abspath(filename)}")
        print(f"Size: {file_size/1024:.2f} KB")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        print(f"Files processed: {processed_files}")

    except Exception as e:
        print(f"Error writing output file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate code review file for a specific directory")
    parser.add_argument("path", help="Path to the directory to review")
    args = parser.parse_args()

    print("Code Review File Generator")
    print("=" * 25)
    create_review_file(args.path)


if __name__ == "__main__":
    main()
