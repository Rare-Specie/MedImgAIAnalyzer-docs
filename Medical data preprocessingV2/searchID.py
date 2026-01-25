#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os

def extract_ids_with_blank_lines(
    root_dir: str,
    input_txt: str,
    output_txt: str
):
    """
    For each subfolder name in input TXT:
    - Extract ID from its first sub-subfolder
    - If not found, write a blank line
    """

    # Read TXT lines in order
    with open(input_txt, 'r', encoding='utf-8-sig') as f:
        subfolder_names = [line.strip() for line in f]

    results = []

    for name in subfolder_names:
        if not name:
            # Preserve empty lines in input
            results.append("")
            continue

        folder_path = os.path.join(root_dir, name)

        if not os.path.isdir(folder_path):
            # Subfolder not found
            results.append("")
            continue

        # List sub-subfolders
        sub_subfolders = sorted(
            d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        )

        if sub_subfolders:
            # First sub-subfolder name is the ID
            results.append(sub_subfolders[0])
        else:
            # No ID folder present
            results.append("")

    # Write output with preserved order and blank lines
    with open(output_txt, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(f"{line}\n")


# Example usage
if __name__ == "__main__":
    extract_ids_with_blank_lines(
        root_dir="F:/guiling-CRC-MSI-CTdata/1-800-sorted-data",
        input_txt="F:/guiling-CRC-MSI-CTdata/1-800-sorted-data/filename.txt",
        output_txt="F:/guiling-CRC-MSI-CTdata/1-800-sorted-data/result.txt"
    )



