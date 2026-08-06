import os
import zipfile
from pathlib import Path
import time


def get_file_size_in_mb(file_path):
    size_in_bytes = os.path.getsize(file_path)
    return size_in_bytes / (1024 * 1024)  # Convert bytes to MB


def save_file_sizes_and_names(zip_path, file_type):
    original_size = 0.0
    with zipfile.ZipFile(zip_path, "r") as zipf:
        for file_info in zipf.infolist():
            original_size += file_info.file_size / (1024 * 1024)  # Size in MB
    compressed_size = get_file_size_in_mb(zip_path)

    with open(f"{file_type}_file_sizes.txt", "w") as f:
        f.write(f"Original {file_type} files total size: {original_size:.2f} MB\n")
        f.write(f"Compressed {file_type} archive size: {compressed_size:.2f} MB\n")

    with open(f"{file_type}_file_names.txt", "w") as f:
        for file_info in zipfile.ZipFile(zip_path, "r").infolist():
            f.write(f"{file_info.filename}\n")


def create_cif_files_zip_archive(path_for_batch: Path, results):
    zip_path = path_for_batch / "cif_files.zip"
    cif_files = results[2]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for cif_file_name, cif_str in cif_files:
            cif_file_name = (
                cif_file_name
                if cif_file_name.endswith(".cif")
                else cif_file_name + ".cif"
            )
            if not isinstance(cif_str, str):
                cif_str = cif_str.getbuffer()
            zipf.writestr(cif_file_name, cif_str)

    # save_file_sizes_and_names(zip_path, "cif")
    return str(zip_path)


def create_pdb_zip_archive(path_for_batch: Path, results):
    zip_path = path_for_batch / "pdb_files.zip"
    all_res_pdbs = results[0]
    all_contents = results[1]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pdb_file_name, pdb_str in zip(all_res_pdbs, all_contents):
            pdb_file_name = (
                pdb_file_name
                if pdb_file_name.endswith(".pdb")
                else pdb_file_name + ".pdb"
            )
            zipf.writestr(pdb_file_name, pdb_str)

    # save_file_sizes_and_names(zip_path, "pdb")
    return str(zip_path)
