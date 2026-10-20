import os
import shutil
import datetime
import yaml


def safe_copy(src, dst):
    """
    Safely copy a file or directory from source to destination.

    Args:
        src (str): Source path.
        dst (str): Destination path.
    """
    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    except OSError as e:
        print(f"Error copying {src} to {dst}: {e}")


def shorten_path(path, max_length=250):
    """
    Shorten a file path if it exceeds the specified maximum length.

    Args:
        path (str): Original file path.
        max_length (int): Maximum allowed length of the path. Defaults to 250.

    Returns:
        str: Shortened file path if necessary, otherwise the original path.
    """
    if len(path) <= max_length:
        return path
    base, ext = os.path.splitext(path)
    return base[:max_length - len(ext)] + ext


def save_drl_project(items_to_save, exclude_folders=None, save_dir='saved_projects'):
    # Get the directory where the script is located
    """Copy project files into an archival directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the project root directory (parent directory of the script)
    project_root = os.path.dirname(script_dir)

    # Create the save directory (under the project root)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.abspath(os.path.join(project_root, save_dir, f"drl_project_{timestamp}"))
    os.makedirs(save_path, exist_ok=True)

    if exclude_folders is None:
        exclude_folders = []

    for item in items_to_save:
        # Convert relative path to absolute path
        item = os.path.abspath(os.path.join(project_root, item))
        if os.path.exists(item):
            rel_path = os.path.relpath(item, start=project_root)
            dest = os.path.join(save_path, rel_path)
            dest = shorten_path(dest)

            if os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if
                               os.path.join(root, d) not in [os.path.abspath(os.path.join(project_root, ex)) for ex in
                                                             exclude_folders]]
                    for file in files:
                        src_file = os.path.join(root, file)
                        rel_file = os.path.relpath(src_file, item)
                        dst_file = os.path.join(dest, rel_file)
                        dst_file = shorten_path(dst_file)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        safe_copy(src_file, dst_file)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                safe_copy(item, dest)
        else:
            print(f"Warning: Item {item} not found.")

    print(f"Project saved to {save_path}")


# Read YAML configuration file
def read_yaml_config(config_path):
    """
    Read a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed YAML configuration data.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# Example usage
def save_source_file():
    # Items to save (folders and files)
    """Save a source snapshot for the current run."""
    items_to_save = [
        "config",
        "environment",
        "policy",
        "robots",
        "tests",
        "thirdparty",
        "utils",
        "visualization",
        ".gitignore",
        "README.md",
        "train_rl_with_configs.py",
    ]

    # Subfolders to exclude
    exclude_folders = [
        "visualization/images/",
    ]

    # Call the function to save the project
    save_drl_project(items_to_save, exclude_folders)
