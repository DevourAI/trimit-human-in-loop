import os
from pathlib import Path


def matching_basename(filename, prefix, suffix, ext):
    return filename.startswith(prefix) and filename.endswith(suffix + ext)


def strip_prefix_suffix(filename, prefix, suffix, ext):
    if len(suffix + ext):
        return filename[len(prefix) : -len(suffix + ext)]
    else:
        return filename[len(prefix) :]


def get_max_existing_version_in_dir(output_dir, ext, prefix="", suffix=""):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    existing_basenames = os.listdir(output_dir)
    max_version = -1
    for basename in existing_basenames:
        if matching_basename(basename, prefix, suffix, ext):
            try:
                version = int(strip_prefix_suffix(basename, prefix, suffix, ext))
            except ValueError:
                continue
            else:
                max_version = max(max_version, version)
    return max_version


def filename_from_parts(dirname, ext, prefix, suffix, version):
    return os.path.join(dirname, f"{prefix}{version}{suffix}{ext}")


def get_new_integer_file_name_in_dir(output_dir, ext, prefix="", suffix=""):
    max_version = get_max_existing_version_in_dir(
        output_dir, ext, prefix=prefix, suffix=suffix
    )
    return filename_from_parts(output_dir, ext, prefix, suffix, max_version + 1)


def get_existing_integer_file_name_in_dir(output_dir, ext, prefix="", suffix=""):
    max_version = get_max_existing_version_in_dir(
        output_dir, ext, prefix=prefix, suffix=suffix
    )
    if max_version == -1:
        return None
    return filename_from_parts(output_dir, ext, prefix, suffix, max_version)
