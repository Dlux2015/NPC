"""Repo-relative path + profile-loading helpers shared by
vision/tracking.py and vision/calibrate.py.

Not one of the pinned SS4 interfaces -- just avoids duplicating
"where is profiles/<name>/profile.yaml" logic. Every function takes an
optional `root` override so tests can point at a tmp_path fixture instead
of the real /profiles directory.
"""
import os

import yaml


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def profiles_root():
    return os.path.join(repo_root(), "profiles")


def profile_dir(name, root=None):
    return os.path.join(root or profiles_root(), name)


def load_profile_yaml(name, root=None):
    path = os.path.join(profile_dir(name, root), "profile.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "No profile.yaml for profile %r at %s" % (name, path)
        )
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
