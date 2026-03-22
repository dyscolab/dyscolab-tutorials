"""
Script to update dependencies in setup cells in marimo noteboks for when they run 
in the marimo playground. Set dependencies in marimo_dependencies.toml and run:
pixi run update-dependencies
to run this script.
"""
import re
import json
import pathlib
import tomllib


def sync_marimo_groups(toml_config_path: str):
    config_file = pathlib.Path(toml_config_path)
    if not config_file.exists():
        print(f"Error: {toml_config_path} not found.")
        return

    with open(config_file, "rb") as f:
        groups = tomllib.load(f)

    # Regex preserves 'await micropip.install(' and any trailing arguments like 'verbose=False'
    pattern = r'(await micropip\.install\(\s*)\[.*?\](.*?)\)'

    for group_name, data in groups.items():
        dependencies = data.get("dependencies", [])
        files = data.get("files", [])
        
        new_deps_json = json.dumps(dependencies)
        
        print(f"--- Processing Group: {group_name} ---")
        
        for file_path in files:
            path = pathlib.Path(file_path)
            if not path.exists():
                f"Skipping: {file_path} (File not found)"
                continue

            content = path.read_text(encoding="utf-8")
            replacement = rf'\1{new_deps_json}\2)'
            new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)

            if count > 0:
                path.write_text(new_content, encoding="utf-8")
                print(f"Successfully updated: {file_path}")
            else:
                print(f"No micropip call found in: {file_path}")

if __name__ == "__main__":
    sync_marimo_groups("marimo_dependencies.toml")