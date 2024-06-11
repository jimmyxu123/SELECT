import argparse
import subprocess
import shlex

def run_script(script_name, arg_str):
    """Run a python script with given arguments as a single string and capture its output."""
    args = shlex.split(arg_str)  # Safely parse the string into a list of arguments
    try:
        result = subprocess.run(
            ["python", script_name] + args,
            text=True,
            capture_output=True,
            check=True  # This will raise an exception if the script exits with a non-zero status
        )
        print(f"Output of {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:\n{e.stderr}")
        raise

def main(args):
    print("Starting to run scripts...")
    run_script("base_ood_eval.py", args.script1_args)
    print("Script execution completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Python script with flexible arguments.')
    parser.add_argument('--base_ood_args', type=str, default='', help='Arguments for script1.py as a single quoted string')
    
    args = parser.parse_args()
    main(args)

