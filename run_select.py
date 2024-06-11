import argparse
import subprocess
import shlex

def run_script(script_name, arg_str):
    args = shlex.split(arg_str)  
    try:
        result = subprocess.run(
            ["python", script_name] + args,
            check=True  
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:\n{e.stderr}")
        raise

def main(args):
    print("Starting SELECT evaluation...")
    run_script("base-ood/base_ood_eval.py", args.base_ood)
    run_script("vtab/TestAllDatasets.py", args.vtab)
    run_script("ssl/eval_knn.py", args.ssl)
    print("SELECT evaluation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run select eval script with flexible arguments.')
    parser.add_argument('--base_ood', type=str, default='', help='base&ood args')
    parser.add_argument('--vtab', type=str, default='', help='vtab')
    parser.add_argument('--ssl', type=str, default='', help='ssl')
    args = parser.parse_args()
    main(args)


