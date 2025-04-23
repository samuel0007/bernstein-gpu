#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Build and run a CMake example"
    )
    parser.add_argument(
        "example",
        help="Name of the example to build (must match a subfolder under examples/)"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Build for CPU"
    )
    parser.add_argument(
        "--nvidia", action="store_true", help="Build for NVIDIA GPUs"
    )
    parser.add_argument(
        "--amd", action="store_true", help="Build for AMD GPUs"
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="CMake build directory (default: build/)"
    )
    parser.add_argument(
        "--run-args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the executable"
    )
    parser.add_argument(
        "-p", "--degree",
        type=int,
        help="Polynomial degree (compile-time P)"
    )
    parser.add_argument("--rebuild", action="store_true", help="Force clean build before building")

    args = parser.parse_args()

    targets = [args.cpu, args.nvidia, args.amd]
    if sum(bool(t) for t in targets) != 1:
        sys.exit("Error: Exactly one of --cpu, --nvidia or --amd must be set")

    

    cmake_defs = []
    if args.cpu:
        cmake_defs += ["-Dcpu=ON"]
    if args.nvidia:
        cmake_defs += ["-Dnvidia=ON"]
    if args.amd:
        cmake_defs += ["-Damd=ON"]
    if args.degree is not None:
        cmake_defs.append(f"-Dpolynomial_degree={args.degree}")

    os.makedirs(args.build_dir, exist_ok=True)

    
    cmake_cmd = ["cmake", "-S", ".", "-B", args.build_dir] + cmake_defs
    print("$", " ".join(cmake_cmd))
    subprocess.check_call(cmake_cmd)

    target = args.example
    build_cmd = [
        "cmake", "--build", args.build_dir,
        "--target", target
    ]
    if args.rebuild and os.path.isdir(args.build_dir):
        build_cmd.append("--clean-first")
        
    print("$", " ".join(build_cmd))
    subprocess.check_call(build_cmd)

    exe = os.path.join(args.build_dir, target)
    if not os.path.isfile(exe):
        sys.exit(f"Error: executable not found: {exe}")

    run_cmd = [exe]
    if args.run_args:
        run_cmd += args.run_args
    print("$", " ".join(run_cmd))
    subprocess.check_call(run_cmd)

if __name__ == "__main__":
    main()
