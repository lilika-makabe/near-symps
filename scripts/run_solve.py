import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

OBJECTS = [
    "ball_undist",
    "buddha_undist",
    "bunny_undist",
    "cat_undist",
    "donut_undist",
    "sailor_undist",
    "santa_undist",
    "sheep_undist",
    "svcat_undist",
]


def main(args):
    data_root = os.path.join(_repo_root, "data")
    output_root = os.path.join(_repo_root, "output")

    for objname in OBJECTS:
        try:
            solve_per_object(args, os.path.join(data_root, objname),
                             os.path.join(output_root, objname))
        except Exception as e:
            print(e)


def solve_per_object(args, data_path, output_path):
    import copy
    tmp_args = copy.deepcopy(args)
    tmp_args.root_dir = data_path
    tmp_args.output_dir = output_path
    print(f"Solving: {data_path} -> {output_path}")
    from solver import SolvePS
    SolvePS().run(tmp_args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", default=3, type=int)
    parser.add_argument("--num_thresh", default=10, type=int)
    parser.add_argument("--initial_resize", type=float, default=1.)
    parser.add_argument("--w_cameraMatrix", action="store_true",
                        help="use camera intrinsic matrix (requires camera_params.txt)")
    parser.add_argument("--all_combinations", action="store_true",
                        help="use all pairwise combinations for constraints instead of consecutive pairs")
    parser.add_argument("--lights_to_load", nargs="+", default=list(range(50)), type=int)

    args = parser.parse_args()
    main(args)
