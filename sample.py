import torch
import argparse

from utils import get_device, save_image, image_to_grid
from unet import UNet
from ddim import DDIM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["normal", "interpolation", "grid_interpolation"],
    )
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--n_ddim_steps", type=int, default=50, required=False)
    parser.add_argument("--eta", type=float, default=0, required=False)
    parser.add_argument("--trunc_normal_thresh", type=float, required=False)

    # For `"normal"`
    parser.add_argument("--batch_size", type=int, required=False)

    # For `"grid_interpolation"`
    parser.add_argument("--n_rows", type=int, default=10, required=False)
    parser.add_argument("--n_cols", type=int, default=10, required=False)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def main():
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()
    args = get_args()

    net = UNet()
    model = DDIM(
        model=net,
        img_size=args.IMG_SIZE,
        n_ddim_steps=args.N_DDIM_STEPS,
        eta=args.ETA,
        device=DEVICE,
    )
    state_dict = torch.load(str(args.MODEL_PARAMS), map_location=DEVICE)
    model.load_state_dict(state_dict)

    if args.MODE == "normal":
        gen_image = model.sample(
            batch_size=args.BATCH_SIZE, thresh=args.TRUNC_NORMAL_THRESH,
        )
        gen_grid = image_to_grid(gen_image, n_cols=int(args.BATCH_SIZE ** 0.5))
        save_image(gen_grid, save_path=args.SAVE_PATH)
    else:
        if args.MODE  == "interpolation":
            gen_image = model.interpolate_in_latent_space(thresh=args.TRUNC_NORMAL_THRESH)
            gen_grid = image_to_grid(gen_image, n_cols=10)
            save_image(gen_grid, save_path=args.SAVE_PATH)
        elif args.MODE  == "grid_interpolation":
            gen_image = model.interpolate_on_grid(
                n_rows=args.N_ROWS, n_cols=args.N_COLS, thresh=args.TRUNC_NORMAL_THRESH,
            )
            gen_grid = image_to_grid(gen_image, n_cols=args.N_COLS)
            save_image(gen_grid, save_path=args.SAVE_PATH)


if __name__ == "__main__":
    main()
