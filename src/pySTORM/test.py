import pySTORM.io as io
import pySTORM.fit as fit
import pySTORM.prev as prev


def main():
    input_path = "/path/to/smlm/data"
    output_path = "/path/where/things/are/saved"

    paths = io.get_movies(input_path)

    specs = io.get_camera_params(pixel_size=97.5, adu=0.59, offset=100, gain=100)

    ## prev.preview(paths, threshold=3.0) ## <- optional

    locs = fit.localize(paths, specs, threshold=3.0)

    io.save_loc_table(locs, output_path, format="csv")


if __name__ == "__main__":
    main()
