## Overview

pySTORM is a user-friendly, lightweight, and minimalist library for single-molecule localization. The aim of this module is to achieve a good balance between speed and accuracy while being easy to set up and use.

**Note, project is still in development. It is unlikely that things will work as written below.**

## Prerequisites

- Python 3.11
- Folder containing .tif files of raw SMLM data
- Ideally a python virtual environment. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for a guide if you are using conda or [here](https://virtualenv.pypa.io/en/latest/user_guide.html) for a guide if you are usng virtualenv.

## Installation

In an activated virtual environment and for use as a script:

```bash
python3 -m pip install pySTORM
```

If using as a command-line script:

```bash
git clone https://github.com/Matthewt303/pySTORM.git
cd pySTORM

python3 -m pip install .
```

For Windows powershell, use ```python``` instead of ```python3```.

## Usage

### As a script

It is recommended to use the module as a script. An example is shown here:

```python
import pySTORM.io as io
import pySTORM.fit as fit
import pySTORM.prev as preview

def main():
    input_path = "/path/to/smlm/data"
    output_path = "/path/where/things/are/saved"

    paths = io.get_movies(input_path)

    prev.preview(paths, threshold=3.0) ## <- optional

    specs = io.get_camera_params(pixel_size=100, adu=13.7, offset=100, gain=100)

    locs = fit.localize(paths, specs, threshold=3.0)

    io.save_loc_table(locs, output_path, format="csv")

if __name__ == "__main__":

    main()
```
You can preview the single-molecule detection with ```prev.preview```. A dialogue box should pop up where clicking 'yes' allows the program to continue while clicking 'no' will exit the program.

## As a command-line script

**work in progress**

## Parameters

Apart from file paths, Users must supply several parameters:

- pixel_size, the length of a pixel at the sample plane.
- adu, analog-digital conversion rate.
- offset, camera baseline graycount.
- gain, EMCCD gain. Set to 1 if not using an EMCCD.
- threshold. Modulates intensity threshold for single-molecule detection. Set to higher for more stringent thresholds, lower vice versa. 
- format. File format for localization table. Either "csv" or "hdf5".

The pixel size can be calculated from the imaging path. The adu and offset should be provided by the camera manufacturer's spec sheets and the gain is set during an experiment. For the file format, csvs are easier to read but use up much more memory than hdf5. For hdf5, specialised modules, such as ```h5py``` will be required.

## Motivation

Single-molecule localization microscopy has enabled us to probe *in situ* biological systems at unprecedented levels of detail. A critical step is the sub-pixel localization of fluorophores. While an enormous number of software are already available for this task, very few are easily accessible, resulting in a handful of 'community favourites', namely ThunderSTORM, SMAP, and Picasso [1, 2, 3]. Out of these, only ThunderSTORM and Picasso are truly open-source. Notably, despite the recent populartity of Python for scientific computing, SMLM features very few software packages that use Python (outside of deep learning).  

The design for software thus far in SMLM has been to deploy 'batteries-included'-type programs wherein a user is able to carry out single-molecule localization followed by a large suite of post-processing methods. The problems with this approach are two-fold. First, while the increased complexity of the software is excellent for expert users, it can be confusing for beginners who may be less familiar with the nuances of different PSF models or different fitting methods.

The second problem is that the complexity of the software brings about additional challenges in code maintenance, and, therefore, long-term sustainability and reproducibility which is a problem made particularly acute by frequent updates for the underlying programming languages. Out of the community favourites, only Picasso is actively maintained. The latest version of SMAP uses a version of Matlab that is already, at the time of writing, 6 versions behind the latest release while ThunderSTORM was last updated almost 10 years ago and is no longer compatible with the latest versions of ImageJ. The reliance on a handful of old community tools simply creates a ticking timebomb for reproducible research in SMLM.

This module aims to counteract some of these problems. It is designed to be easily installable and usable. Users need only supply file paths, camera parameters, and a threshold for single-molecule detection. The design is also simple and streamlined; only single-molecule localization is carried out and users are recommended to use the 'community favourites' for post-processing. This simplicity makes the code easy to maintain long-term. Finally, the module is fast, achieving similar speeds to ThunderSTORM and SMAP, due to its use of the Numba module.

## Acknowledgements and References

This project is inspired by the open-source approach taken by [Picasso](https://github.com/jungmannlab/picasso).

[1](https://academic.oup.com/bioinformatics/article/30/16/2389/2748167) Ovesny, M., Krızek, P., Borkovec, J., Svindrych, Z. & Hagen, G. M. Thunder-STORM: a comprehensive ImageJ plug-in for PALM and STORM data
analysis and super-resolution imaging. *Bioinformatics* **30**, 2389–2390 (2014)

[2](https://www.nature.com/articles/s41592-020-0938-1) Ries, J. SMAP: a modular super-resolution microscopy analysis platform for
SMLM data. *Nature Methods* **17**, 870–872 (2020)

[3](https://www.nature.com/articles/nprot.2017.024) Schnitzbauer, J., Strauss, M., Schlichthaerle, T. et al. Super-resolution microscopy with DNA-PAINT. *Nature Protocols* **12**, 1198–1228 (2017)