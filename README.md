## Overview

pySTORM is a user-friendly, lightweight, and minimalist library for single-molecule localization. The aim of this module is to achieve a good balance between speed and accuracy while being easy to set up and use.

**Note** Project is still in development. It is unlikely that things will work as written below. 

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

def main():
    input_path = "/path/to/smlm/data"
    output_path = "/path/where/things/are/saved"

    paths = io.get_movies(input_path)

    specs = io.get_camera_params(pixel_size=100, adu=13.7, offset=100, gain=100)

    locs = fit.localize(paths, specs, threshold=3.0)

    io.save_loc_table(locs, output_path, format="csv")

if __name__ == "__main__":

    main()
```

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

## Motivation

Single-molecule localization microscopy has enabled us to probe *in situ* biological systems at unprecedented levels of detail. A critical step is the sub-pixel localization of fluorophores. While an enormous number of software are already available for this task, very few are easily accessible, resulting in a handful of 'community favourites', namely ThunderSTORM, SMAP, and Picasso [1, 2, 3]. Out of these, only ThunderSTORM and Picasso are truly open-source. 

The design for software thus far in SMLM has been to deploy 'batteries-included'-type programs wherein a user is able to carry out single-molecule localization followed by a large suite of post-processing methods. The problems with this approach are two-fold. First, while the increased complexity of the software is excellent for expert users, it can be confusing for beginners who may be less familiar with the nuances of different PSF models or different fitting methods.

The second problem is that the complexity of the software brings about additional challenges in code maintenance, and, therefore, long-term sustainability and reproducibility which is a problem made particularly acute by frequent updates for the underlying programming languages. Out of the community favourites, only Picasso is actively maintained. SMAP is already, at the time of writing, 13 versions behind the latest release while ThunderSTORM was last updated almost 10 years ago and is no longer compatible with the latest versions of ImageJ. The reliance on a handful of old community tools simply creates a ticking timebomb for reproducible research in SMLM.

This module aims to counteract some of these problems. It is designed to be easily installable and usable. Users need only supply file paths, camera parameters, and a threshold for single-molecule detection. The design is also simple and streamlined; only single-molecule localization is carried out and users are recommended to use the 'community favourites' for post-processing. This simplicity makes the code easy to maintain long-term. Finally, the module is fast, achieving similar speeds to ThunderSTORM and SMAP, due to its use of the Numba module.