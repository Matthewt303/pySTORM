import pySTORM.io as io
import pySTORM.fit as fit

def main():

    path = 'C:/users/matth/Downloads/test_module'

    paths = io.get_movies(path)
    print(paths)

    specs = io.get_camera_params(pixel_size=97.5,
                                 adu=0.59,
                                 offset=100,
                                 gain=100)
    
    print(specs)
    
    x = fit.localize(paths, specs, threshold=3.0)
    return x[0:5, :]

#if __name__ == "__main__":

#    main()
