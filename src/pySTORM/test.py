import pySTORM.io as io
import pySTORM.fit as fit

def main():

    path = ''

    paths = io.get_movies(path)

    specs = io.get_camera_params()

if __name__ == "__main__":

    main()