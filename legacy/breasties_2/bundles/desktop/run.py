import multiprocessing

from breasties_2.app.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main(exec_mode="desktop")
