from src.correct import CorrectImage


def main():
    pic = CorrectImage('data/')
    pic.detect_edges('initial.png')
    pic.hough_transform('initial.png', plot=True)


if __name__ == '__main__':
    main()
