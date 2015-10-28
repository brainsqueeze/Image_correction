from src.correct import CorrectImage

pic = CorrectImage('data/')
pic.detect_edges('initial.png')
pic.hough_transform('initial.png', plot=True)
