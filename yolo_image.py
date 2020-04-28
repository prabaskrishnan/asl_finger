import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os


def detect_img(yolo, input_folder=os.getcwd(), output_folder=os.getcwd()):
    if not input_folder.endswith('/'):
        input_folder = input_folder + '/'

    if not output_folder.endswith('/'):
        output_folder = output_folder + '/'

    log_file = open(output_folder + "output.txt", "a")

    with os.scandir(input_folder) as root_dir:

        for path in root_dir:
            if path.is_file():
                if path.name.endswith(('.png', '.jpg', '.jpeg')):
                    print(input_folder + path.name)
                    image = Image.open(input_folder + path.name)
                    r_image, predict_dict = yolo.detect_image(image)
                    r_image.save(output_folder + 'output' + path.name)

                    for key, values in predict_dict.items():
                        predict_str = input_folder + path.name + ',' + str(values[0]) + ',' + str(values[1]) + ',' + str(values[2]) + ',' + str(values[3]) + ',' + str(values[4]) + ',' + str(values[5]) + '\n'
                        log_file.write(predict_str)


    yolo.close_session()
    log_file.close()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--input', type=str,
        help='Input images folder '
    )

    parser.add_argument(
        '--output', type=str,
        help='Output images folder '
    )

    FLAGS = parser.parse_args()

    detect_img(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)

