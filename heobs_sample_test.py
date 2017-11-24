from network.download_google_drive import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.zip_utils import unzip_with_progress
from utils.make_predictions import *
import shutil

# google_download = DownloadGoogleDrive()
# test_zip = GoogleFile('0BzL8pCLanAIAd0hBV2NUVHpmckE',
#                        'heobs_large_dataset.zip', workspace('data/heobs_large_dataset.zip'))
#
# print "\n\n------------------------PREPARE PHRASE----------------------------\n\n"
#
# print "Starting download test file"
# google_download.download_file_from_google_drive(test_zip)
# print "Finish"
#
# print "Extracting test zip file"
# unzip_with_progress(test_zip.file_path, workspace("data"))
# print "Finish"


def test():
    print "\n\n------------------------TESTING PHRASE-----------------------------\n\n"
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    mean_proto = workspace("data/mean.binaryproto")

    caffe_deploy = workspace("caffe_model/caffenet_deploy.prototxt")

    py_render_template("template/" + template() + "/caffenet_deploy.template", caffe_deploy,
           num_output=Constant.NUMBER_OUTPUT, img_width=Constant.IMAGE_WIDTH, img_height=Constant.IMAGE_HEIGHT)
    classes = ["being", "heritage", "scenery"]
    mean_data = read_mean_data(mean_proto)
    net = read_model_and_weight(caffe_deploy, workspace("caffe_model/snapshot_iter_20000.caffemodel"))
    transformer = image_transformers(net, mean_data)
    prediction = making_predictions(workspace("data/extracted/test"), transformer, net)

    empty_dir("result")

    export_to_csv(prediction, workspace("result/test_result.csv"))
    export_data(prediction, workspace("data/extracted/test"), workspace("result/data"))
    show_result(classes, prediction)

    print "\n\n-------------------------FINISH------------------------------------\n\n"


if __name__ == '__main__':
    test()
