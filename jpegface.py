import collections
import sys
import xml.etree.ElementTree

import cv2
import numpy

BASIC_QUANT_TABLE = numpy.array([
    [16,  11,  10,  16,  24,  40,  51,  61],
    [12,  12,  14,  19,  26,  58,  60,  55],
    [14,  13,  16,  24,  40,  57,  69,  56],
    [14,  17,  22,  29,  51,  87,  80,  62],
    [18,  22,  37,  56,  68, 109, 103,  77],
    [24,  35,  55,  64,  81, 104, 113,  92],
    [49,  64,  78,  87, 103, 121, 120, 101],
    [72,  92,  95,  98, 112, 100, 103,  99]
])

def make_quant_table(quality):
    # Clamp quality to 1 <= quality <= 100
    quality = min(100, max(1, quality))

    # Scale factor is then defined piece-wise, and is inversely related to
    # quality.
    if quality < 50:
        scale_factor = 5000 // quality
    else:
        scale_factor = 200 - quality * 2

    out = numpy.clip((BASIC_QUANT_TABLE * scale_factor + 50) / 100, 0, 255)
                
    assert out.dtype == numpy.int32
    return out

Stage = collections.namedtuple('Stage', ['threshold', 'weak_classifiers'])
WeakClassifier = collections.namedtuple('WeakClassifier',
                          ['feature_idx', 'threshold', 'fail_val', 'pass_val'])
Rect = collections.namedtuple('Rect', ['x', 'y', 'w', 'h', 'weight'])

class Cascade(collections.namedtuple('_CascadeBase',
                                 ['width', 'height', 'stages', 'features'])):
    @staticmethod
    def _split_text_content(n):
        return n.text.strip().split(' ')

    @classmethod
    def load(cls, fname):
        root = xml.etree.ElementTree.parse(fname)

        width = int(root.find('./cascade/width').text.strip())
        height = int(root.find('./cascade/height').text.strip())

        stages = []
        for stage_node in root.findall('./cascade/stages/_'):
            stage_threshold = float(
                              stage_node.find('./stageThreshold').text.strip())
            weak_classifiers = []
            for classifier_node in stage_node.findall('weakClassifiers/_'):
                sp = cls._split_text_content(
                                       classifier_node.find('./internalNodes'))
                assert sp[0] == "0"
                assert sp[1] == "-1"
                feature_idx = int(sp[2])
                threshold = float(sp[3])

                sp = cls._split_text_content(
                                          classifier_node.find('./leafValues'))
                fail_val = float(sp[0])
                pass_val = float(sp[1])
                weak_classifiers.append(
                    WeakClassifier(feature_idx, threshold, fail_val, pass_val))
            stages.append(Stage(stage_threshold, weak_classifiers))

        features = []
        for feature_node in root.findall('./cascade/features/_'):
            feature = []
            for rect_node in feature_node.findall('./rects/_'):
                sp = cls._split_text_content(rect_node)
                x, y, w, h = (int(x) for x in sp[:4])
                weight = float(sp[4])
                feature.append(Rect(x, y, w, h, weight))
            features.append(feature)
                
        return cls(width, height, stages, features)

    def _feature_to_array(self, feature_idx):
        out = numpy.zeros((self.height, self.width))
        feature = self.features[feature_idx]
        for x, y, w, h, weight in feature:
            out[y:(y + h), x:(x + w)] += weight
        return out

    def detect(self, im):
        if im.shape != (self.height, self.width):
            im = cv2.resize(im, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)

        im = im.astype(numpy.float64)

        im /= numpy.std(im) * (self.height * self.width)
        #im /= 256. * (self.height * self.width)

        for stage_idx, stage in enumerate(self.stages):
            total = 0
            for classifier in stage.weak_classifiers:
                feature_array = self._feature_to_array(classifier.feature_idx)
                if numpy.sum(feature_array * im) >= classifier.threshold:
                    total += classifier.pass_val
                else:
                    total += classifier.fail_val
            if total < stage.threshold:
                print "Bailing out at stage {}".format(stage_idx)
                return -stage_idx
        return 1


def test_cascade_detect(im, cascade_file):
    my_cascade = Cascade.load(cascade_file)

    im = im[:]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    opencv_cascade = cv2.CascadeClassifier(cascade_file)
    objs = opencv_cascade.detectMultiScale(gray, 1.3, 5)
    for idx, (x, y, w, h) in enumerate(objs):
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite('out{:02d}.jpg'.format(idx), im)

        assert my_cascade.detect(gray[y:(y + h), x:(x + w)]) == 1

test_cascade_detect(cv2.imread(sys.argv[1]), sys.argv[2])

