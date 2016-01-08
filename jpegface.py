import collections
import sys
import xml.etree.ElementTree

import cv2
import numpy
import pulp


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

        stages = stages[:3]

        return cls(width, height, stages, features)

    def feature_to_array(self, feature_idx):
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

        #im /= numpy.std(im) * (self.height * self.width)
        im /= 256. * (self.height * self.width)

        for stage_idx, stage in enumerate(self.stages):
            total = 0
            for classifier in stage.weak_classifiers:
                feature_array = self.feature_to_array(classifier.feature_idx)
                if numpy.sum(feature_array * im) >= classifier.threshold:
                    total += classifier.pass_val
                else:
                    total += classifier.fail_val

            if total < stage.threshold:
                print "Bailing out at stage {}".format(stage_idx)
                return -stage_idx
        return 1


class CascadeConstraints(object):
    def __init__(self, cascade):
        pixel_vars = {(x, y): pulp.LpVariable("pixel_{}_{}".format(x, y),
                                              lowBound=0.0,
                                              upBound=1.0)
                      for y in range(cascade.height)
                      for x in range(cascade.width)}
        feature_vars = {idx: pulp.LpVariable("feature_{}".format(idx),
                                             cat=pulp.LpBinary)
                                for idx in range(len(cascade.features))}

        constraints = []
        for stage in cascade.stages:
            # Feature var implies the feature dotted with the pixel values exceeds
            # the classifier threshold.
            for classifier in stage.weak_classifiers:
                feature_array = cascade.feature_to_array(
                                                        classifier.feature_idx)
                if classifier.pass_val < classifier.fail_val:
                    feature_array = -feature_array
                constraints.append(sum(pixel_vars[x, y] * feature_array[y, x] / 
                                               (cascade.width * cascade.height)
                             for y in range(cascade.height)
                             for x in range(cascade.width)
                             if feature_array[y, x] != 0.) *
                         (1. / classifier.threshold) -
                         feature_vars[classifier.feature_idx] >= 0)

            # Enforce that the sum of features present in this stage exceeds
            # the stage threshold.
            fail_val_total = sum(min(c.fail_val, c.pass_val)
                                               for c in stage.weak_classifiers)
            constraints.append(sum(abs(c.pass_val - c.fail_val) *
                                                    feature_vars[c.feature_idx] 
                         for c in stage.weak_classifiers) >=
                                              stage.threshold - fail_val_total)

        self.cascade = cascade
        self.pixel_vars = pixel_vars
        self.feature_vars = feature_vars
        self._constraints = constraints

    def __iter__(self):
        return iter(self._constraints)


def test_cascade_detect(im, cascade_file):
    my_cascade = Cascade.load(cascade_file)

    im = im[:]
    if len(im.shape) == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        assert len(im.shape) == 2
        gray = im

    opencv_cascade = cv2.CascadeClassifier(cascade_file)
    objs = opencv_cascade.detectMultiScale(gray, 1.3, 5)
    for idx, (x, y, w, h) in enumerate(objs):
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite('out{:02d}.jpg'.format(idx), im)

        assert my_cascade.detect(gray[y:(y + h), x:(x + w)]) == 1


def find_min_face(cascade_file):
    cascade = Cascade.load(cascade_file)
    
    constraints = CascadeConstraints(cascade)

    prob = pulp.LpProblem("Reverse haar cascade", pulp.LpMinimize)
    prob += sum(v for v in constraints.pixel_vars.values())
    for c in constraints:
        prob += c

    prob.writeLP("min_face.lp")
    prob.solve()
    print "Status: {}".format(pulp.LpStatus[prob.status])
    if prob.status != pulp.LpStatusOptimal:
        raise Exception("Failed to find solution")

    for v in prob.variables():
        print "{}: {}".format(v.name, v.varValue)

    sol = numpy.array([
             [constraints.pixel_vars[x, y].varValue
                                                 for x in range(cascade.width)]
                  for y in range(cascade.height)])

    numpy.set_printoptions(precision=4)
    print sol

    return sol
    

#test_cascade_detect(cv2.imread(sys.argv[1]), sys.argv[2])
im = find_min_face(sys.argv[1])
im *= 256.
im_resized = cv2.resize(im, (im.shape[1] * 10, im.shape[0] * 10),
                        interpolation=cv2.INTER_NEAREST)
cv2.imwrite("out.png", im_resized)

cascade = Cascade.load(sys.argv[1])
assert cascade.detect(im) == 1

