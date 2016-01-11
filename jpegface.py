import collections
import sys
import xml.etree.ElementTree

import cv2
import numpy

from docplex.mp.context import DOcloudContext
from docplex.mp.environment import Environment
from docplex.mp.model import Model

DOCLOUD_URL = 'https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/'
docloud_context = DOcloudContext.make_default_context(DOCLOUD_URL)
docloud_context.print_information()
env = Environment()
env.print_information()


Stage = collections.namedtuple('Stage', ['threshold', 'weak_classifiers'])
WeakClassifier = collections.namedtuple('WeakClassifier',
                          ['feature_idx', 'threshold', 'fail_val', 'pass_val'])
Rect = collections.namedtuple('Rect',
                              ['x', 'y', 'w', 'h', 'tilted', 'weight'])

class TiltedGrid(object):
    def __init__(self, width, height):
        self._width = width
        self._height = height
        
        self._cell_indices = {(d, x, y): 4 * ((width * y) + x) + d
                              for y in range(height)
                              for x in range(width)
                              for d in range(4)}
        self._cell_points = numpy.zeros((width * height * 4, 2))

        for y in range(height):
            for x in range(width):
                self._cell_points[self._cell_indices[0, x, y], :] = \
                        numpy.array([x + 0.5, y + 0.25])
                self._cell_points[self._cell_indices[1, x, y], :] = \
                        numpy.array([x + 0.75, y + 0.5])
                self._cell_points[self._cell_indices[2, x, y], :] = \
                        numpy.array([x + 0.5, y + 0.75])
                self._cell_points[self._cell_indices[3, x, y], :] = \
                        numpy.array([x + 0.25, y + 0.5])

    @property
    def vec_size(self):
        return self._width * self._height * 4

    def _rect_to_bounds(self, r):
        if not r.tilted:
            dirs = numpy.matrix([[0, 1], [-1, 0], [0, -1], [1, 0]])
        else:
            dirs = numpy.matrix([[-1, 1], [-1, -1], [1, -1], [1, 1]])
        limits = numpy.matrix([[r.y, -(r.x + r.w), -(r.y + r.h), r.x]]).T

        return dirs, limits

    def rect_vec(self, r):
        dirs, limits = self._rect_to_bounds(r)
        out = numpy.all(numpy.array(dirs * numpy.matrix(self._cell_points).T)
                                                                     >= limits,
                        axis=0)
        return numpy.array(out)[0]
        
    def render_vec(self, vec, im_width, im_height):
        out_im = numpy.zeros((im_height, im_width), dtype=numpy.uint8)

        vec = (vec * 255.).astype(numpy.uint8)

        tris = numpy.array([[[0, 0], [1, 0], [0.5, 0.5]],
                            [[1, 0], [1, 1], [0.5, 0.5]],
                            [[1, 1], [0, 1], [0.5, 0.5]],
                            [[0, 1], [0, 0], [0.5, 0.5]]])

        scale_factor = numpy.array([im_width / self._width,
                                    im_height / self._height])
        for y in range(self._height):
            for x in range(self._width):
                for d in range(4):
                    points = (tris[d] + numpy.array([x, y])) * scale_factor 
                    cv2.fillConvexPoly(
                                   img=out_im,
                                   points=points.astype(numpy.int32),
                                   color=int(vec[self._cell_indices[d, x, y]]))
        return out_im

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

        stages = stages[:]

        return cls(width, height, stages, features)

    def feature_to_array(self, feature_idx):
        out = numpy.zeros((self.height, self.width))
        feature = self.features[feature_idx]
        for x, y, w, h, weight in feature:
            out[y:(y + h), x:(x + w)] += weight
        return out

    def detect(self, im, epsilon=0.00001):
        if im.shape != (self.height, self.width):
            im = cv2.resize(im, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)

        cv2.imwrite('t.png', im)

        im = im.astype(numpy.float64)

        #im /= numpy.std(im) * (self.height * self.width)
        im /= 256. * (self.height * self.width)

        for stage_idx, stage in enumerate(self.stages):
            total = 0
            for classifier in stage.weak_classifiers:
                feature_array = self.feature_to_array(classifier.feature_idx)
                if (numpy.sum(feature_array * im) >=
                                               classifier.threshold - epsilon):
                    total += classifier.pass_val
                else:
                    total += classifier.fail_val

            if total < stage.threshold - epsilon:
                print "Bailing out at stage {}".format(stage_idx)
                return -stage_idx
        return 1


class CascadeModel(object):
    def __init__(self, cascade, symmetrical=False, max_delta=None):
        model = Model("Inverse haar cascade", docloud_context=docloud_context)

        pixel_vars = {(x, y): model.continuous_var(
                               name="pixel_{}_{}".format(x, y), lb=0.0, ub=1.0)
                      for y in range(cascade.height)
                      for x in range(cascade.width)}
        feature_vars = {idx: model.binary_var(name="feature_{}".format(idx))
                        for idx in range(len(cascade.features))}

        """
        temp_img = cv2.imread('t.png', cv2.IMREAD_GRAYSCALE).astype(
                                                          numpy.float64) / 256.
        for y in range(cascade.height):
            for x in range(cascade.width):
                model.add_constraint(pixel_vars[x, y] - temp_img[y, x] == 0.)
        """

        for stage in cascade.stages:
            # If the classifier's pass value is greater than its fail value,
            # then add a constraint equivalent to the following:
            #   
            #   feature var set => corresponding feature is present in image
            #
            # This is sufficient because if a feature is present, but the
            # corresponding feature var is not set, then setting the feature
            # var will only help the stage constraint pass (due to the feature
            # var appearing with a positive coefficient there).
            #   
            # Conversely, if the classifier's pass vlaue is less than its fail
            # value, add a constraint equivalent to:
            #
            #   corresponding feature is present in image => feature var set
            for classifier in stage.weak_classifiers:
                feature_array = cascade.feature_to_array(
                                                        classifier.feature_idx)
                feature_array /= (cascade.width * cascade.height)
                thr = classifier.threshold
                feature_var = feature_vars[classifier.feature_idx]
                feature_val = sum(pixel_vars[x, y] * feature_array[y, x]
                                  for y in range(cascade.height)
                                  for x in range(cascade.width)
                                  if feature_array[y, x] != 0.)
                if classifier.pass_val >= classifier.fail_val:
                    big_num = 0.1 + thr - numpy.sum(numpy.min(
                             [feature_array, numpy.zeros(feature_array.shape)],
                             axis=0))
                    model.add_constraint(feature_val - feature_var * big_num >=
                                                                 thr - big_num)
                else:
                    big_num = 0.1 + numpy.sum(numpy.max(
                             [feature_array, numpy.zeros(feature_array.shape)],
                             axis=0)) - thr
                    model.add_constraint(feature_val - feature_var * big_num <=
                                                                           thr)

            # Enforce that the sum of features present in this stage exceeds
            # the stage threshold.
            fail_val_total = sum(c.fail_val for c in stage.weak_classifiers)
            adjusted_stage_threshold = stage.threshold
            model.add_constraint(sum((c.pass_val - c.fail_val) *
                                                    feature_vars[c.feature_idx] 
                         for c in stage.weak_classifiers) >=
                                     adjusted_stage_threshold - fail_val_total)

        # Constrain adjacent pixels to be within a given range.
        if max_delta is not None:
            for y in range(cascade.height - 1):
                for x in range(cascade.width - 1):
                    model.add_constraint(
                        pixel_vars[x, y] - pixel_vars[x + 1, y] <= max_delta)
                    model.add_constraint(
                        pixel_vars[x, y] - pixel_vars[x + 1, y] >= -max_delta)
                    model.add_constraint(
                        pixel_vars[x, y] - pixel_vars[x, y + 1] <= max_delta)
                    model.add_constraint(
                        pixel_vars[x, y] - pixel_vars[x, y + 1] >= -max_delta)

        if symmetrical:
            for y in range(cascade.height):
                for x in range(cascade.width // 2):
                    model.add_constraint(
                        pixel_vars[x, y] -
                        pixel_vars[cascade.width - x - 1, y] == 0.)

        self.cascade = cascade
        self.pixel_vars = pixel_vars
        self.feature_vars = feature_vars
        self.model = model

    def set_best_fit_objective(self):
        self.model.set_objective("max",
                    sum((c.pass_val - c.fail_val) *
                                               self.feature_vars[c.feature_idx] 
                        for s in self.cascade.stages
                        for c in s.weak_classifiers))


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
        print "{} {} {} {}".format(x, y, w, h)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite('out{:02d}.jpg'.format(idx), im)

        assert my_cascade.detect(gray[y:(y + h), x:(x + w)]) == 1


def find_min_face(cascade_file):
    cascade = Cascade.load(cascade_file)
    
    cascade_model = CascadeModel(cascade)
    #cascade_model.model.set_objective("min",
    #                         sum(v for v in cascade_model.pixel_vars.values()))
    cascade_model.set_best_fit_objective()
    cascade_model.model.set_time_limit(1 * 3600.)

    cascade_model.model.print_information()
    cascade_model.model.export_as_lp(basename='docplex_%s', path='/home/matt')

    if not cascade_model.model.solve():
        raise Exception("Failed to find solution")
    cascade_model.model.report()

    sol = numpy.array([[cascade_model.pixel_vars[x, y].solution_value
                        for x in range(cascade.width)]
                       for y in range(cascade.height)])

    return sol
    

#test_cascade_detect(cv2.imread(sys.argv[1]), sys.argv[2])
#raise SystemExit
im = find_min_face(sys.argv[1])
im *= 256.
im_resized = cv2.resize(im, (im.shape[1] * 10, im.shape[0] * 10),
                        interpolation=cv2.INTER_NEAREST)
cv2.imwrite("out.png", im_resized)

cascade = Cascade.load(sys.argv[1])
assert cascade.detect(im) == 1

