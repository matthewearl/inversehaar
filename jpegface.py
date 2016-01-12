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


class SquareGrid(object):
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self.cell_names = ["pixel_{}_{}".format(x, y)
                           for y in range(height) for x in range(width)]

    @property
    def vec_size(self):
        return self._width * self._height

    def rect_vec(self, r):
        assert not r.tilted
        out = numpy.zeros((self._width, self._height), dtype=numpy.bool)
        out[r.y:r.y + r.h, r.x:r.x + r.w] = True
        return out.flatten()

    def render_vec(self, vec, im_width, im_height):
        im = (vec * 255.).astype(numpy.uint8).reshape(self._height,
                                                      self._width)
        return cv2.resize(im, (im_width, im_height),
                          interpolation=cv2.INTER_NEAREST)
        

class TiltedGrid(object):
    def __init__(self, width, height):
        self._width = width
        self._height = height
        
        self._cell_indices = {(d, x, y): 4 * ((width * y) + x) + d
                              for y in range(height)
                              for x in range(width)
                              for d in range(4)}
        self.cell_names = ['cell_{}_{}_{}'.format(x, y, "NESW"[d])
                           for y in range(height)
                           for x in range(width)
                           for d in range(4)]
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
            limits = numpy.matrix([[r.y, -(r.x + r.w), -(r.y + r.h), r.x]]).T
        else:
            dirs = numpy.matrix([[-1, 1], [-1, -1], [1, -1], [1, 1]])
            limits = numpy.matrix([[r.y - r.x,
                                    2 + -r.x -r.y - 2 * r.w,
                                    r.x - r.y - 2 * r.h,
                                    -2 + r.x + r.y]]).T

        return dirs, limits

    def rect_vec(self, r):
        dirs, limits = self._rect_to_bounds(r)
        out = numpy.all(numpy.array(dirs * numpy.matrix(self._cell_points).T)
                                                                     >= limits,
                        axis=0)
        return numpy.array(out)[0]

    def render_vec(self, vec, im_width, im_height):
        out_im = numpy.zeros((im_height, im_width), dtype=vec.dtype)

        tris = numpy.array([[[0, 0], [1, 0], [0.5, 0.5]],
                            [[1, 0], [1, 1], [0.5, 0.5]],
                            [[1, 1], [0, 1], [0.5, 0.5]],
                            [[0, 1], [0, 0], [0.5, 0.5]]])

        scale_factor = numpy.array([im_width / self._width,
                                    im_height / self._height])
        for y in reversed(range(self._height)):
            for x in range(self._width):
                for d in (2, 3, 1, 0):
                    points = (tris[d] + numpy.array([x, y])) * scale_factor 
                    cv2.fillConvexPoly(
                                   img=out_im,
                                   points=points.astype(numpy.int32),
                                   color=vec[self._cell_indices[d, x, y]])
        return out_im 

class Cascade(collections.namedtuple('_CascadeBase',
                 ['width', 'height', 'stages', 'features', 'tilted', 'grid'])):

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
            tilted_node = feature_node.find('./tilted')
            if tilted_node is not None:
                tilted = bool(int(tilted_node.text))
            else:
                tilted = False
            for rect_node in feature_node.findall('./rects/_'):
                sp = cls._split_text_content(rect_node)
                x, y, w, h = (int(x) for x in sp[:4])
                weight = float(sp[4])
                feature.append(Rect(x, y, w, h, tilted, weight))
            features.append(feature)

        tilted = any(r.tilted for f in features for r in f)

        if tilted:
            grid = TiltedGrid(width, height)
        else:
            grid = SquareGrid(width, height)

        stages = stages[:]

        return cls(width, height, stages, features, tilted, grid)

    def detect(self, im, epsilon=0.00001):
        im = im.astype(numpy.float64)

        im = cv2.resize(im, (self.width, self.height),
                        interpolation=cv2.INTER_AREA)

        im /= numpy.std(im) * (im.shape[1] * im.shape[0])
        #im /= 256. * (im.shape[1] * im.shape[0])

        debug_im = numpy.zeros(im.shape, dtype=numpy.float64)

        for stage_idx, stage in enumerate(self.stages):
            print "Stage {}".format(stage_idx)
            total = 0
            for classifier in stage.weak_classifiers:
                feature_array = self.grid.render_vec(
                    sum(self.grid.rect_vec(r) * r.weight
                               for r in self.features[classifier.feature_idx]),
                    im.shape[1], im.shape[0])
                t = self.features[classifier.feature_idx][0].tilted
                def do_write():
                    cv2.imwrite("feat{:02d}.png".format(classifier.feature_idx),
                                (feature_array + 2.) * (255. / 4))

                if (numpy.sum(feature_array * im) >=
                              classifier.threshold - epsilon):
                    if t and classifier.fail_val > classifier.pass_val:
                        print "{}: {} >?= {}".format(classifier.feature_idx,
                                                     numpy.sum(feature_array * im),
                                                     classifier.threshold - epsilon)
                        do_write()
                    total += classifier.pass_val
                else:
                    if t and classifier.fail_val < classifier.pass_val:
                        print "{}: {} >?= {}".format(classifier.feature_idx,
                                                     numpy.sum(feature_array * im),
                                                     classifier.threshold - epsilon)
                        do_write()
                    total += classifier.fail_val

            if total < stage.threshold - epsilon:
                print "Bailing out at stage {}".format(stage_idx)
                return -stage_idx
        return 1


class CascadeModel(object):
    def __init__(self, cascade):
        model = Model("Inverse haar cascade", docloud_context=docloud_context)

        cell_vars = [model.continuous_var(
                        name=cascade.grid.cell_names[i],
                        lb=0., ub=1.)
                        for i in range(cascade.grid.vec_size)]
        feature_vars = {idx: model.binary_var(name="feature_{}".format(idx))
                        for idx in range(len(cascade.features))}

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
                feature_vec = numpy.sum(cascade.grid.rect_vec(r) * r.weight
                             for r in cascade.features[classifier.feature_idx])
                feature_vec /= (cascade.width * cascade.height * 4.)
                thr = classifier.threshold
                feature_var = feature_vars[classifier.feature_idx]
                feature_val = sum(cell_vars[i] * feature_vec[i]
                                  for i in numpy.argwhere(
                                                  feature_vec != 0.).flatten())
                if classifier.pass_val >= classifier.fail_val:
                    big_num = 0.1 + thr - numpy.sum(numpy.min(
                             [feature_vec, numpy.zeros(feature_vec.shape)],
                             axis=0))
                    model.add_constraint(feature_val - feature_var * big_num >=
                                                                 thr - big_num)
                else:
                    big_num = 0.1 + numpy.sum(numpy.max(
                             [feature_vec, numpy.zeros(feature_vec.shape)],
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

        self.cascade = cascade
        self.cell_vars = cell_vars
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
    #cascade_model.set_best_fit_objective()
    #cascade_model.model.set_time_limit(1200.)

    cascade_model.model.print_information()
    cascade_model.model.export_as_lp(basename='docplex_%s', path='/home/matt')

    if not cascade_model.model.solve():
        raise Exception("Failed to find solution")
    cascade_model.model.report()

    sol_vec = numpy.array([v.solution_value for v in cascade_model.cell_vars])
    return cascade_model.cascade.grid.render_vec(sol_vec,
                                                 10 * cascade.width,
                                                 10 * cascade.height)
    

im = find_min_face(sys.argv[1])
cv2.imwrite("out.png", im)

cascade = Cascade.load(sys.argv[1])
assert cascade.detect(im) == 1

