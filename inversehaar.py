# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Invert OpenCV haar cascades.

"""


__all__ = (
    'Cascade',
    'inverse_haar',
)


import collections
import sys
import xml.etree.ElementTree

import cv2
import numpy

from docplex.mp.context import DOcloudContext
from docplex.mp.environment import Environment
from docplex.mp.model import Model


DOCLOUD_URL = 'https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/'


# Grid classes

class Grid(object):
    """
    A division of an image area into cells.

    For example, `SquareGrid` divides the image into pixels.

    Cell values are represented with "cell vectors", so for example,
    `Grid.render_cell_vec` will take a cell vector and produce an image.

    """

    @property
    def num_cells(self):
        """The number of cells in this grid"""
        raise NotImplementedError

    def rect_to_cell_vec(self, r):
        """
        Return a boolean cell vector corresponding with the input rectangle.

        Elements of the returned vector are True if and only if the
        corresponding cells fall within the input rectangle

        """
        raise NotImplementedError

    def render_cell_vec(self, vec, im_width, im_height):
        """Render an image, using a cell vector and image dimensions."""
        raise NotImplementedError


class SquareGrid(Grid):
    """
    A grid where cells correspond with pixels.

    This grid type is used for cascades which do not contain diagonal features.

    """
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self.cell_names = ["pixel_{}_{}".format(x, y)
                           for y in range(height) for x in range(width)]

    @property
    def num_cells(self):
        return self._width * self._height

    def rect_to_cell_vec(self, r):
        assert not r.tilted
        out = numpy.zeros((self._width, self._height), dtype=numpy.bool)
        out[r.y:r.y + r.h, r.x:r.x + r.w] = True
        return out.flatten()

    def render_cell_vec(self, vec, im_width, im_height):
        im = vec.reshape(self._height, self._width)
        return cv2.resize(im, (im_width, im_height),
                          interpolation=cv2.INTER_NEAREST)
        

class TiltedGrid(Grid):
    """
    A square grid, but each square consists of 4 cells.

    The squares are cut diagonally, resulting in a north, east, south and west
    triangle for each cell.

    This grid type is used for cascades which contain diagonal features: The
    idea is that the area which a diagonal feature should be integrated can be
    represented exactly by this structure.

    Unfortunately, this is not quite accurate: OpenCV's trainer and detector
    always resizes its images so that pixels correspond with one grid cell. As
    such cascades which contain diagonal features will not be accurately
    inverted by this script, however, they will have more detail as a result of
    the grid square subdivision.

    """
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
    def num_cells(self):
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

    def rect_to_cell_vec(self, r):
        dirs, limits = self._rect_to_bounds(r)
        out = numpy.all(numpy.array(dirs * numpy.matrix(self._cell_points).T)
                                                                     >= limits,
                        axis=0)
        return numpy.array(out)[0]

    def render_cell_vec(self, vec, im_width, im_height):
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


# Cascade definition

class Stage(collections.namedtuple('_StageBase',
                                   ['threshold', 'weak_classifiers'])):
    """
    A stage in an OpenCV cascade.

    .. attribute:: weak_classifiers

        A list of weak classifiers in this stage.

    .. attribute:: threshold

        The value that the weak classifiers must exceed for this stage to pass.

    """


class WeakClassifier(collections.namedtuple('_WeakClassifierBase',
                        ['feature_idx', 'threshold', 'fail_val', 'pass_val'])):
    """
    A weak classifier in an OpenCV cascade.

    .. attribute:: feature_idx

        Feature associated with this classifier.

    .. attribute:: threshold

        The value that this feature dotted with the input image must exceed for the
        feature to have passed.

    .. attribute:: fail_val

        The value contributed to the stage threshold if this classifier fails.

    .. attribute:: pass_val

        The value contributed to the stage threshold if this classifier passes.

    """


class Rect(collections.namedtuple('_RectBase',
                                  ['x', 'y', 'w', 'h', 'tilted', 'weight'])):
    """
    A rectangle in an OpenCV cascade.

    Two or more of these make up a feature.

    .. attribute:: x, y
        
        Coordinates of the rectangle.

    .. attribute:: w, h

        Width and height of the rectangle, respectively.

    .. attribute:: tilted

        If true, the rectangle is to be considered rotated 45 degrees clockwise
        about its top-left corner. (+X is right, +Y is down.)
        
    .. attribute:: weight

        The value this rectangle contributes to the feature.

    """


class Cascade(collections.namedtuple('_CascadeBase',
                 ['width', 'height', 'stages', 'features', 'tilted', 'grid'])):
    """
    Pythonic interface to an OpenCV cascade file.

    .. attribute:: width
        
        Width of the cascade grid.

    .. attribute:: height

        Height of the cascade grid.

    .. attribute:: stages

        List of :class:`.Stage` objects.

    .. attribute:: features

        List of features. Each feature is in turn a list of :class:`.Rect`s. 

    .. attribute:: tilted

        True if any of the features are tilted.

    .. attribute:: grid

        A :class:`.Grid` object suitable for use with the cascade.

    """
    @staticmethod
    def _split_text_content(n):
        return n.text.strip().split(' ')

    @classmethod
    def load(cls, fname):
        """
        Parse an OpenCV haar cascade XML file.

        """
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
                if sp[0] != "0" or sp[1] != "-1":
                    raise Exception("Only simple cascade files are supported")
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

    def detect(self, im, epsilon=0.00001, scale_by_std_dev=False):
        """
        Apply the cascade forwards on a potential face image.

        The algorithm is relatively slow compared to the integral image
        implementation, but is relatively terse and consequently useful for
        debugging.

        :param im:
            
            Image to apply the detector to.

        :param epsilon:

            Maximum rounding error to account for. This biases the classifier
            and stage thresholds towards passing. As a result, passing too
            large a value may result in false positive detections.

        :param scale_by_std_dev:

            If true, divide the input image by its standard deviation before
            processing. This simulates OpenCV's algorithm, however the reverse
            haar mapping implemented by this script does not account for the
            standard deviation divide, so to get the forward version of
            `inverse_haar`, pass False.

        """
        im = im.astype(numpy.float64)

        im = cv2.resize(im, (self.width, self.height),
                        interpolation=cv2.INTER_AREA)

        scale_factor = numpy.std(im) if scale_by_std_dev else 256.
        im /= scale_factor * (im.shape[1] * im.shape[0])

        debug_im = numpy.zeros(im.shape, dtype=numpy.float64)

        for stage_idx, stage in enumerate(self.stages):
            total = 0
            for classifier in stage.weak_classifiers:
                feature_array = self.grid.render_cell_vec(
                    sum(self.grid.rect_to_cell_vec(r) * r.weight
                               for r in self.features[classifier.feature_idx]),
                    im.shape[1], im.shape[0])
                if classifier.pass_val > classifier.fail_val:
                    thr = classifier.threshold - epsilon
                else:
                    thr = classifier.threshold + epsilon
                if numpy.sum(feature_array * im) >= thr:
                    total += classifier.pass_val
                else:
                    total += classifier.fail_val

            if total < stage.threshold - epsilon:
                return -stage_idx
        return 1


class CascadeModel(Model):
    """
    Model of the variables and constraints associated with a Haar cascade.

    This is in fact a wrapper around a docplex model.

    .. attribute:: cell_vars

        List of variables corresponding with the cells in the cascade's grid.

    .. attribute:: feature_vars

        Dict of feature indices to binary variables. Each variable represents
        whether the corresponding feature is present.

    .. attribute:: cascade

        The underlying :class:`.Cascade`.

    """
    def __init__(self, cascade, docloud_context):
        """Make a model from a :class:`.Cascade`."""
        super(CascadeModel, self).__init__("Inverse haar cascade",
                                           docloud_context=docloud_context)

        cell_vars = [self.continuous_var(
                        name=cascade.grid.cell_names[i],
                        lb=0., ub=1.)
                        for i in range(cascade.grid.num_cells)]
        feature_vars = {idx: self.binary_var(name="feature_{}".format(idx))
                        for idx in range(len(cascade.features))}

        for stage in cascade.stages:
            # Add constraints for the feature vars.
            #
            # If the classifier's pass value is greater than its fail value,
            # then add a constraint equivalent to the following:
            #   
            #   feature var set => corresponding feature is present in image
            #
            # Conversely, if the classifier's pass vlaue is less than its fail
            # value, add a constraint equivalent to:
            #
            #   corresponding feature is present in image => feature var set
            for classifier in stage.weak_classifiers:
                feature_vec = numpy.sum(
                             cascade.grid.rect_to_cell_vec(r) * r.weight
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
                    self.add_constraint(feature_val - feature_var * big_num >=
                                                                 thr - big_num)
                else:
                    big_num = 0.1 + numpy.sum(numpy.max(
                             [feature_vec, numpy.zeros(feature_vec.shape)],
                             axis=0)) - thr
                    self.add_constraint(feature_val - feature_var * big_num <=
                                                                           thr)

            # Enforce that the sum of features present in this stage exceeds
            # the stage threshold.
            fail_val_total = sum(c.fail_val for c in stage.weak_classifiers)
            adjusted_stage_threshold = stage.threshold
            self.add_constraint(sum((c.pass_val - c.fail_val) *
                                                    feature_vars[c.feature_idx] 
                         for c in stage.weak_classifiers) >=
                                     adjusted_stage_threshold - fail_val_total)

        self.cascade = cascade
        self.cell_vars = cell_vars
        self.feature_vars = feature_vars

    def set_best_objective(self):
        """
        Amend the model with an objective.

        The objective used is to maximise the score from each stage of the
        cascade.

        """
        self.set_objective("max",
                    sum((c.pass_val - c.fail_val) *
                                               self.feature_vars[c.feature_idx] 
                        for s in self.cascade.stages
                        for c in s.weak_classifiers))


def inverse_haar(cascade, optimize=False, time_limit=None,
                 docloud_context=None, lp_path=None):
    """
    Invert a haar cascade.

    :param cascade:
        A :class:`.Cascade` to invert.

    :param optimize:
        Attempt to find an optimal solution, rather than just a feasible
        solution.

    :param time_limit:
        Maximum time to allow the solver to work, in seconds.

    :param docloud_context:
        :class:`docplex.mp.context.DOcloudContext` to use for solving.

    :param lp_path:
        File to write the LP constraints to. Useful for debugging. (Optional).

    """
    
    cascade_model = CascadeModel(cascade, docloud_context)
    if optimize:
        cascade_model.set_best_objective()
    if time_limit is not None:
        cascade_model.set_time_limit(time_limit)

    cascade_model.print_information()
    if lp_path:
        cascade_model.export_as_lp(path=lp_path)

    if not cascade_model.solve():
        raise Exception("Failed to find solution")
    sol_vec = numpy.array([v.solution_value for v in cascade_model.cell_vars])
    im = cascade_model.cascade.grid.render_cell_vec(sol_vec,
                                                    10 * cascade.width,
                                                    10 * cascade.height)
    im = (im * 255.).astype(numpy.uint8)

    return im


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=
            'Inverse haar feature object detection')
    parser.add_argument('-c', '--cascade', type=str, required=True,
                       help='OpenCV cascade file to be reversed')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output image name')
    parser.add_argument('-t', '--time-limit', type=float, default=None,
                       help='Maximum time to allow the solver to work, in '
                            'seconds')
    parser.add_argument('-O', '--optimize', action='store_true',
                        help='Try and find an optimal solution, rather than '
                             'just a feasible solution')
    parser.add_argument('-C', '--check', action='store_true',
                        help='Check the result against the (forward) cascade')
    parser.add_argument('-l', '--lp-path',type=str, default=None,
                        help='File to write LP constraints to.')
    args = parser.parse_args()

    print "Loading cascade..."
    cascade = Cascade.load(args.cascade)

    docloud_context = DOcloudContext.make_default_context(DOCLOUD_URL)
    docloud_context.print_information()
    env = Environment()
    env.print_information()

    print "Solving..."
    im = inverse_haar(cascade,
                      optimize=args.optimize,
                      time_limit=args.time_limit,
                      docloud_context=docloud_context,
                      lp_path=args.lp_path)

    cv2.imwrite(args.output, im)
    print "Wrote {}".format(args.output)

    if args.check:
        print "Checking..."
        ret = cascade.detect(im)
        if ret != 1:
            print "Image failed the forward cascade at stage {}".format(-ret)

