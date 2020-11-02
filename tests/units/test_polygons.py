# -*- coding: utf-8 -*-
# @Time : 2020/7/30 4:40 下午
# @Author : SongWeinan
# @Software: PyCharm
# 醉后不知天在水，满船清梦压星河。
# ======================================================================================================================
import unittest
import numpy as np
from structures import Polygons


class PolygonsTestCase(unittest.TestCase):
    def test_len_0(self):
        contours = []
        polygons = Polygons(contours)
        self.assertEqual(len(polygons), 0)

    def test_wrong_shape(self):
        contours = [np.zeros([4, 2]), np.zeros([2, 2])]
        with self.assertRaises(ValueError):
            polygons = Polygons(contours)
        contours = [np.zeros([4, 1])]
        with self.assertRaises(ValueError):
            polygons = Polygons(contours)
        contours = [np.zeros([4, 1, 2])]
        with self.assertRaises(ValueError):
            polygons = Polygons(contours)

    def test_has_field(self):
        contours = [np.zeros([4, 2]), np.zeros([4, 2]), np.zeros([4, 2])]
        polygons = Polygons(contours)
        self.assertTrue(polygons.has_field('num_points'))
        self.assertFalse(polygons.has_field('abc'))

    def test_add_field_fifferent_len(self):
        contours = [np.zeros([4, 2]), np.zeros([4, 2]), np.zeros([4, 2])]
        polygons = Polygons(contours)
        field = ['s', 's']
        with self.assertRaises(ValueError):
            polygons.add_field('field', field)

    def test_add_field_fifferent_type(self):
        contours = [np.zeros([4, 2]), np.zeros([4, 2]), np.zeros([4, 2])]
        polygons = Polygons(contours)
        field = [0, 's', 0.001]
        with self.assertRaises(TypeError):
            polygons.add_field('field', field)

    def test_add_field_alread_in(self):
        contours = [np.zeros([4, 2]), np.zeros([4, 2]), np.zeros([4, 2])]
        polygons = Polygons(contours)
        field = ['s', 's', 's']
        with self.assertRaises(ValueError):
            polygons.add_field('num_points', field)

    def test_set_field_not_exist(self):
        contours = [np.zeros([4, 2]), np.zeros([4, 2]), np.zeros([4, 2])]
        polygons = Polygons(contours)
        field = ['s', 's', 's']
        with self.assertRaises(ValueError):
            polygons.set_field('field', field)

    def test_get_field_normal(self):
        contours = [np.zeros([4, 2]), np.zeros([4, 2]), np.zeros([4, 2])]
        polygons = Polygons(contours)
        num_points = polygons.get_field('num_points')
        num_points[0] = 1
        self.assertEqual(num_points, [1, 4, 4])
        self.assertEqual(polygons.get_field('num_points'), [4, 4, 4])

    def test_get_field_numpy(self):
        contours = [np.zeros([4, 2])]
        polygons = Polygons(contours)
        np_field = [np.zeros([4])]
        polygons.add_field('np_field', np_field)
        np_field_modified = polygons.get_field('np_field')
        np_field_modified[0][0] = 1
        np.testing.assert_equal(np_field_modified, np.array([[1, 0, 0, 0]]))
        np_field_original = polygons.get_field('np_field')
        np.testing.assert_equal(np_field_original, np.array([[0, 0, 0, 0]]))

    def test_get_data(self):
        contours = [np.zeros([4, 2])]
        polygons = Polygons(contours)
        contours_modified = polygons.get_contours()
        contours_modified[0][0][0] = 1
        np.testing.assert_equal(contours_modified, np.array([[[1, 0], [0, 0], [0, 0], [0, 0]]]))
        contours_original = polygons.get_contours()
        np.testing.assert_equal(contours_original, np.array([[[0, 0], [0, 0], [0, 0], [0, 0]]]))

    def test_set_data(self):
        contours = [np.zeros([4, 2])]
        polygons = Polygons(contours)
        polygons.set_contours([np.ones([4, 2])])
        contours = polygons.get_contours()
        np.testing.assert_equal(contours, np.array([[[1, 1], [1, 1], [1, 1], [1, 1]]]))

    def test_copy(self):
        contours = [np.zeros([4, 2]), np.zeros([4, 2]), np.zeros([4, 2])]
        polygons = Polygons(contours)
        field = ['s', 's', 's']
        polygons.add_field('field', field)
        p = polygons.copy()
        np.testing.assert_equal(p.get_contours(), contours)
        self.assertEqual(p.get_field('num_points'), [4, 4, 4])
        self.assertEqual(p.get_field('field'), ['s', 's', 's'])
        p.set_contours([np.ones([4, 2]), np.ones([4, 2]), np.ones([4, 2])])
        p.set_field('field', ['w', 'w', 'w'])
        np.testing.assert_equal(polygons.get_contours(), contours)
        self.assertEqual(polygons.get_field('field'), ['s', 's', 's'])

    def test_scale_and_pad(self):
        contours = [np.array([[2, 1], [3, 5], [4, 2], [1, 3]]),
                    np.array([[0, 0], [1, 0], [1, 1]]),
                    np.array([[1, 0], [1, 2], [0, 1], [2, 1], [2, 2]])]
        polygons = Polygons(contours)
        polygons.scale_and_pad([2, 2], [1, 0])
        target_contours = [
            np.array([[5, 2], [7, 10], [9, 4], [3, 6]]),
            np.array([[1, 0], [3, 0], [3, 2]]),
            np.array([[3, 0], [3, 4], [1, 2], [5, 2], [5, 4]])
        ]
        np.testing.assert_equal(polygons.get_contours(), target_contours)

    def test_scale_and_pad_with_single_scale(self):
        contours = [np.array([[2, 1], [3, 5], [4, 2], [1, 3]]),
                  np.array([[0, 0], [1, 0], [1, 1]]),
                  np.array([[1, 0], [1, 2], [0, 1], [2, 1], [2, 2]])]
        polygons = Polygons(contours)
        polygons.scale_and_pad(2, [1, 0])
        target_contours = [
            np.array([[5, 2], [7, 10], [9, 4], [3, 6]]),
            np.array([[1, 0], [3, 0], [3, 2]]),
            np.array([[3, 0], [3, 4], [1, 2], [5, 2], [5, 4]])
        ]
        np.testing.assert_equal(polygons.get_contours(), target_contours)

    def test_scale_and_pad_recover(self):
        contours = [np.array([[5, 2], [7, 10], [9, 4], [3, 6]]),
                    np.array([[1, 0], [3, 0], [3, 2]]),
                    np.array([[3, 0], [3, 4], [1, 2], [5, 2], [5, 4]])]
        polygons = Polygons(contours)
        polygons.scale_and_pad([0.5, 0.5], [-1, 0], recover=True)
        target_contours = [np.array([[2, 1], [3, 5], [4, 2], [1, 3]]),
                           np.array([[0, 0], [1, 0], [1, 1]]),
                           np.array([[1, 0], [1, 2], [0, 1], [2, 1], [2, 2]])]
        np.testing.assert_equal(polygons.get_contours(), target_contours)

    def test_scale_and_pad_0(self):
        contours = []
        polygons = Polygons(contours)
        polygons.scale_and_pad([2, 2], [1, 0])
        np.testing.assert_equal(polygons.get_contours(), [])

    def test_fliter_out(self):
        contours = [
            np.array([[20, 10], [10, 30], [30, 50], [40, 20]]),
            np.array([[0, 0], [10, 0], [10, 10]]),
            np.array([[10, 0], [10, 20], [0, 10], [20, 10], [20, 20]])
        ]
        polygons = Polygons(contours)
        window = np.array([[0, 0], [20, 20]])
        polygons.fliter_out(window)
        np.testing.assert_equal(polygons.get_field('training_tag'), np.array([False, True, False]))
        np.testing.assert_equal(polygons.get_contours(), contours)

    def test_fliter_out_with_exist(self):
        contours = [
            np.array([[20, 10], [10, 30], [30, 50], [40, 20]]),
            np.array([[0, 0], [10, 0], [10, 10]]),
            np.array([[10, 0], [10, 19], [0, 10], [19, 10], [19, 19]])
        ]
        polygons = Polygons(contours)
        polygons.add_field('training_tag', [True, False, True])
        window = np.array([[0, 0], [20, 20]])
        polygons.fliter_out(window)
        np.testing.assert_equal(polygons.get_field('training_tag'), np.array([False, False, True]))
        np.testing.assert_equal(polygons.get_contours(), contours)

    def test_fliter_small(self):
        contours = [
            np.array([[2., 1.], [1., 3.], [3., 5.], [4., 2.]]),
            np.array([[0.5, 0.5], [10.5, 0.5], [10.5, 10.5]]),
            np.array([[10, 0], [10, 20], [0, 10], [20, 10], [20, 20]])
        ]
        polygons = Polygons(contours)
        polygons.fliter_small(10)
        np.testing.assert_equal(polygons.get_field('training_tag'), np.array([False, True, True]))
        np.testing.assert_equal(polygons.get_contours(), contours)

    def test_fliter_small_with_exist(self):
        contours = [
            np.array([[2., 1.], [1., 3.], [3., 5.], [4., 2.]]),
            np.array([[0.5, 0.5], [10.5, 0.5], [10.5, 10.5]]),
            np.array([[10, 0], [10, 20], [0, 10], [20, 10], [20, 20]])
        ]
        polygons = Polygons(contours)
        polygons.add_field('training_tag', [True, False, True])
        polygons.fliter_small(10)
        np.testing.assert_equal(polygons.get_field('training_tag'), np.array([False, False, True]))
        np.testing.assert_equal(polygons.get_contours(), contours)

    def test_get(self):
        contours = [
            np.array([[2., 1.], [1., 3.], [3., 5.], [4., 2.]]),
            np.array([[0.5, 0.5], [10.5, 0.5], [10.5, 10.5]]),
            np.array([[10, 0], [10, 20], [0, 10], [20, 10], [20, 20]])
        ]
        polygons = Polygons(contours)
        polygons.add_field('training_tag', [True, False, True])
        polygon = polygons.get(1)
        np.testing.assert_equal(polygon['contour'], np.array([[0.5, 0.5], [10.5, 0.5], [10.5, 10.5]]))
        self.assertEqual(polygon['fields']['training_tag'], False)
        self.assertEqual(polygon['fields']['num_points'], 3)

        polygon['fields']['training_tag'] = True
        polygon['contour'] = np.array([[10, 0], [10, 20], [0, 10], [20, 10], [20, 20]])
        np.testing.assert_equal(polygons.get_contours(),
                                [
                                    np.array([[2., 1.], [1., 3.], [3., 5.], [4., 2.]]),
                                    np.array([[0.5, 0.5], [10.5, 0.5], [10.5, 10.5]]),
                                    np.array([[10, 0], [10, 20], [0, 10], [20, 10], [20, 20]])
                                ]
                                )
        self.assertEqual(polygons.get_field('training_tag'), [True, False, True])

    def test_delete(self):
        contours = [
            np.array([[2., 1.], [1., 3.], [3., 5.], [4., 2.]]),
            np.array([[0.5, 0.5], [10.5, 0.5], [10.5, 10.5]]),
            np.array([[10, 0], [10, 20], [0, 10], [20, 10], [20, 20]])
        ]
        polygons = Polygons(contours)
        polygons.add_field('training_tag', [True, False, True])
        polygons.delete([0, 2])
        np.testing.assert_equal(polygons.get_contours(), [contours[1]])


if __name__ == '__main__':
    unittest.main()
