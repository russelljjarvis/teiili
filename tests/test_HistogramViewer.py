import unittest

from teili.tools.visualizer.DataViewers import HistogramViewer


class TestHistogramViewer(unittest.TestCase):

    def test_getmostcommonelement(self):
        my_list = [2] * 5 + [3] * 4 + [1] * 8
        self.assertEqual(
            HistogramViewer()._get_most_common_element(my_list), 1)

        my_list = []
        self.assertEqual(
            HistogramViewer()._get_most_common_element(my_list), None)

    def test_gethighestcount(self):
        my_list = [2] * 5 + [3] * 4 + [1] * 8
        self.assertEqual(HistogramViewer().get_highest_count(my_list), 8)

        my_list = []
        self.assertEqual(HistogramViewer().get_highest_count(my_list), 0)


if __name__ == '__main__':
    unittest.main()
