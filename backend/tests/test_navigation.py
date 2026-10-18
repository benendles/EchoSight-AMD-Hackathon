import os
import unittest
from unittest.mock import patch
import numpy as np

from backend.navigation.engine import find_primary_obstacle, make_navigation_decision
from backend.navigation.spatial import classify_distance
from backend.vision.scene import analyze_scene
from backend.vision.metric_depth import calibrate_depth


def obstacle(label='chair', **kwargs):
    return dict({'class': label, 'confidence': .95, 'direction': 'center', 'vertical_position': 'center',
                 'position': 'directly ahead', 'distance': 'close', 'path_score': .95,
                 'box_area_ratio': .2, 'in_walking_path': True}, **kwargs)


class NavigationTests(unittest.TestCase):
    def test_chair_ahead(self):
        self.assertEqual(make_navigation_decision([obstacle('chair')])['action'], 'STOP')

    def test_bus_ahead(self):
        self.assertEqual(make_navigation_decision([obstacle('bus')])['action'], 'STOP')

    def test_person_ahead(self):
        self.assertEqual(make_navigation_decision([obstacle('person')])['action'], 'STOP')

    def test_bench_and_table(self):
        for label in ['bench', 'dining table', 'suitcase']:
            with self.subTest(label=label):
                self.assertEqual(make_navigation_decision([obstacle(label)])['action'], 'STOP')

    def test_unknown_substantial_object(self):
        self.assertEqual(make_navigation_decision([obstacle('unusual object')])['action'], 'STOP')

    def test_small_unrelated_object(self):
        self.assertEqual(make_navigation_decision([obstacle('cup', box_area_ratio=.002, path_score=.65)])['action'], 'CONTINUE')

    def test_empty_scene(self):
        result = make_navigation_decision([])
        self.assertEqual(result['action'], 'CONTINUE')
        self.assertIn('does not confirm', result['message'])

    def test_missing_depth_is_caution(self):
        self.assertEqual(make_navigation_decision([], depth_available=False)['action'], 'CAUTION')

    def test_unknown_distance_is_caution(self):
        self.assertEqual(make_navigation_decision([obstacle(distance='unknown')])['action'], 'CAUTION')

    def test_medium_does_not_direct_into_unobserved_side(self):
        self.assertEqual(make_navigation_decision([obstacle(distance='medium', direction='left')])['action'], 'CAUTION')

    def test_low_confidence_ignored(self):
        self.assertEqual(make_navigation_decision([obstacle(confidence=.2)])['action'], 'CONTINUE')

    def test_off_path_ignored(self):
        self.assertEqual(make_navigation_decision([obstacle(in_walking_path=False)])['action'], 'CONTINUE')

    def test_prioritizes_urgent_chair_over_distant_bus(self):
        chair = obstacle('chair', path_score=.8)
        bus = obstacle('bus', distance='far', path_score=.9)
        self.assertEqual(find_primary_obstacle([bus, chair])['class'], 'chair')

    def test_close_chair_has_priority_over_larger_medium_bus(self):
        chair = obstacle('chair', path_score=.7)
        bus = obstacle('bus', distance='medium', path_score=1.)
        self.assertEqual(make_navigation_decision([bus, chair])['action'], 'STOP')
        self.assertEqual(find_primary_obstacle([bus, chair])['class'], 'chair')

    def test_missing_depth_scene_fusion(self):
        detections = [{'class': 'chair', 'confidence': .9, 'box': [20, 20, 80, 90], 'direction': 'center', 'vertical_position': 'center'}]
        scene = analyze_scene(detections, None, image_size=(100, 100))
        self.assertEqual(scene[0]['distance'], 'unknown')
        self.assertIsNone(scene[0]['distance_meters'])
        self.assertTrue(scene[0]['in_walking_path'])

    def test_invalid_depth_values(self):
        self.assertEqual(classify_distance(None, np.ones((2, 2))), 'unknown')
        self.assertEqual(classify_distance(1, np.ones((2, 2))), 'unknown')
        self.assertEqual(classify_distance(float('nan'), np.ones((2, 2))), 'unknown')

    def test_uncalibrated_meters_disabled(self):
        with patch.dict(os.environ, {'ECHOSIGHT_METRIC_DEPTH_ENABLED': 'false'}):
            self.assertIsNone(calibrate_depth(3.5))
