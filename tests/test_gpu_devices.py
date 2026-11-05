"""Portable UUID selection does not assume the source server's six cards."""
import os
import unittest
from unittest.mock import patch

from ego_annotation.gpu_devices import configure, device


class DeviceTests(unittest.TestCase):
    def test_single_card_is_bound_by_uuid(self):
        with patch('ego_annotation.gpu_devices.subprocess.check_output', return_value='0, GPU-local, P8\n'), \
                patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configure(0), 'GPU-local')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-local')

    def test_fault_is_rejected_before_cuda_environment_changes(self):
        with patch('ego_annotation.gpu_devices.subprocess.check_output', return_value='2, GPU-fault, [GPU requires reset]\n'), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'original'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'requires reset'):
                configure(2)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'original')

    def test_empty_inventory_fails(self):
        with patch('ego_annotation.gpu_devices.subprocess.check_output', return_value=''):
            with self.assertRaisesRegex(RuntimeError, 'No GPU'):
                device(0)
